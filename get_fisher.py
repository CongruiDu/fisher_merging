import argparse
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--all_models",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument("--glue_task", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fisher_path", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--backbone_only", action="store_true")
    parser.add_argument("--normalize_fisher", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--low_cpu_mem_usage", action="store_true")

    return parser.parse_args()


def resolve_glue_task_and_split(task_name, split):
    task_name = task_name.lower()
    if task_name not in TASK_TO_KEYS:
        raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {sorted(TASK_TO_KEYS.keys())}")

    if task_name == "mnli_matched":
        dataset_name = "mnli"
        if split == "validation":
            split = "validation_matched"
        elif split == "test":
            split = "test_matched"
        return dataset_name, split

    if task_name == "mnli_mismatched":
        dataset_name = "mnli"
        if split == "validation":
            split = "validation_mismatched"
        elif split == "test":
            split = "test_mismatched"
        return dataset_name, split

    return task_name, split


def get_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_model_for_shapes(model_name_or_path, low_cpu_mem_usage= False):
    kwargs = {}
    if low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return model


def build_tokenize_fn(tokenizer, task_name, max_length):
    key1, key2 = TASK_TO_KEYS[task_name]

    def tokenize_fn(examples):
        if key2 is None:
            return tokenizer(
                examples[key1],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples[key1],
            examples[key2],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return tokenize_fn


def load_glue_dataloader(
    task_name,
    split,
    tokenizer,
    max_length,
    batch_size,
    n_examples,
    shuffle,
    num_workers,
    pin_memory,
):
    actual_task, actual_split = resolve_glue_task_and_split(task_name, split)
    ds = load_dataset("glue", actual_task, split=actual_split)

    if n_examples is not None and n_examples > 0:
        n = min(n_examples, len(ds))
        ds = ds.shuffle(seed=42).select(range(n)) if shuffle else ds.select(range(n))

    tokenize_fn = build_tokenize_fn(tokenizer, task_name, max_length)
    ds = ds.map(tokenize_fn, batched=True)

    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    if "label" in ds.column_names:
        cols.append("label")

    ds.set_format(type="torch", columns=cols)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, len(ds)


def maybe_cast_tensor(t, dtype_name):
    if dtype_name == "float16":
        return t.half()
    return t.float()


def l2_normalize_fisher(fisher):
    total_sq = 0.0
    for v in fisher.values():
        total_sq += torch.sum(v.float() ** 2).item()
    denom = math.sqrt(total_sq) + 1e-12
    return {k: (v / denom) for k, v in fisher.items()}


def get_shared_mergeable_param_names(
    current_model,
    all_model_names,
    backbone_only= True,
    low_cpu_mem_usage = False,
):
    current_named_params = dict(current_model.named_parameters())
    prefix = current_model.base_model_prefix + "." if backbone_only else None

    candidate_shapes = OrderedDict()
    for name, param in current_named_params.items():
        if not param.requires_grad:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        candidate_shapes[name] = tuple(param.shape)

    shared_names = set(candidate_shapes.keys())
    for model_name in all_model_names:
        other_model = load_model_for_shapes(model_name, low_cpu_mem_usage=low_cpu_mem_usage)
        other_state = other_model.state_dict()

        valid_here = set()
        for name in shared_names:
            if name in other_state and tuple(other_state[name].shape) == candidate_shapes[name]:
                valid_here.add(name)

        shared_names = shared_names.intersection(valid_here)
        del other_model
    ordered = [name for name in candidate_shapes.keys() if name in shared_names]
    return ordered


def compute_exact_diagonal_fisher_shared_only(
    model,
    dataloader,
    device,
    shared_param_names,
    progress_every=50,
):
    model.eval()
    model.to(device)

    named_params = dict(model.named_parameters())
    fisher = {
        name: torch.zeros_like(named_params[name], device=device)
        for name in shared_param_names
    }

    n_examples = 0
    num_labels = model.config.num_labels

    for batch_idx, batch in enumerate(dataloader):
        batch_size = batch["input_ids"].size(0)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        for i in range(batch_size):
            example_inputs = {
                "input_ids": input_ids[i:i + 1],
                "attention_mask": attention_mask[i:i + 1],
            }
            if token_type_ids is not None:
                example_inputs["token_type_ids"] = token_type_ids[i:i + 1]

            outputs = model(**example_inputs)
            logits = outputs.logits.squeeze(0)                  
            log_probs = torch.log_softmax(logits, dim=-1)       
            probs = torch.softmax(logits.detach(), dim=-1)      

            # E_y [(grad log p(y|x))^2]
            for c in range(num_labels):
                model.zero_grad(set_to_none=True)
                retain_graph = (c != num_labels - 1)
                log_probs[c].backward(retain_graph=retain_graph)

                with torch.no_grad():
                    pc = probs[c]
                    for name in shared_param_names:
                        grad = named_params[name].grad
                        if grad is not None:
                            fisher[name] += pc * (grad.detach() ** 2)

            n_examples += 1

        if progress_every > 0 and (batch_idx + 1) % progress_every == 0:
            print(f"[progress] processed {n_examples} examples")

    with torch.no_grad():
        for name in fisher:
            fisher[name] /= max(n_examples, 1)

    return fisher


def save_fisher_pt(
    fisher,
    metadata,
    path,
    save_dtype,
):
    obj = {
        "metadata": metadata,
        "fisher": {
            k: maybe_cast_tensor(v.detach().cpu(), save_dtype)
            for k, v in fisher.items()
        }
    }
    torch.save(obj, path)


def main():
    args = parse_args()

    if not args.fisher_path.endswith(".pt"):
        raise ValueError("This script only saves .pt files. Please set --fisher_path to end with .pt")

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.fisher_path) or ".", exist_ok=True)

    all_models = unique_preserve_order(args.all_models)
    if args.model not in all_models:
        all_models = [args.model] + all_models

    print("=" * 72)
    print("compute_fisher.py (shared / mergeable params across ALL merge models)")
    print("=" * 72)
    print(f"Current model       : {args.model}")
    print(f"All merge models    : {len(all_models)}")
    for i, m in enumerate(all_models):
        print(f"  [{i}] {m}")
    print(f"Task                : {args.glue_task}")
    print(f"Split               : {args.split}")
    print(f"Output              : {args.fisher_path}")
    print(f"N examples          : {args.n_examples}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Sequence length     : {args.sequence_length}")
    print(f"Device              : {device}")
    print(f"Backbone only       : {args.backbone_only}")
    print(f"Normalize Fisher    : {args.normalize_fisher}")
    print(f"Save dtype          : {args.save_dtype}")
    print("=" * 72)

    print("Loading tokenizer and current model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    current_model = load_model_for_shapes(args.model, low_cpu_mem_usage=args.low_cpu_mem_usage)

    print("Finding shared / mergeable parameter intersection across all models...")
    shared_param_names = get_shared_mergeable_param_names(
        current_model=current_model,
        all_model_names=all_models,
        backbone_only=args.backbone_only,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if len(shared_param_names) == 0:
        raise RuntimeError("No shared / mergeable parameters found. Check model list and backbone_only setting.")

    print(f"Found {len(shared_param_names)} shared / mergeable parameters.")

    print("Building dataloader...")
    dataloader, actual_n_examples = load_glue_dataloader(
        task_name=args.glue_task.lower(),
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.sequence_length,
        batch_size=args.batch_size,
        n_examples=args.n_examples,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print(f"Loaded {actual_n_examples} examples from dataset.")

    print("Computing exact diagonal Fisher on shared params only...")
    fisher = compute_exact_diagonal_fisher_shared_only(
        model=current_model,
        dataloader=dataloader,
        device=device,
        shared_param_names=shared_param_names,
        progress_every=args.progress_every,
    )

    if args.normalize_fisher:
        print("Applying global L2 normalization to Fisher...")
        fisher = l2_normalize_fisher(fisher)

    metadata = {
        "model": args.model,
        "all_models": all_models,
        "glue_task": args.glue_task,
        "split": args.split,
        "n_examples": actual_n_examples,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "backbone_only": args.backbone_only,
        "normalize_fisher": args.normalize_fisher,
        "save_dtype": args.save_dtype,
        "num_labels": int(current_model.config.num_labels),
        "num_shared_params": len(shared_param_names),
        "shared_param_names": shared_param_names,
    }

    print("Saving .pt file...")
    save_fisher_pt(
        fisher=fisher,
        metadata=metadata,
        path=args.fisher_path,
        save_dtype=args.save_dtype,
    )

    print(f"Done. Saved Fisher to: {args.fisher_path}")


if __name__ == "__main__":
    main()