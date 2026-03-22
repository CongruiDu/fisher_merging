import argparse
import copy
import math
from collections import OrderedDict

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

    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--fishers", type=str, nargs="+", required=True)
    parser.add_argument("--glue_task", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backbone_only", action="store_true")
    parser.add_argument("--merge_base_model_idx", type=int, default=0)
    parser.add_argument("--fisher_floor", type=float, default=1e-8)
    parser.set_defaults(normalize_fishers=True)
    parser.add_argument(
        "--normalize_fishers",
        dest="normalize_fishers",
        action="store_true",
    )
    parser.add_argument(
        "--no_normalize_fishers",
        dest="normalize_fishers",
        action="store_false",
    )
    parser.add_argument("--save_isotropic_model_path", type=str, default=None)
    parser.add_argument("--save_fisher_model_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--favor_model_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ensemble_average",
        type=str,
        choices=["logits", "probs"],
        default="logits",
    )

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


def load_model(model_name_or_path, low_cpu_mem_usage=False):
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


def build_eval_loader(
    task_name,
    split,
    tokenizer,
    batch_size,
    max_length,
    num_workers,
    pin_memory,
):
    actual_task, actual_split = resolve_glue_task_and_split(task_name, split)
    ds = load_dataset("glue", actual_task, split=actual_split)

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


def load_fisher_pt(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "fisher" not in obj:
        raise ValueError(f"Invalid fisher file: {path}")
    fisher = obj["fisher"]
    if not isinstance(fisher, dict):
        raise ValueError(f"Invalid fisher dict in: {path}")
    return fisher


def l2_normalize_fisher(fisher):
    total_sq = 0.0
    for v in fisher.values():
        total_sq += torch.sum(v.float() ** 2).item()
    denom = math.sqrt(total_sq) + 1e-12
    return {k: (v / denom) for k, v in fisher.items()}


def validate_lambdas(lambdas, n_models):
    if lambdas is None:
        return [1.0 / n_models] * n_models
    if len(lambdas) != n_models:
        raise ValueError(f"Expected {n_models} lambdas, got {len(lambdas)}")
    s = sum(lambdas)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Lambdas must sum to 1. Got sum={s}")
    return lambdas


def get_shared_param_names(models, backbone_only=True):
    if len(models) < 2:
        raise ValueError("Need at least 2 models")

    ref_model = models[0]
    ref_named_params = OrderedDict(ref_model.named_parameters())
    prefix = ref_model.base_model_prefix + "." if backbone_only else None

    shared = []
    state_dicts = [m.state_dict() for m in models]

    for name, param in ref_named_params.items():
        if not param.requires_grad:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue

        ok = True
        shape0 = tuple(param.shape)
        for sd in state_dicts[1:]:
            if name not in sd or tuple(sd[name].shape) != shape0:
                ok = False
                break

        if ok:
            shared.append(name)

    return shared


def intersect_shared_and_fisher_keys(
    shared_param_names,
    fishers,
):
    fisher_key_sets = [set(f.keys()) for f in fishers]
    common_fisher_keys = set(shared_param_names)
    for ks in fisher_key_sets:
        common_fisher_keys &= ks

    ordered = [name for name in shared_param_names if name in common_fisher_keys]
    return ordered


def isotropic_merge_many(
    models,
    lambdas,
    merge_param_names,
    merge_base_model_idx=0,
):
    merged = copy.deepcopy(models[merge_base_model_idx])
    merged_state = merged.state_dict()
    state_dicts = [m.state_dict() for m in models]

    with torch.no_grad():
        for name in merge_param_names:
            merged_param = torch.zeros_like(state_dicts[0][name])
            for lam, sd in zip(lambdas, state_dicts):
                merged_param += lam * sd[name]
            merged_state[name].copy_(merged_param)

    merged.load_state_dict(merged_state)
    return merged


def fisher_merge_many(
    models,
    fishers,
    lambdas,
    merge_param_names,
    merge_base_model_idx=0,
    fisher_floor=1e-8,
    favor_model_idx=0,
):
    merged = copy.deepcopy(models[merge_base_model_idx])
    merged_state = merged.state_dict()
    state_dicts = [m.state_dict() for m in models]

    with torch.no_grad():
        for name in merge_param_names:
            numerator = torch.zeros_like(state_dicts[0][name], dtype=state_dicts[0][name].dtype)
            denom = torch.zeros_like(state_dicts[0][name], dtype=state_dicts[0][name].dtype)

            for lam, sd, fdict in zip(lambdas, state_dicts, fishers):
                Fi = fdict[name].to(sd[name].dtype).to(sd[name].device)
                numerator += lam * Fi * sd[name]
                denom += lam * Fi

            low_mask = denom < fisher_floor
            safe_denom = torch.where(low_mask, torch.ones_like(denom), denom)
            merged_param = numerator / safe_denom

            fallback_param = state_dicts[favor_model_idx][name]
            merged_param = torch.where(low_mask, fallback_param, merged_param)

            merged_state[name].copy_(merged_param)

    merged.load_state_dict(merged_state)
    return merged


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        labels = batch["label"].to(device)

        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ensemble_accuracy(
    models,
    dataloader,
    device,
    average_mode="logits",
):
    for model in models:
        model.eval()
        model.to(device)

    correct = 0
    total = 0

    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        labels = batch["label"].to(device)
        outputs = [model(**inputs).logits for model in models]

        if average_mode == "probs":
            merged = torch.stack([torch.softmax(logits, dim=-1) for logits in outputs], dim=0).mean(dim=0)
        else:
            merged = torch.stack(outputs, dim=0).mean(dim=0)

        preds = torch.argmax(merged, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    for model in models:
        model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return correct / max(total, 1)


def maybe_save_model(model, path):
    if path is None:
        return
    model.save_pretrained(path)


def main():
    args = parse_args()

    if len(args.models) != len(args.fishers):
        raise ValueError("Number of models must equal number of fishers")

    device = get_device(args.device)
    lambdas = validate_lambdas(args.lambdas, len(args.models))

    print("=" * 80)
    print(f"Task                 : {args.glue_task}")
    tokenizer = AutoTokenizer.from_pretrained(args.models[0])

    print("Loading models...")
    models = [load_model(m, low_cpu_mem_usage=args.low_cpu_mem_usage) for m in args.models]

    print("Loading fishers...")
    fishers = [load_fisher_pt(p) for p in args.fishers]
    if args.normalize_fishers:
        print("Normalizing fishers...")
        fishers = [l2_normalize_fisher(f) for f in fishers]

    print("Building eval dataloader...")
    eval_loader, n_eval = build_eval_loader(
        task_name=args.glue_task.lower(),
        split=args.split,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.sequence_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print(f"Loaded {n_eval} evaluation examples.")

    print("Finding shared mergeable params across models...")
    shared_param_names = get_shared_param_names(models, backbone_only=args.backbone_only)
    print(f"Shared params for isotropic merge       : {len(shared_param_names)}")

    fisher_merge_param_names = intersect_shared_and_fisher_keys(shared_param_names, fishers)
    print(f"Shared params for fisher merge          : {len(fisher_merge_param_names)}")

    if len(shared_param_names) == 0:
        raise RuntimeError("No shared merge parameters found across models.")

    if len(fisher_merge_param_names) == 0:
        raise RuntimeError("No common merge parameters found across models and fishers.")

    print(f"Running ensemble ({args.ensemble_average})...")
    ensemble_acc = evaluate_ensemble_accuracy(
        models=models,
        dataloader=eval_loader,
        device=device,
        average_mode=args.ensemble_average,
    )
    print(f"Ensemble ({args.ensemble_average}) acc : {ensemble_acc:.4f}")

    print("Running isotropic merge...")
    isotropic_model = isotropic_merge_many(
        models=models,
        lambdas=lambdas,
        merge_param_names=shared_param_names,
        merge_base_model_idx=args.merge_base_model_idx,
    )
    isotropic_acc = evaluate_accuracy(isotropic_model, eval_loader, device)
    print(f"Isotropic merged acc: {isotropic_acc:.4f}")

    print("Running fisher merge...")
    fisher_model = fisher_merge_many(
        models=models,
        fishers=fishers,
        lambdas=lambdas,
        merge_param_names=fisher_merge_param_names,
        merge_base_model_idx=args.merge_base_model_idx,
        fisher_floor=args.fisher_floor,
        favor_model_idx=args.favor_model_idx,
    )
    fisher_acc = evaluate_accuracy(fisher_model, eval_loader, device)
    print(f"Fisher merged acc   : {fisher_acc:.4f}")

    maybe_save_model(isotropic_model, args.save_isotropic_model_path)
    maybe_save_model(fisher_model, args.save_fisher_model_path)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Ensemble ({args.ensemble_average})    : {ensemble_acc:.4f}")
    print(f"Isotropic merged      : {isotropic_acc:.4f}")
    print(f"Fisher merged         : {fisher_acc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
