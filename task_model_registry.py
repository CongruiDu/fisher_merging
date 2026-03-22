#!/usr/bin/env python3

from typing import Dict, List


PAPER_EXACT_TASK_MODELS: Dict[str, List[str]] = {
    "rte": [
        "textattack/bert-base-uncased-RTE",
        "yoshitomo-matsubara/bert-base-uncased-rte",
        "Ruizhou/bert-base-uncased-finetuned-rte",
        "howey/bert-base-uncased-rte",
        "anirudh21/bert-base-uncased-finetuned-rte",
    ],
    "mrpc": [
        "textattack/bert-base-uncased-MRPC",
        "yoshitomo-matsubara/bert-base-uncased-mrpc",
        "Maelstrom77/bert-base-uncased-MRPC",
        "Ruizhou/bert-base-uncased-finetuned-mrpc",
        "TehranNLP-org/bert-base-uncased-mrpc-2e-5-42",
    ],
    "sst2": [
        "aviator-neural/bert-base-uncased-sst2",
        "howey/bert-base-uncased-sst2",
        "yoshitomo-matsubara/bert-base-uncased-sst2",
        "ikevin98/bert-base-uncased-finetuned-sst2",
        "TehranNLP-org/bert-base-uncased-cls-sst2",
    ],
}

# The paper's exact MRPC list contains one checkpoint that is no longer available
# on Hugging Face as of 2026-03-12, so the available variant keeps the remaining
# exact checkpoints and drops the missing one.
PAPER_AVAILABLE_TASK_MODELS: Dict[str, List[str]] = {
    "rte": PAPER_EXACT_TASK_MODELS["rte"],
    "mrpc": [
        "textattack/bert-base-uncased-MRPC",
        "yoshitomo-matsubara/bert-base-uncased-mrpc",
        "Ruizhou/bert-base-uncased-finetuned-mrpc",
        "TehranNLP-org/bert-base-uncased-mrpc-2e-5-42",
    ],
    "sst2": PAPER_EXACT_TASK_MODELS["sst2"],
}

MODEL_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    "paper_exact": PAPER_EXACT_TASK_MODELS,
    "paper_available": PAPER_AVAILABLE_TASK_MODELS,
}


def get_task_models(task: str, variant: str = "paper_available") -> List[str]:
    task = task.lower()
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model variant: {variant}. Available: {sorted(MODEL_VARIANTS.keys())}")
    task_models = MODEL_VARIANTS[variant]
    if task not in task_models:
        raise ValueError(f"No model registry for task={task} in variant={variant}")
    return list(task_models[task])
