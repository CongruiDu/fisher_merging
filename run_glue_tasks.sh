#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS=("$@")
MODEL_VARIANT="${MODEL_VARIANT:-paper_available}"

if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(rte mrpc sst2)
fi

FISHER_EXAMPLES="${FISHER_EXAMPLES:-4096}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-128}"
MERGE_BASE_MODEL_IDX="${MERGE_BASE_MODEL_IDX:-0}"
FAVOR_MODEL_IDX="${FAVOR_MODEL_IDX:-0}"
FISHER_FLOOR="${FISHER_FLOOR:-1e-8}"
NORMALIZE_FISHERS="${NORMALIZE_FISHERS:-1}"
SAVE_MODELS="${SAVE_MODELS:-0}"

task_models() {
  local task="$1"
  "$PYTHON_BIN" -c \
    "from task_model_registry import get_task_models; import sys; [print(m) for m in get_task_models(sys.argv[1], sys.argv[2])]" \
    "$task" "$MODEL_VARIANT"
}

run_task() {
  local task="$1"
  local task_dir="$ROOT_DIR/runs/$MODEL_VARIANT/$task"
  local fisher_dir="$task_dir/fishers"
  local model_dir="$task_dir/models"
  local log_dir="$task_dir/logs"
  local -a models=()
  local -a fishers=()
  local -a eval_cmd=()
  local model
  local safe_name

  mkdir -p "$fisher_dir" "$model_dir" "$log_dir"

  mapfile -t models < <(task_models "$task")

  echo
  echo "======================================================================"
  echo "Task: $task"
  echo "Model variant: $MODEL_VARIANT"
  echo "Models:"
  printf '  %s\n' "${models[@]}"
  echo "======================================================================"

  for model in "${models[@]}"; do
    safe_name="$(echo "$model" | tr '/' '_' | tr '-' '_')"
    fishers+=("$fisher_dir/${safe_name}.pt")

    echo
    echo "[${task}] Computing Fisher for: $model"
    "$PYTHON_BIN" "$ROOT_DIR/get_fisher.py" \
      --model "$model" \
      --all_models "${models[@]}" \
      --glue_task "$task" \
      --split train \
      --fisher_path "$fisher_dir/${safe_name}.pt" \
      --n_examples "$FISHER_EXAMPLES" \
      --batch_size "$FISHER_BATCH_SIZE" \
      --sequence_length "$SEQUENCE_LENGTH" \
      --shuffle
  done

  eval_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/merge_and_evaluate.py"
    --models "${models[@]}"
    --fishers "${fishers[@]}"
    --glue_task "$task"
    --split validation
    --batch_size "$EVAL_BATCH_SIZE"
    --sequence_length "$SEQUENCE_LENGTH"
    --merge_base_model_idx "$MERGE_BASE_MODEL_IDX"
    --favor_model_idx "$FAVOR_MODEL_IDX"
    --fisher_floor "$FISHER_FLOOR"
  )

  if [ "$NORMALIZE_FISHERS" = "1" ]; then
    eval_cmd+=(--normalize_fishers)
  else
    eval_cmd+=(--no_normalize_fishers)
  fi

  if [ "$SAVE_MODELS" = "1" ]; then
    eval_cmd+=(
      --save_isotropic_model_path "$model_dir/isotropic_merged_model"
      --save_fisher_model_path "$model_dir/fisher_merged_model"
    )
  fi

  echo
  echo "[${task}] Running merge_and_evaluate.py"
  "${eval_cmd[@]}" | tee "$log_dir/eval.log"
}

cd "$ROOT_DIR"
for task in "${TASKS[@]}"; do
  run_task "$task"
done
