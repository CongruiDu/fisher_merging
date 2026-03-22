mkdir -p fishers

MODELS=(
  "textattack/bert-base-uncased-RTE"
  "yoshitomo-matsubara/bert-base-uncased-rte"
  "Ruizhou/bert-base-uncased-finetuned-rte"
  "howey/bert-base-uncased-rte"
  "anirudh21/bert-base-uncased-finetuned-rte"
)

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
  python get_fisher.py \
    --model "$MODEL" \
    --all_models "${MODELS[@]}" \
    --glue_task rte \
    --split train \
    --fisher_path "fishers/${SAFE_NAME}.pt" \
    --n_examples 4096 \
    --batch_size 1 \
    --sequence_length 128 \
    --shuffle
done