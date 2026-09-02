#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
LOG="${VQA_LOG_ROOT}/h1_pope_zeroshot_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "H1 zero-shot POPE control started at $(date --iso-8601=seconds)"

grounded-vqa-eval-pope \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --pope-root "${PROJECT_ROOT}/data/pope" \
  --image-root "${VQA_DATA_ROOT}/val2014" \
  --quantization 4bit \
  --batch-size 8 \
  --max-new-tokens 5 \
  --resume \
  --output-name H1_InstructBLIP_zeroshot_official_COCO_POPE

echo "H1 zero-shot POPE control completed at $(date --iso-8601=seconds)"
