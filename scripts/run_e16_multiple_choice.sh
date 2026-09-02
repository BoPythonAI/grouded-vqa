#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
DATASET="${PROJECT_ROOT}/data/vqav2/multiple_choice/vqav2_val_mcq_5000_seed42.jsonl"
LOG="${VQA_LOG_ROOT}/e16_multiple_choice_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E16 multiple-choice evaluation started at $(date --iso-8601=seconds)"

grounded-vqa-build-mcq \
  --split val \
  --max-samples 5000 \
  --max-options 4 \
  --seed 42 \
  --output-file "${DATASET}"

grounded-vqa-eval-mcq \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --dataset "${DATASET}" \
  --quantization 4bit \
  --max-new-tokens 4 \
  --resume \
  --output-name E16a_instructblip_zeroshot_mcq5000

grounded-vqa-eval-mcq \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --dataset "${DATASET}" \
  --quantization 4bit \
  --adapter "${E6}" \
  --max-new-tokens 4 \
  --resume \
  --output-name E16b_E6_instructblip_mcq5000

echo "E16 multiple-choice evaluation completed at $(date --iso-8601=seconds)"
