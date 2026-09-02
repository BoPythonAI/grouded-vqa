#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
OUTPUT_NAME="E15_E6_instructblip_full_vqav2_val214354"
LOG="${VQA_LOG_ROOT}/e15_full_validation_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E15 full VQAv2 validation started at $(date --iso-8601=seconds)"
echo "Output: ${VQA_OUTPUT_ROOT}/${OUTPUT_NAME}"
echo "Questions: 214354"

grounded-vqa-predict \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --quantization 4bit \
  --adapter "${E6}" \
  --split val \
  --seed 42 \
  --max-new-tokens 10 \
  --prompt-style short \
  --resume \
  --output-name "${OUTPUT_NAME}"

echo "E15 full VQAv2 validation completed at $(date --iso-8601=seconds)"
