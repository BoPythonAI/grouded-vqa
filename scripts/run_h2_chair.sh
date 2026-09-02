#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
ADAPTER="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
SELECTION="${VQA_OUTPUT_ROOT}/H2_CHAIR_selection_seed42.json"
COMMON=(
  --model-id "${MODEL}"
  --model-kind instructblip
  --annotation-root "${PROJECT_ROOT}/data/coco/annotations"
  --image-root "${VQA_DATA_ROOT}/val2014"
  --selection-file "${SELECTION}"
  --sample-size 500
  --seed 42
  --quantization 4bit
  --batch-size 8
  --max-new-tokens 64
  --resume
)

grounded-vqa-eval-chair \
  "${COMMON[@]}" \
  --output-name H2_InstructBLIP_zeroshot_CHAIR500

grounded-vqa-eval-chair \
  "${COMMON[@]}" \
  --adapter "${ADAPTER}" \
  --output-name H2_E6_CHAIR500

grounded-vqa-compare-chair \
  --baseline "${VQA_OUTPUT_ROOT}/H2_InstructBLIP_zeroshot_CHAIR500" \
  --candidate "${VQA_OUTPUT_ROOT}/H2_E6_CHAIR500" \
  --output "${VQA_OUTPUT_ROOT}/H2_E6_CHAIR500/paired_vs_zeroshot.json"
