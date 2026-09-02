#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
ADAPTER="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
PARQUET_ROOT="${PROJECT_ROOT}/data/hallusionbench/hf/data"
COMMON=(
  --model-id "${MODEL}"
  --model-kind instructblip
  --parquet-root "${PARQUET_ROOT}"
  --quantization 4bit
  --batch-size 8
  --max-new-tokens 5
  --resume
)

grounded-vqa-eval-hallusionbench \
  "${COMMON[@]}" \
  --output-name H3_InstructBLIP_zeroshot_HallusionBench

grounded-vqa-eval-hallusionbench \
  "${COMMON[@]}" \
  --adapter "${ADAPTER}" \
  --output-name H3_E6_HallusionBench

grounded-vqa-compare-hallusionbench \
  --baseline "${VQA_OUTPUT_ROOT}/H3_InstructBLIP_zeroshot_HallusionBench" \
  --candidate "${VQA_OUTPUT_ROOT}/H3_E6_HallusionBench" \
  --output "${VQA_OUTPUT_ROOT}/H3_E6_HallusionBench/paired_vs_zeroshot.json"
