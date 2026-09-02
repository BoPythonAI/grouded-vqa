#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/server_env.sh"

MODEL_ID="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
E15="${VQA_OUTPUT_ROOT}/E15_E6_instructblip_full_vqav2_val214354/predictions.jsonl"

grounded-vqa-rerank-short \
  --model-id "${MODEL_ID}" \
  --model-kind instructblip \
  --adapter "${E6}" \
  --predictions "${E15}" \
  --split val \
  --quantization 4bit \
  --batch-size 8 \
  --prompt-style short \
  --resume \
  --output-name E19_E6_short_answer_rerank_full_val
