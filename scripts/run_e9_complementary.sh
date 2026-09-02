#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
CONTROL="E9a_instructblip_complementary_control_500pairs_from_e6"
TREATMENT="E9b_instructblip_complementary_logprob_w01_t05_500pairs_from_e6"
LOG="${VQA_LOG_ROOT}/e9_complementary_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E9 complementary-pair sequence started at $(date --iso-8601=seconds)"

grounded-vqa-train-complementary \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter-init "${E6}" \
  --quantization 4bit \
  --learning-rate 5e-5 \
  --contrastive-weight 0 \
  --temperature 0.5 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --max-pairs 500 \
  --num-workers 4 \
  --seed 43 \
  --output-name "${CONTROL}" \
  --log-every 10 \
  --save-every 250

grounded-vqa-train-complementary \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter-init "${E6}" \
  --quantization 4bit \
  --learning-rate 5e-5 \
  --contrastive-weight 0.1 \
  --temperature 0.5 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --max-pairs 500 \
  --num-workers 4 \
  --seed 43 \
  --output-name "${TREATMENT}" \
  --log-every 10 \
  --save-every 250

grounded-vqa-predict \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --quantization 4bit \
  --adapter "${VQA_OUTPUT_ROOT}/${CONTROL}/final-adapter" \
  --split val \
  --max-samples 1000 \
  --seed 42 \
  --max-new-tokens 10 \
  --prompt-style short \
  --output-name E9a_control_eval1000_shortprompt

grounded-vqa-predict \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --quantization 4bit \
  --adapter "${VQA_OUTPUT_ROOT}/${TREATMENT}/final-adapter" \
  --split val \
  --max-samples 1000 \
  --seed 42 \
  --max-new-tokens 10 \
  --prompt-style short \
  --output-name E9b_complementary_eval1000_shortprompt

grounded-vqa-diagnose-alignment \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --quantization 4bit \
  --adapter "${VQA_OUTPUT_ROOT}/${CONTROL}/final-adapter" \
  --split val \
  --max-samples 1000 \
  --seed 42 \
  --max-new-tokens 10 \
  --prompt-style short \
  --output-name E9a_control_alignment_diag_val1000

grounded-vqa-diagnose-alignment \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --quantization 4bit \
  --adapter "${VQA_OUTPUT_ROOT}/${TREATMENT}/final-adapter" \
  --split val \
  --max-samples 1000 \
  --seed 42 \
  --max-new-tokens 10 \
  --prompt-style short \
  --output-name E9b_complementary_alignment_diag_val1000

echo "E9 complementary-pair sequence completed at $(date --iso-8601=seconds)"
