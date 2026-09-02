#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
HARD_PAIRS="${VQA_OUTPUT_ROOT}/E10_hard_pair_mining_e6_candidates10k_select500/selected_pairs.json"
CONTROL="E10a_hardpair_control_lr2e5_500_from_e6"
TREATMENT="E10b_hardpair_logprob_w02_t05_lr2e5_500_from_e6"
LOG="${VQA_LOG_ROOT}/e10_hard_pairs_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E10 hard-pair sequence started at $(date --iso-8601=seconds)"
sha256sum "${HARD_PAIRS}"

grounded-vqa-train-complementary \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter-init "${E6}" \
  --pairs-file "${HARD_PAIRS}" \
  --quantization 4bit \
  --learning-rate 2e-5 \
  --contrastive-weight 0 \
  --temperature 0.5 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --max-pairs 500 \
  --num-workers 4 \
  --seed 44 \
  --output-name "${CONTROL}" \
  --log-every 10 \
  --save-every 250

grounded-vqa-train-complementary \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter-init "${E6}" \
  --pairs-file "${HARD_PAIRS}" \
  --quantization 4bit \
  --learning-rate 2e-5 \
  --contrastive-weight 0.2 \
  --temperature 0.5 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --max-pairs 500 \
  --num-workers 4 \
  --seed 44 \
  --output-name "${TREATMENT}" \
  --log-every 10 \
  --save-every 250

for SPEC in \
  "${CONTROL}:E10a_hardpair_control" \
  "${TREATMENT}:E10b_hardpair_logprob"
do
  ADAPTER_RUN="${SPEC%%:*}"
  PREFIX="${SPEC##*:}"

  grounded-vqa-predict \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${VQA_OUTPUT_ROOT}/${ADAPTER_RUN}/final-adapter" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${PREFIX}_eval1000_shortprompt"

  grounded-vqa-diagnose-alignment \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${VQA_OUTPUT_ROOT}/${ADAPTER_RUN}/final-adapter" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${PREFIX}_alignment_diag_val1000"

  grounded-vqa-mine-complementary \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${VQA_OUTPUT_ROOT}/${ADAPTER_RUN}/final-adapter" \
    --split val \
    --quantization 4bit \
    --candidate-pairs 1000 \
    --selected-pairs 1000 \
    --batch-size 16 \
    --num-workers 4 \
    --seed 42 \
    --output-name "${PREFIX}_val_complementary1000_logprob"
done

echo "E10 hard-pair sequence completed at $(date --iso-8601=seconds)"
