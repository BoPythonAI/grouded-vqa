#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
HARD_PAIRS="${VQA_OUTPUT_ROOT}/E10_hard_pair_mining_e6_candidates10k_select500/selected_pairs.json"
CONTROL="E11a_mixed70_30_qformer_control_from_e6"
TREATMENT="E11b_mixed70_30_qformer_logprob_w02_from_e6"
LOG="${VQA_LOG_ROOT}/e11_mixed_qformer_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E11 mixed Q-Former sequence started at $(date --iso-8601=seconds)"
sha256sum "${HARD_PAIRS}"

for SPEC in "${CONTROL}:0" "${TREATMENT}:0.2"
do
  RUN_NAME="${SPEC%%:*}"
  WEIGHT="${SPEC##*:}"
  grounded-vqa-train-mixed-qformer \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter-init "${E6}" \
    --hard-pairs-file "${HARD_PAIRS}" \
    --quantization 4bit \
    --qformer-rank 8 \
    --qformer-alpha 16 \
    --qformer-dropout 0.05 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --contrastive-weight "${WEIGHT}" \
    --temperature 0.5 \
    --ordinary-examples 700 \
    --hard-pairs 150 \
    --validation-pairs 128 \
    --batch-size 1 \
    --validation-batch-size 8 \
    --gradient-accumulation 2 \
    --epochs 1 \
    --num-workers 4 \
    --seed 45 \
    --eval-every 50 \
    --early-stop-patience 3 \
    --early-stop-min-delta 0 \
    --output-name "${RUN_NAME}" \
    --log-every 10 \
    --save-every 250
done

for SPEC in \
  "${CONTROL}:E11a_control" \
  "${TREATMENT}:E11b_logprob"
do
  RUN_NAME="${SPEC%%:*}"
  PREFIX="${SPEC##*:}"
  for CHECKPOINT in best final
  do
    ADAPTER="${VQA_OUTPUT_ROOT}/${RUN_NAME}/${CHECKPOINT}-adapter"
    grounded-vqa-predict \
      --model-id "${MODEL}" \
      --model-kind instructblip \
      --quantization 4bit \
      --adapter "${ADAPTER}" \
      --split val \
      --max-samples 1000 \
      --seed 42 \
      --max-new-tokens 10 \
      --prompt-style short \
      --output-name "${PREFIX}_${CHECKPOINT}_eval1000_shortprompt"

    grounded-vqa-mine-complementary \
      --model-id "${MODEL}" \
      --model-kind instructblip \
      --adapter "${ADAPTER}" \
      --split val \
      --quantization 4bit \
      --candidate-pairs 1000 \
      --selected-pairs 1000 \
      --batch-size 16 \
      --num-workers 4 \
      --seed 42 \
      --output-name "${PREFIX}_${CHECKPOINT}_val_complementary1000"
  done

  grounded-vqa-diagnose-alignment \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${VQA_OUTPUT_ROOT}/${RUN_NAME}/best-adapter" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${PREFIX}_best_alignment_diag_val1000"
done

echo "E11 mixed Q-Former sequence completed at $(date --iso-8601=seconds)"
