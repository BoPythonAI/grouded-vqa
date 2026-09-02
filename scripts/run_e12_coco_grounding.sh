#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
GROUNDING_TRAIN="${PROJECT_ROOT}/data/coco/grounding/train_grounding_seed47.jsonl"
GROUNDING_VAL="${PROJECT_ROOT}/data/coco/grounding/val_grounding_seed47.jsonl"
CONTROL="E12a_coco_grounding_control_from_e6"
TREATMENT="E12b_coco_grounding_70_30_from_e6"
LOG="${VQA_LOG_ROOT}/e12_coco_grounding_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E12 COCO grounding sequence started at $(date --iso-8601=seconds)"
sha256sum "${GROUNDING_TRAIN}" "${GROUNDING_VAL}"

# Both arms see 1,000 training examples and share the same first 700 ordinary
# VQAv2 examples. The treatment replaces the remaining 300 ordinary examples
# with COCO existence/counting supervision.
for SPEC in "${CONTROL}:1000:0" "${TREATMENT}:700:300"
do
  RUN_NAME="${SPEC%%:*}"
  REMAINDER="${SPEC#*:}"
  ORDINARY="${REMAINDER%%:*}"
  GROUNDING="${REMAINDER##*:}"
  grounded-vqa-train-grounding-qformer \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter-init "${E6}" \
    --grounding-train "${GROUNDING_TRAIN}" \
    --grounding-val "${GROUNDING_VAL}" \
    --quantization 4bit \
    --qformer-rank 8 \
    --qformer-alpha 16 \
    --qformer-dropout 0.05 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --ordinary-pool-size 1000 \
    --ordinary-examples "${ORDINARY}" \
    --grounding-examples "${GROUNDING}" \
    --validation-grounding-examples 128 \
    --batch-size 1 \
    --validation-batch-size 8 \
    --gradient-accumulation 4 \
    --epochs 1 \
    --num-workers 4 \
    --seed 48 \
    --eval-every 50 \
    --early-stop-patience 3 \
    --early-stop-min-delta 0 \
    --output-name "${RUN_NAME}" \
    --log-every 10 \
    --save-every 250
done

for SPEC in "${CONTROL}:E12a_control" "${TREATMENT}:E12b_grounding"
do
  RUN_NAME="${SPEC%%:*}"
  PREFIX="${SPEC##*:}"
  for CHECKPOINT in best final
  do
    ADAPTER="${VQA_OUTPUT_ROOT}/${RUN_NAME}/${CHECKPOINT}-adapter"
    grounded-vqa-eval-grounding \
      --model-id "${MODEL}" \
      --model-kind instructblip \
      --adapter "${ADAPTER}" \
      --dataset "${GROUNDING_VAL}" \
      --quantization 4bit \
      --batch-size 8 \
      --max-new-tokens 10 \
      --prompt-style short \
      --output-name "${PREFIX}_${CHECKPOINT}_grounding_val1000"

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
      --output-name "${PREFIX}_${CHECKPOINT}_vqa_val1000"

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
      --output-name "${PREFIX}_${CHECKPOINT}_complementary_val1000"
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

echo "E12 COCO grounding sequence completed at $(date --iso-8601=seconds)"
