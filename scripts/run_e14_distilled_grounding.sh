#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
GROUNDING_TRAIN="${PROJECT_ROOT}/data/coco/grounding/e13_error_grounding_100_seed49.jsonl"
GROUNDING_VAL="${PROJECT_ROOT}/data/coco/grounding/val_grounding_seed47.jsonl"
CONTROL="E14a_distilled_rehearsal_control_from_e6"
TREATMENT="E14b_distilled_rehearsal_grounding100_from_e6"
LOG="${VQA_LOG_ROOT}/e14_distilled_grounding_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E14 distilled grounding sequence started at $(date --iso-8601=seconds)"
sha256sum "${GROUNDING_TRAIN}" "${GROUNDING_VAL}"

run_training() {
  local run_name="$1"
  local grounding_examples="$2"
  grounded-vqa-train-distilled-grounding \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter-init "${E6}" \
    --grounding-train "${GROUNDING_TRAIN}" \
    --grounding-val "${GROUNDING_VAL}" \
    --quantization 4bit \
    --qformer-rank 4 \
    --qformer-alpha 8 \
    --qformer-dropout 0 \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --distillation-weight 0.5 \
    --distillation-temperature 2 \
    --ordinary-examples 1000 \
    --grounding-examples "${grounding_examples}" \
    --validation-vqa-examples 128 \
    --validation-grounding-examples 128 \
    --vqa-accuracy-tolerance 0.2 \
    --evaluation-exclusion-size 1000 \
    --evaluation-exclusion-seed 42 \
    --batch-size 1 \
    --validation-batch-size 8 \
    --gradient-accumulation 4 \
    --epochs 1 \
    --num-workers 4 \
    --seed 51 \
    --eval-every 50 \
    --early-stop-patience 3 \
    --output-name "${run_name}" \
    --log-every 10 \
    --save-every 250
}

run_training "${CONTROL}" 0
run_training "${TREATMENT}" 100

for SPEC in "${CONTROL}:E14a_control" "${TREATMENT}:E14b_grounding"
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

echo "E14 distilled grounding sequence completed at $(date --iso-8601=seconds)"
