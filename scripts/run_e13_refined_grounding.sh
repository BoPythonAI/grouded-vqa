#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
PREDICTIONS="${VQA_OUTPUT_ROOT}/E13_e6_train_heldout5000_seed49/predictions.jsonl"
INSTANCES="${PROJECT_ROOT}/data/coco/annotations/instances_train2014.json"
GROUNDING_DIR="${PROJECT_ROOT}/data/coco/grounding"
MASTER="${GROUNDING_DIR}/e13_error_grounding_200_seed49.jsonl"
SUBSET="${GROUNDING_DIR}/e13_error_grounding_100_seed49.jsonl"
GROUNDING_VAL="${GROUNDING_DIR}/val_grounding_seed47.jsonl"
LOG="${VQA_LOG_ROOT}/e13_refined_grounding_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E13 refined grounding sequence started at $(date --iso-8601=seconds)"

grounded-vqa-build-error-grounding \
  --instances "${INSTANCES}" \
  --predictions "${PREDICTIONS}" \
  --positive-count 80 \
  --negative-count 80 \
  --counting-count 40 \
  --min-area-ratio 0.001 \
  --max-count-answer 10 \
  --seed 49 \
  --output "${MASTER}" \
  --subset-output "${SUBSET}" \
  --subset-positive-count 40 \
  --subset-negative-count 40 \
  --subset-counting-count 20

sha256sum "${PREDICTIONS}" "${MASTER}" "${SUBSET}" "${GROUNDING_VAL}"

run_training() {
  local run_name="$1"
  local ordinary_examples="$2"
  local grounding_examples="$3"
  local grounding_file="$4"
  grounded-vqa-train-grounding-qformer \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter-init "${E6}" \
    --grounding-train "${grounding_file}" \
    --grounding-val "${GROUNDING_VAL}" \
    --quantization 4bit \
    --qformer-rank 8 \
    --qformer-alpha 16 \
    --qformer-dropout 0.05 \
    --learning-rate 3e-5 \
    --weight-decay 0.01 \
    --ordinary-pool-size 1000 \
    --ordinary-examples "${ordinary_examples}" \
    --grounding-examples "${grounding_examples}" \
    --validation-grounding-examples 128 \
    --validation-vqa-examples 128 \
    --vqa-nll-tolerance 0.005 \
    --evaluation-exclusion-size 1000 \
    --evaluation-exclusion-seed 42 \
    --batch-size 1 \
    --validation-batch-size 8 \
    --gradient-accumulation 4 \
    --epochs 1 \
    --num-workers 4 \
    --seed 50 \
    --eval-every 50 \
    --early-stop-patience 3 \
    --early-stop-min-delta 0 \
    --output-name "${run_name}" \
    --log-every 10 \
    --save-every 250
}

run_training "E13a_refined_control_from_e6" 1000 0 "${MASTER}"
run_training "E13b_error_grounding_10pct_from_e6" 900 100 "${SUBSET}"
run_training "E13c_error_grounding_20pct_from_e6" 800 200 "${MASTER}"

for SPEC in \
  "E13a_refined_control_from_e6:E13a_control" \
  "E13b_error_grounding_10pct_from_e6:E13b_grounding10" \
  "E13c_error_grounding_20pct_from_e6:E13c_grounding20"
do
  RUN_NAME="${SPEC%%:*}"
  PREFIX="${SPEC##*:}"
  ADAPTER="${VQA_OUTPUT_ROOT}/${RUN_NAME}/best-adapter"

  grounded-vqa-eval-grounding \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${ADAPTER}" \
    --dataset "${GROUNDING_VAL}" \
    --quantization 4bit \
    --batch-size 8 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${PREFIX}_best_grounding_val1000"

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
    --output-name "${PREFIX}_best_vqa_val1000"

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
    --output-name "${PREFIX}_best_complementary_val1000"

  grounded-vqa-diagnose-alignment \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${ADAPTER}" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${PREFIX}_best_alignment_diag_val1000"
done

echo "E13 refined grounding sequence completed at $(date --iso-8601=seconds)"
