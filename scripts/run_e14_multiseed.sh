#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${PROJECT_ROOT}/models/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
GROUNDING_TRAIN="${PROJECT_ROOT}/data/coco/grounding/e13_error_grounding_100_seed49.jsonl"
GROUNDING_VAL="${PROJECT_ROOT}/data/coco/grounding/val_grounding_seed47.jsonl"
LOG="${VQA_LOG_ROOT}/e14_multiseed_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E14 seed 52/53 sequence started at $(date --iso-8601=seconds)"
sha256sum "${GROUNDING_TRAIN}" "${GROUNDING_VAL}"

run_training() {
  local seed="$1"
  local arm="$2"
  local grounding_examples="$3"
  local run_name="E14${arm}_seed${seed}_distilled_rehearsal_from_e6"
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
    --seed "${seed}" \
    --eval-every 50 \
    --early-stop-patience 3 \
    --output-name "${run_name}" \
    --log-every 10 \
    --save-every 250
}

run_evaluation() {
  local seed="$1"
  local arm="$2"
  local label="$3"
  local run_name="E14${arm}_seed${seed}_distilled_rehearsal_from_e6"
  local prefix="E14_seed${seed}_${label}"
  local adapter="${VQA_OUTPUT_ROOT}/${run_name}/best-adapter"

  grounded-vqa-eval-grounding \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${adapter}" \
    --dataset "${GROUNDING_VAL}" \
    --quantization 4bit \
    --batch-size 8 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${prefix}_best_grounding_val1000"

  grounded-vqa-predict \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${adapter}" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${prefix}_best_vqa_val1000"

  grounded-vqa-mine-complementary \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${adapter}" \
    --split val \
    --quantization 4bit \
    --candidate-pairs 1000 \
    --selected-pairs 1000 \
    --batch-size 16 \
    --num-workers 4 \
    --seed 42 \
    --output-name "${prefix}_best_complementary_val1000"

  grounded-vqa-diagnose-alignment \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${adapter}" \
    --split val \
    --max-samples 1000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --output-name "${prefix}_best_alignment_diag_val1000"
}

for SEED in 52 53
do
  run_training "${SEED}" "a" 0
  run_training "${SEED}" "b" 100
  run_evaluation "${SEED}" "a" "control"
  run_evaluation "${SEED}" "b" "grounding"
done

echo "E14 seed 52/53 sequence completed at $(date --iso-8601=seconds)"
