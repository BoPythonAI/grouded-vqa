#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
LOG="${VQA_LOG_ROOT}/e18_answer_supervision_10k_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "E18 fixed-10k answer-supervision sequence started at $(date --iso-8601=seconds)"

for SPEC in \
  "E18b_instructblip_llm_lora_r8_train10k_frequency:frequency:" \
  "E18c_instructblip_llm_lora_r8_train10k_frequency_number25:frequency:0.25"
do
  IFS=: read -r RUN_NAME ANSWER_TARGET NUMBER_FRACTION <<<"${SPEC}"
  TRAIN_ARGS=(
    --model-id "${MODEL}"
    --model-kind instructblip
    --scope llm
    --split train
    --quantization 4bit
    --rank 8
    --alpha 16
    --dropout 0.05
    --learning-rate 1e-4
    --weight-decay 0.01
    --batch-size 1
    --gradient-accumulation 4
    --epochs 1
    --max-samples 10000
    --num-workers 4
    --seed 42
    --answer-target "${ANSWER_TARGET}"
    --output-name "${RUN_NAME}"
    --log-every 50
    --save-every 2500
  )
  if [[ -n "${NUMBER_FRACTION}" ]]; then
    TRAIN_ARGS+=(--number-fraction "${NUMBER_FRACTION}")
  fi

  if [[ ! -d "${VQA_OUTPUT_ROOT}/${RUN_NAME}/final-adapter" ]]; then
    grounded-vqa-train "${TRAIN_ARGS[@]}"
  fi

  EVAL_NAME="${RUN_NAME}_eval5000_short"
  grounded-vqa-predict \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${VQA_OUTPUT_ROOT}/${RUN_NAME}/final-adapter" \
    --split val \
    --max-samples 5000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --resume \
    --output-name "${EVAL_NAME}"

  grounded-vqa-rerank-short \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${VQA_OUTPUT_ROOT}/${RUN_NAME}/final-adapter" \
    --predictions "${VQA_OUTPUT_ROOT}/${EVAL_NAME}/predictions.jsonl" \
    --split val \
    --quantization 4bit \
    --batch-size 8 \
    --prompt-style short \
    --resume \
    --output-name "${EVAL_NAME}_rerank"
done

echo "E18 fixed-10k answer-supervision sequence completed at $(date --iso-8601=seconds)"
