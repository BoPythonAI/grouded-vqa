#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
INSTANCES="${PROJECT_ROOT}/data/coco/annotations/instances_train2014.json"
CAPTIONS="${PROJECT_ROOT}/data/coco/annotations/captions_train2014.json"
POOL="${PROJECT_ROOT}/data/coco/grounding/e20_hard_negative_candidate10000_seed60.jsonl"
MINING_RUN="E20_hard_negative_mining10000_seed60"
NEGATIVES="${VQA_OUTPUT_ROOT}/${MINING_RUN}/selected.jsonl"
RUN="E20_lite_e6_llm_hardnegative10pct_distill"

grounded-vqa-build-hard-negatives \
  --instances "${INSTANCES}" \
  --captions "${CAPTIONS}" \
  --split train \
  --count 10000 \
  --max-per-category 200 \
  --seed 60 \
  --output "${POOL}"

grounded-vqa-mine-hard-negatives \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter "${E6}" \
  --candidates "${POOL}" \
  --quantization 4bit \
  --batch-size 8 \
  --selected-count 1000 \
  --max-per-category 50 \
  --resume \
  --output-name "${MINING_RUN}"

sha256sum "${INSTANCES}" "${CAPTIONS}" "${POOL}" "${NEGATIVES}"

grounded-vqa-train-negative-distillation \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter-init "${E6}" \
  --negative-train "${NEGATIVES}" \
  --quantization 4bit \
  --ordinary-examples 9000 \
  --negative-examples 1000 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --negative-loss-weight 1.0 \
  --distillation-weight 0.5 \
  --distillation-temperature 2 \
  --distillation-interval 4 \
  --batch-size 8 \
  --gradient-accumulation 1 \
  --epochs 1 \
  --num-workers 8 \
  --seed 60 \
  --output-name "${RUN}" \
  --log-every 20 \
  --save-every 5000
