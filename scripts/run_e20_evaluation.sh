#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

MODEL="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
E6="${VQA_OUTPUT_ROOT}/E6_instructblip_llm_lora_r8_train10k_random/final-adapter"
E20="${VQA_OUTPUT_ROOT}/E20_lite_e6_llm_hardnegative10pct_distill/final-adapter"

run_vqa() {
  local adapter="$1"
  local direct_name="$2"
  local rerank_name="$3"
  grounded-vqa-predict \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --quantization 4bit \
    --adapter "${adapter}" \
    --split val \
    --max-samples 5000 \
    --seed 42 \
    --max-new-tokens 10 \
    --prompt-style short \
    --resume \
    --output-name "${direct_name}"
  grounded-vqa-rerank-short \
    --model-id "${MODEL}" \
    --model-kind instructblip \
    --adapter "${adapter}" \
    --predictions "${VQA_OUTPUT_ROOT}/${direct_name}/predictions.jsonl" \
    --split val \
    --quantization 4bit \
    --batch-size 8 \
    --prompt-style short \
    --resume \
    --output-name "${rerank_name}"
}

run_vqa \
  "${E6}" \
  E20_control_E6_vqa_dev5000_short \
  E20_control_E6_vqa_dev5000_short_rerank
run_vqa \
  "${E20}" \
  E20_candidate_vqa_dev5000_short \
  E20_candidate_vqa_dev5000_short_rerank

grounded-vqa-eval-pope \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter "${E20}" \
  --pope-root "${PROJECT_ROOT}/data/pope" \
  --image-root "${VQA_DATA_ROOT}/val2014" \
  --quantization 4bit \
  --batch-size 8 \
  --max-new-tokens 5 \
  --resume \
  --output-name E20_candidate_official_COCO_POPE

grounded-vqa-compare-pope \
  --baseline "${VQA_OUTPUT_ROOT}/H1_E6_official_COCO_POPE" \
  --candidate "${VQA_OUTPUT_ROOT}/E20_candidate_official_COCO_POPE" \
  --output "${VQA_OUTPUT_ROOT}/E20_candidate_official_COCO_POPE/paired_vs_e6.json"

grounded-vqa-eval-chair \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter "${E20}" \
  --annotation-root "${PROJECT_ROOT}/data/coco/annotations" \
  --image-root "${VQA_DATA_ROOT}/val2014" \
  --selection-file "${VQA_OUTPUT_ROOT}/H2_CHAIR_selection_seed42.json" \
  --sample-size 500 \
  --seed 42 \
  --quantization 4bit \
  --batch-size 8 \
  --max-new-tokens 64 \
  --resume \
  --output-name E20_candidate_CHAIR500

grounded-vqa-compare-chair \
  --baseline "${VQA_OUTPUT_ROOT}/H2_E6_CHAIR500" \
  --candidate "${VQA_OUTPUT_ROOT}/E20_candidate_CHAIR500" \
  --output "${VQA_OUTPUT_ROOT}/E20_candidate_CHAIR500/paired_vs_e6.json"

grounded-vqa-eval-hallusionbench \
  --model-id "${MODEL}" \
  --model-kind instructblip \
  --adapter "${E20}" \
  --parquet-root "${PROJECT_ROOT}/data/hallusionbench/hf/data" \
  --quantization 4bit \
  --batch-size 8 \
  --max-new-tokens 5 \
  --resume \
  --output-name E20_candidate_HallusionBench

grounded-vqa-compare-hallusionbench \
  --baseline "${VQA_OUTPUT_ROOT}/H3_E6_HallusionBench" \
  --candidate "${VQA_OUTPUT_ROOT}/E20_candidate_HallusionBench" \
  --output "${VQA_OUTPUT_ROOT}/E20_candidate_HallusionBench/paired_vs_e6.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/vision-language/outputs")

def load(relative: str):
    return json.loads((root / relative).read_text())

control_vqa = load("E20_control_E6_vqa_dev5000_short_rerank/metrics.json")
candidate_vqa = load("E20_candidate_vqa_dev5000_short_rerank/metrics.json")
e6_pope = load("H1_E6_official_COCO_POPE/metrics.json")
candidate_pope = load("E20_candidate_official_COCO_POPE/metrics.json")
pope_pair = load("E20_candidate_official_COCO_POPE/paired_vs_e6.json")
chair_pair = load("E20_candidate_CHAIR500/paired_vs_e6.json")
e6_hallusion = load("H3_E6_HallusionBench/metrics.json")
candidate_hallusion = load("E20_candidate_HallusionBench/metrics.json")
hallusion_pair = load("E20_candidate_HallusionBench/paired_vs_e6.json")

e6_pope_macro = e6_pope["macro_average"]["generated_official_parser"]
candidate_pope_macro = candidate_pope["macro_average"]["generated_official_parser"]
chair = chair_pair["metrics"]
e6_hb = e6_hallusion["metrics"]
candidate_hb = candidate_hallusion["metrics"]
gates = {
    "vqa_noninferior": candidate_vqa["overall"] - control_vqa["overall"] >= -0.30,
    "pope_accuracy_noninferior": (
        pope_pair["strategies"]["combined"]["accuracy_delta_points"] >= -0.30
    ),
    "pope_precision_or_f1_improved": (
        100 * (candidate_pope_macro["precision"] - e6_pope_macro["precision"]) >= 1.0
        or 100 * (candidate_pope_macro["f1"] - e6_pope_macro["f1"]) >= 1.0
    ),
    "chair_not_worse": (
        chair["chair_s"]["delta_points"] <= 0
        and chair["chair_i"]["delta_points"] <= 0
    ),
    "chair_significant_improvement": (
        chair["chair_s"]["bootstrap_95_ci_points"][1] < 0
        or chair["chair_i"]["bootstrap_95_ci_points"][1] < 0
    ),
    "chair_recall_preserved": chair["object_recall"]["delta_points"] >= -2.0,
    "hallusion_accuracy_noninferior": (
        hallusion_pair["accuracy_delta_points"] >= -1.0
    ),
    "hallusion_fpr_improved": (
        candidate_hb["false_positive_rate"] < e6_hb["false_positive_rate"]
    ),
    "hallusion_fnr_controlled": (
        candidate_hb["false_negative_rate"] <= e6_hb["false_negative_rate"] + 0.10
    ),
}
summary = {
    "vqa": {
        "control": control_vqa["overall"],
        "candidate": candidate_vqa["overall"],
        "delta": candidate_vqa["overall"] - control_vqa["overall"],
    },
    "pope_macro": {
        "e6": e6_pope_macro,
        "candidate": candidate_pope_macro,
        "paired": pope_pair["strategies"]["combined"],
    },
    "chair_paired": chair_pair,
    "hallusionbench": {
        "e6": e6_hb,
        "candidate": candidate_hb,
        "paired": hallusion_pair,
    },
    "gates": gates,
    "eligible_for_full_validation": all(gates.values()),
}
target = root / "E20_lite_e6_llm_hardnegative10pct_distill/evaluation_gate.json"
target.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
