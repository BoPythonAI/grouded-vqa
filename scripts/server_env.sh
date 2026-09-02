#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/vision-language"
DATA_DISK="/root/autodl-tmp"

if [[ ! -d "${DATA_DISK}" ]] || ! mountpoint -q "${DATA_DISK}"; then
  echo "Expected data disk ${DATA_DISK} is not mounted." >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT
export VQA_CODE_ROOT="${PROJECT_ROOT}/code/grounded-vqa"
export VQA_DATA_ROOT="${PROJECT_ROOT}/data/vqav2"
export VQA_MODEL_ROOT="${PROJECT_ROOT}/models"
export VQA_OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
export VQA_LOG_ROOT="${PROJECT_ROOT}/logs"

export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export TMPDIR="${PROJECT_ROOT}/cache/tmp"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"

mkdir -p \
  "${VQA_DATA_ROOT}" \
  "${VQA_MODEL_ROOT}" \
  "${VQA_OUTPUT_ROOT}" \
  "${VQA_LOG_ROOT}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}"

if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/venv/bin/activate"
fi

cd "${VQA_CODE_ROOT}"
