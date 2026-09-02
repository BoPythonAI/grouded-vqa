#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo >/dev/null
export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_DISABLE_XET=1
source /root/autodl-tmp/vision-language/code/grounded-vqa/scripts/server_env.sh

model_id="Salesforce/instructblip-flan-t5-xl"
model_dir="${VQA_MODEL_ROOT}/Salesforce--instructblip-flan-t5-xl"
mkdir -p "${model_dir}"

if [[ ! -f "${model_dir}/config.json" || \
      ! -f "${model_dir}/model.safetensors.index.json" ]]; then
  hf download "${model_id}" \
    --exclude "*.safetensors" \
    --exclude "*.bin" \
    --local-dir "${model_dir}"
fi

base_url="https://hf-mirror.com/${model_id}/resolve/main"

download_shard() {
  local filename="$1"
  local expected_sha256="$2"
  local final_path="${model_dir}/${filename}"
  local partial_path="${final_path}.part"

  if [[ -f "${final_path}" ]]; then
    printf '%s  %s\n' "${expected_sha256}" "${final_path}" \
      | sha256sum --check --strict
    return
  fi
  aria2c \
    --continue=true \
    --max-tries=0 \
    --retry-wait=3 \
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=4M \
    --file-allocation=none \
    --summary-interval=10 \
    --dir="${model_dir}" \
    --out="$(basename "${partial_path}")" \
    "${base_url}/${filename}"
  printf '%s  %s\n' "${expected_sha256}" "${partial_path}" \
    | sha256sum --check --strict
  mv "${partial_path}" "${final_path}"
}

download_shard \
  model-00001-of-00002.safetensors \
  17cee128ee8cd7594fdbab95129819ea07997851a2426bd3ee0c24c4583d9054 &
pid_one=$!
download_shard \
  model-00002-of-00002.safetensors \
  56d404fca61133f4dd9b7607e4528ab68882fc4a649d26b18c341fb129951050 &
pid_two=$!
wait "${pid_one}"
wait "${pid_two}"

grounded-vqa-smoke \
  --model-id "${model_dir}" \
  --model-kind instructblip \
  --quantization 4bit
