#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo >/dev/null
source /root/autodl-tmp/vision-language/code/grounded-vqa/scripts/server_env.sh

archive_dir="${VQA_DATA_ROOT}/archives"
mkdir -p "${archive_dir}"

download_metadata() {
  local url="$1"
  local filename="$2"
  local final_path="${archive_dir}/${filename}"
  local partial_path="${final_path}.part"
  [[ -f "${final_path}" ]] && return

  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    aria2c \
      --continue=true \
      --max-tries=0 \
      --retry-wait=3 \
      --max-connection-per-server=4 \
      --split=4 \
      --min-split-size=1M \
      --file-allocation=none \
      --summary-interval=10 \
      --dir="${archive_dir}" \
      --out="$(basename "${partial_path}")" \
      "${url}"
  unzip -tq "${partial_path}"
  mv "${partial_path}" "${final_path}"
}

download_images() {
  local filename="train2014.zip"
  local final_path="${archive_dir}/${filename}"
  local partial_path="${final_path}.part"
  local expected_sha256="ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775"
  local url="https://hf-mirror.com/datasets/GAIA-URJC/COCO_2014/resolve/main/${filename}"

  if [[ ! -f "${final_path}" ]]; then
    aria2c \
      --continue=true \
      --max-tries=0 \
      --retry-wait=3 \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=4M \
      --file-allocation=none \
      --summary-interval=10 \
      --dir="${archive_dir}" \
      --out="$(basename "${partial_path}")" \
      "${url}"
    printf '%s  %s\n' "${expected_sha256}" "${partial_path}" \
      | sha256sum --check --strict
    mv "${partial_path}" "${final_path}"
  fi
  printf '%s  %s\n' "${expected_sha256}" "${final_path}" \
    | sha256sum --check --strict
}

download_metadata \
  https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip \
  v2_Questions_Train_mscoco.zip &
questions_pid=$!
download_metadata \
  https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip \
  v2_Annotations_Train_mscoco.zip &
annotations_pid=$!
download_images &
images_pid=$!

wait "${questions_pid}"
wait "${annotations_pid}"
wait "${images_pid}"

# Re-validates all ZIPs, extracts the official layout, then removes archives.
grounded-vqa-download --split train --include-images
