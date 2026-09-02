#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo >/dev/null
source /root/autodl-tmp/vision-language/code/grounded-vqa/scripts/server_env.sh

archive_dir="${VQA_DATA_ROOT}/archives"
archive="${archive_dir}/val2014.zip"
partial="${archive}.part"
expected_sha256="fe9be816052049c34717e077d9e34aa60814a55679f804cd043e3cbee3bb9fde0"
mirror_url="https://hf-mirror.com/datasets/GAIA-URJC/COCO_2014/resolve/main/val2014.zip"
mkdir -p "${archive_dir}"

if [[ ! -d "${VQA_DATA_ROOT}/val2014" ]]; then
  if [[ ! -f "${archive}" ]]; then
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
      --out="$(basename "${partial}")" \
      "${mirror_url}"
    printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check --strict
    mv "${partial}" "${archive}"
  fi
  printf '%s  %s\n' "${expected_sha256}" "${archive}" | sha256sum --check --strict
fi

# This also obtains official questions/annotations, verifies the ZIP, extracts
# it, and removes the large archive after successful extraction.
grounded-vqa-download --split val --include-images
