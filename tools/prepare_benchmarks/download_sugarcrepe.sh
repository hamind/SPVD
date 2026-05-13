#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/sugarcrepe"
RAW="${RAW_ROOT}/sugarcrepe"
REPO="${RAW}/sugar-crepe"
mkdir -p "${ROOT}/annotations" "${ROOT}/splits" "${ROOT}/metadata" "${RAW}"

log "prepare SugarCrepe under ${ROOT}"
clone_or_update "https://github.com/RAIVNLab/sugar-crepe.git" "${REPO}" || true

if [ -d "/vepfs/dataset/sugarcrepe/ready/data" ]; then
  for src in /vepfs/dataset/sugarcrepe/ready/data/*.json; do
    [ -f "${src}" ] && copy_file_once "${src}" "${ROOT}/annotations/$(basename "${src}")" || true
  done
fi

if [ -d "${REPO}/data" ]; then
  for src in "${REPO}"/data/*.json; do
    [ -f "${src}" ] && copy_file_once "${src}" "${ROOT}/annotations/$(basename "${src}")" || true
  done
fi

if [ -d "/vepfs/dataset/sugarcrepe/ready/coco2017/val2017" ]; then
  mkdir -p "${BENCHMARK_ROOT}/coco/images"
  link_dir_once "/vepfs/dataset/sugarcrepe/ready/coco2017/val2017" "${BENCHMARK_ROOT}/coco/images/val2017" || true
fi
link_dir_once "${BENCHMARK_ROOT}/coco/images/val2017" "${ROOT}/coco_val2017_images" || true

find "${ROOT}/annotations" -maxdepth 1 -type f -name "*.json" -printf "%f\n" | sort > "${ROOT}/splits/categories.txt" || true

write_status_marker "sugarcrepe" "attempted" "SugarCrepe annotations prepared; COCO val2017 images are referenced by symlink."
log "SugarCrepe done"

