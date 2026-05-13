#!/usr/bin/env bash

set -u

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/vepfs/dataset/benchmark}"
RAW_ROOT="${RAW_ROOT:-${BENCHMARK_ROOT}/_raw}"
LOG_ROOT="${LOG_ROOT:-${BENCHMARK_ROOT}/download_logs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${BENCHMARK_ROOT}/manifests}"
SPVD_ROOT="${SPVD_ROOT:-/vepfs/code/SPVD}"

mkdir -p "${BENCHMARK_ROOT}" "${RAW_ROOT}" "${LOG_ROOT}" "${MANIFEST_ROOT}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

count_files_1() {
  local path="$1"
  if [ -d "${path}" ]; then
    find "${path}" -maxdepth 1 -type f | wc -l
  else
    printf '0\n'
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [ -s "${out}" ]; then
    log "skip existing file: ${out}"
    return 0
  fi

  log "download: ${url}"
  if have_cmd aria2c; then
    aria2c -c -x 8 -s 8 --allow-overwrite=false --auto-file-renaming=false -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}"
  elif have_cmd wget; then
    wget -c -O "${out}" "${url}"
  elif have_cmd curl; then
    curl -L -C - -o "${out}" "${url}"
  else
    log "no downloader found for ${url}"
    return 127
  fi
}

extract_zip() {
  local zip_path="$1"
  local dest="$2"
  if [ ! -s "${zip_path}" ]; then
    log "missing zip: ${zip_path}"
    return 1
  fi
  mkdir -p "${dest}"
  log "extract: ${zip_path} -> ${dest}"
  unzip -q -n "${zip_path}" -d "${dest}"
}

link_dir_once() {
  local src="$1"
  local dst="$2"
  if [ ! -d "${src}" ]; then
    return 1
  fi
  mkdir -p "$(dirname "${dst}")"
  if [ -e "${dst}" ] || [ -L "${dst}" ]; then
    if [ -d "${dst}" ] && [ ! -L "${dst}" ] && [ -z "$(find "${dst}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
      rmdir "${dst}"
    else
      log "keep existing path: ${dst}"
      return 0
    fi
  fi
  if [ -e "${dst}" ] || [ -L "${dst}" ]; then
    log "keep existing path: ${dst}"
    return 0
  fi
  ln -s "${src}" "${dst}"
  log "linked ${dst} -> ${src}"
}

link_file_once() {
  local src="$1"
  local dst="$2"
  if [ ! -f "${src}" ]; then
    return 1
  fi
  mkdir -p "$(dirname "${dst}")"
  if [ -e "${dst}" ] || [ -L "${dst}" ]; then
    log "keep existing file: ${dst}"
    return 0
  fi
  ln -s "${src}" "${dst}"
  log "linked ${dst} -> ${src}"
}

copy_file_once() {
  local src="$1"
  local dst="$2"
  if [ ! -f "${src}" ]; then
    return 1
  fi
  mkdir -p "$(dirname "${dst}")"
  if [ -s "${dst}" ]; then
    log "keep existing file: ${dst}"
    return 0
  fi
  cp -a "${src}" "${dst}"
  log "copied ${src} -> ${dst}"
}

clone_or_update() {
  local repo="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [ -d "${dst}/.git" ]; then
    log "update repo: ${dst}"
    git -C "${dst}" fetch --depth 1 origin || return 1
    git -C "${dst}" reset --hard FETCH_HEAD || return 1
  else
    log "clone repo: ${repo}"
    git clone --depth 1 "${repo}" "${dst}"
  fi
}

write_status_marker() {
  local bench="$1"
  local status="$2"
  local note="$3"
  local path="${BENCHMARK_ROOT}/${bench}/metadata/download_status.json"
  mkdir -p "$(dirname "${path}")"
  python3 - "$path" "$bench" "$status" "$note" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, bench, status, note = sys.argv[1:5]
payload = {
    "benchmark": bench,
    "status": status,
    "note": note,
    "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
    f.write("\n")
PY
}
