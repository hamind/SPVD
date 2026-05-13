#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash prepare_all.sh --p0
  bash prepare_all.sh --p1
  bash prepare_all.sh --all
  bash prepare_all.sh --only coco
  bash prepare_all.sh --only sugarcrepe

Environment:
  BENCHMARK_ROOT=/vepfs/dataset/benchmark
  HF_TOKEN=<huggingface token for gated datasets>
  KAGGLE_CONFIG_DIR=<directory containing kaggle.json>
  FLICKR30K_IMAGE_ROOT=<authorized local Flickr30k image directory>
  ALLOW_FLICKR30K_KAGGLE=1
  ALLOW_KARPATHY_MIRROR=1
EOF
}

P0=(coco flickr30k aro sugarcrepe winoground)
P1=(svo_probes bivlc)

run_one() {
  local name="$1"
  local log_path="${LOG_ROOT}/${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  log "start ${name}; log=${log_path}"
  local rc=0
  case "${name}" in
    winoground)
      python3 "${SCRIPT_DIR}/download_winoground.py" >"${log_path}" 2>&1 || rc=$?
      ;;
    *)
      bash "${SCRIPT_DIR}/download_${name}.sh" >"${log_path}" 2>&1 || rc=$?
      ;;
  esac
  if [ "${rc}" -ne 0 ]; then
    log "FAILED ${name}; see ${log_path}"
    printf '%s\t%s\t%s\n' "$(timestamp)" "${name}" "${log_path}" >> "${LOG_ROOT}/failures.tsv"
  else
    log "finished ${name}; see ${log_path}"
  fi
  return 0
}

run_group() {
  local item
  for item in "$@"; do
    run_one "${item}"
  done
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

case "$1" in
  --p0)
    run_group "${P0[@]}"
    ;;
  --p1)
    run_group "${P1[@]}"
    ;;
  --all)
    run_group "${P0[@]}"
    run_group "${P1[@]}"
    ;;
  --only)
    if [ "$#" -ne 2 ]; then
      usage
      exit 2
    fi
    run_one "$2"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

log "building manifests"
python3 "${SCRIPT_DIR}/build_manifest.py" || true
log "running verifier"
python3 "${SCRIPT_DIR}/verify_benchmarks.py" || true
log "refreshing manifests with verification"
python3 "${SCRIPT_DIR}/build_manifest.py" || true
log "all requested benchmark preparation steps completed"

