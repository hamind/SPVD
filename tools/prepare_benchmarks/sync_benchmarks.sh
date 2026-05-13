#!/usr/bin/env bash

set -euo pipefail

SRC="/vepfs/dataset/benchmark"
DST=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash sync_benchmarks.sh --src /vepfs/dataset/benchmark --dst <TARGET_PATH> --dry-run
  bash sync_benchmarks.sh --src /vepfs/dataset/benchmark --dst <TARGET_PATH>

The destination can be a local path or rsync remote such as user@host:/path.
This script never deletes destination files by default.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --src)
      SRC="$2"
      shift 2
      ;;
    --dst)
      DST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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
done

if [ -z "${DST}" ]; then
  usage
  exit 2
fi

ARGS=(-aH --info=progress2)
if [ "${DRY_RUN}" -eq 1 ]; then
  ARGS+=(--dry-run)
fi

rsync "${ARGS[@]}" "${SRC%/}/" "${DST%/}/"

