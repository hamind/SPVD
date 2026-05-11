#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/svo_probes"
RAW="${RAW_ROOT}/svo_probes"
mkdir -p "${ROOT}/annotations" "${ROOT}/images_or_links" "${ROOT}/metadata" "${RAW}"

log "prepare SVO-Probes under ${ROOT}"
clone_or_update "https://github.com/google-deepmind/svo_probes.git" "${RAW}/svo_probes" || true

copy_file_once "${RAW}/svo_probes/svo_probes.csv" "${ROOT}/annotations/svo_probes.csv" || \
  download_file "https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv" "${ROOT}/annotations/svo_probes.csv" || true
copy_file_once "${RAW}/svo_probes/image_urls.txt" "${ROOT}/images_or_links/image_urls.txt" || \
  download_file "https://raw.githubusercontent.com/google-deepmind/svo_probes/main/image_urls.txt" "${ROOT}/images_or_links/image_urls.txt" || true

write_status_marker "svo_probes" "attempted" "SVO-Probes annotations and image URL list prepared; original image availability depends on upstream URLs."
log "SVO-Probes done"

