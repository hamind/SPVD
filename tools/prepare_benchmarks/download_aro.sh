#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/aro"
RAW="${RAW_ROOT}/aro"
REPO="${RAW}/vision-language-models-are-bows"
mkdir -p "${ROOT}/annotations" "${ROOT}/vg_relation" "${ROOT}/vg_attribution" "${ROOT}/coco_order" "${ROOT}/flickr30k_order" "${RAW}"

log "prepare ARO under ${ROOT}"
clone_or_update "https://github.com/mertyg/vision-language-models-are-bows.git" "${REPO}" || true

download_gdrive() {
  local id="$1"
  local out="$2"
  if [ -s "${out}" ]; then
    return 0
  fi
  if have_cmd gdown; then
    gdown --id "${id}" --output "${out}"
  else
    log "gdown is not installed; cannot fetch Google Drive id ${id}"
    return 1
  fi
}

copy_file_once "/vepfs/dataset/aro/ready/vg/visual_genome_relation.json" "${ROOT}/annotations/visual_genome_relation.json" || \
  download_gdrive "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3" "${ROOT}/annotations/visual_genome_relation.json" || true

copy_file_once "/vepfs/dataset/aro/ready/vg/visual_genome_attribution.json" "${ROOT}/annotations/visual_genome_attribution.json" || \
  download_gdrive "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw" "${ROOT}/annotations/visual_genome_attribution.json" || true

if [ ! -d "${ROOT}/vg_relation/images" ] && [ ! -d "${ROOT}/vg_attribution/images" ]; then
  if [ -d "/vepfs/dataset/aro/ready/vg/images" ]; then
    link_dir_once "/vepfs/dataset/aro/ready/vg/images" "${ROOT}/vg_relation/images" || true
    link_dir_once "/vepfs/dataset/aro/ready/vg/images" "${ROOT}/vg_attribution/images" || true
  else
    download_gdrive "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9" "${RAW}/vgr_vga_images.zip" || true
    extract_zip "${RAW}/vgr_vga_images.zip" "${ROOT}/vg_relation" || true
    link_dir_once "${ROOT}/vg_relation/images" "${ROOT}/vg_attribution/images" || true
  fi
fi

link_file_once "${ROOT}/annotations/visual_genome_relation.json" "${ROOT}/vg_relation/annotations.json" || true
link_file_once "${ROOT}/annotations/visual_genome_attribution.json" "${ROOT}/vg_attribution/annotations.json" || true

for split in val test; do
  copy_file_once "/vepfs/dataset/aro/ready/coco2014/coco_karpathy_${split}.json" "${ROOT}/coco_order/coco_karpathy_${split}.json" || \
    download_file "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_${split}.json" "${ROOT}/coco_order/coco_karpathy_${split}.json" || true
  copy_file_once "/vepfs/dataset/aro/ready/flickr30k/flickr30k_${split}.json" "${ROOT}/flickr30k_order/flickr30k_${split}.json" || \
    download_file "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_${split}.json" "${ROOT}/flickr30k_order/flickr30k_${split}.json" || true
done

link_dir_once "${BENCHMARK_ROOT}/coco/images/val2014" "${ROOT}/coco_order/val2014" || true
link_dir_once "/vepfs/dataset/aro/ready/coco2014/test2014" "${ROOT}/coco_order/test2014" || true
link_dir_once "${BENCHMARK_ROOT}/flickr30k/images" "${ROOT}/flickr30k_order/flickr30k-images" || true

write_status_marker "aro" "attempted" "ARO official repo cloned; VG, COCO-order, and Flickr30k-order files linked or downloaded when possible."
log "ARO done"

