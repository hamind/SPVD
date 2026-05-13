#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/Flickr30k"
RAW="${RAW_ROOT}/flickr30k"
mkdir -p "${ROOT}/images" "${ROOT}/annotations" "${ROOT}/splits" "${RAW}"

log "prepare Flickr30k retrieval under ${ROOT}"

if [ -n "${FLICKR30K_IMAGE_ROOT:-}" ] && [ -d "${FLICKR30K_IMAGE_ROOT}" ]; then
  link_dir_once "${FLICKR30K_IMAGE_ROOT}" "${ROOT}/images" || true
elif [ -d "/vepfs/dataset/aro/ready/flickr30k/flickr30k-images" ]; then
  link_dir_once "/vepfs/dataset/aro/ready/flickr30k/flickr30k-images" "${ROOT}/images" || true
elif [ -d "/vepfs/dataset/flickr30k/flickr30k-images" ]; then
  link_dir_once "/vepfs/dataset/flickr30k/flickr30k-images" "${ROOT}/images" || true
else
  log "Flickr30k images are not present. Official access requires the UIUC request form."
  if [ "${ALLOW_FLICKR30K_KAGGLE:-0}" = "1" ] && have_cmd kaggle; then
    if [ -f "${KAGGLE_CONFIG_DIR:-${HOME}/.kaggle}/kaggle.json" ]; then
      kaggle datasets download -d eeshawn/flickr30k -p "${RAW}" || true
      if [ -s "${RAW}/flickr30k.zip" ]; then
        extract_zip "${RAW}/flickr30k.zip" "${RAW}/kaggle" || true
        for d in "${RAW}/kaggle/flickr30k_images" "${RAW}/kaggle/flickr30k-images"; do
          [ -d "${d}" ] && link_dir_once "${d}" "${ROOT}/images" && break
        done
      fi
    else
      log "Kaggle enabled but kaggle.json was not found."
    fi
  fi
fi

for split in val test; do
  copy_file_once "/vepfs/dataset/aro/ready/flickr30k/flickr30k_${split}.json" "${ROOT}/annotations/flickr30k_${split}.json" || true
done

KARPATHY_DIR="${RAW_ROOT}/coco/karpathy"
if [ ! -s "${KARPATHY_DIR}/dataset_flickr30k.json" ]; then
  mkdir -p "${KARPATHY_DIR}"
  if [ -s "${RAW_ROOT}/coco/caption_datasets.zip" ]; then
    extract_zip "${RAW_ROOT}/coco/caption_datasets.zip" "${KARPATHY_DIR}" || true
  else
    download_file "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" "${RAW_ROOT}/coco/caption_datasets.zip" || true
    extract_zip "${RAW_ROOT}/coco/caption_datasets.zip" "${KARPATHY_DIR}" || true
  fi
fi

if [ -s "${KARPATHY_DIR}/dataset_flickr30k.json" ]; then
  copy_file_once "${KARPATHY_DIR}/dataset_flickr30k.json" "${ROOT}/annotations/dataset_flickr30k_karpathy.json" || true
  python3 - "${KARPATHY_DIR}/dataset_flickr30k.json" "${ROOT}" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
data = json.load(open(src, encoding="utf-8"))
images = data.get("images", [])
groups = {"train": [], "val": [], "test": []}
jsonl = root / "annotations" / "captions_karpathy.jsonl"
with open(jsonl, "w", encoding="utf-8") as jf:
    for item in images:
        split = item.get("split")
        if split in groups:
            groups[split].append(item.get("filename", ""))
        captions = [s.get("raw", "") for s in item.get("sentences", [])]
        jf.write(json.dumps({"image": item.get("filename", ""), "split": split, "captions": captions}, ensure_ascii=True) + "\n")
for split, names in groups.items():
    with open(root / "splits" / f"{split}.txt", "w", encoding="utf-8") as f:
        for name in names:
            f.write(str(name) + "\n")
PY
fi

write_status_marker "flickr30k" "attempted" "Flickr30k images reused from local authorized copy when present; official download is manual/form-gated."
log "Flickr30k done"

