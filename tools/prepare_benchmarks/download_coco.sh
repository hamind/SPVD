#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/coco"
RAW="${RAW_ROOT}/coco"
mkdir -p "${ROOT}/images" "${ROOT}/annotations" "${ROOT}/splits" "${RAW}"

log "prepare COCO retrieval under ${ROOT}"

link_known_coco_dir() {
  local split="$1"
  local expected_min="$2"
  local dst="${ROOT}/images/${split}"
  if [ -d "${dst}" ] && [ "$(count_files_1 "${dst}")" -ge "${expected_min}" ]; then
    log "COCO ${split} already present with $(count_files_1 "${dst}") files"
    return 0
  fi
  for src in \
    "/vepfs/dataset/aro/ready/coco2014/${split}" \
    "/vepfs/dataset/coco/${split}" \
    "/vepfs/dataset/coco2014/${split}" \
    "/vepfs/dataset/COCO/${split}" \
    "/vepfs/dataset/sugarcrepe/ready/coco2017/${split}"; do
    if [ -d "${src}" ] && [ "$(count_files_1 "${src}")" -ge "${expected_min}" ]; then
      link_dir_once "${src}" "${dst}"
      return 0
    fi
  done
  return 1
}

if ! link_known_coco_dir train2014 80000; then
  download_file "http://images.cocodataset.org/zips/train2014.zip" "${RAW}/train2014.zip" || true
  extract_zip "${RAW}/train2014.zip" "${ROOT}/images" || true
fi

if ! link_known_coco_dir val2014 40000; then
  download_file "http://images.cocodataset.org/zips/val2014.zip" "${RAW}/val2014.zip" || true
  extract_zip "${RAW}/val2014.zip" "${ROOT}/images" || true
fi

if ! link_known_coco_dir val2017 5000; then
  download_file "http://images.cocodataset.org/zips/val2017.zip" "${RAW}/val2017.zip" || true
  extract_zip "${RAW}/val2017.zip" "${ROOT}/images" || true
fi

if [ ! -s "${ROOT}/annotations/captions_train2014.json" ] || [ ! -s "${ROOT}/annotations/captions_val2014.json" ]; then
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "${RAW}/annotations_trainval2014.zip" || true
  extract_zip "${RAW}/annotations_trainval2014.zip" "${ROOT}" || true
fi

if [ ! -s "${ROOT}/annotations/captions_val2017.json" ]; then
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${RAW}/annotations_trainval2017.zip" || true
  extract_zip "${RAW}/annotations_trainval2017.zip" "${ROOT}" || true
fi

KARPATHY_ZIP="${RAW}/caption_datasets.zip"
KARPATHY_DIR="${RAW}/karpathy"
if [ ! -s "${KARPATHY_DIR}/dataset_coco.json" ]; then
  mkdir -p "${KARPATHY_DIR}"
  if download_file "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" "${KARPATHY_ZIP}"; then
    extract_zip "${KARPATHY_ZIP}" "${KARPATHY_DIR}" || true
  elif [ "${ALLOW_KARPATHY_MIRROR:-0}" = "1" ]; then
    log "official Karpathy zip failed; using opt-in GitHub mirror"
    download_file "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download=" "${KARPATHY_DIR}/dataset_coco.json" || true
  else
    log "Karpathy official zip unavailable; set ALLOW_KARPATHY_MIRROR=1 to use a third-party mirror"
  fi
fi

if [ -s "${KARPATHY_DIR}/dataset_coco.json" ]; then
  copy_file_once "${KARPATHY_DIR}/dataset_coco.json" "${ROOT}/annotations/dataset_coco_karpathy.json" || true
  python3 - "${KARPATHY_DIR}/dataset_coco.json" "${ROOT}/splits" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
data = json.load(open(src, encoding="utf-8"))
images = data.get("images", [])
groups = {"train": [], "val": [], "test": []}
for item in images:
    split = item.get("split")
    if split in {"train", "restval"}:
        groups["train"].append(item)
    elif split in groups:
        groups[split].append(item)
for name, rows in groups.items():
    path = out / f"karpathy_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True)
        f.write("\n")
    txt = out / f"karpathy_{name}.txt"
    with open(txt, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row.get("filename", "")) + "\n")
PY
fi

for split in val test; do
  src="/vepfs/dataset/aro/ready/coco2014/coco_karpathy_${split}.json"
  copy_file_once "${src}" "${ROOT}/splits/coco_karpathy_${split}.json" || true
done

write_status_marker "coco" "attempted" "COCO 2014 train/val images, captions, and Karpathy splits prepared when available."
log "COCO done"
