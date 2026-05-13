#!/usr/bin/env python3

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/vepfs/dataset/benchmark"))
MANIFEST_ROOT = BENCHMARK_ROOT / "manifests"
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def count_files(path: Path, suffixes=None):
    if not path.exists():
        return 0
    total = 0
    for p in path.iterdir():
        if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes):
            total += 1
    return total


def file_list(paths):
    return [str(p) for p in paths if p.exists()]


def read_status_marker(name):
    path = BENCHMARK_ROOT / name / "metadata" / "download_status.json"
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def git_commit(path: Path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def verification_for(name):
    report_path = MANIFEST_ROOT / "verify_report.json"
    default = {"checked_files_exist": False, "num_missing_files": None, "sample_read_success": False}
    if not report_path.exists():
        return default
    try:
        report = json.load(open(report_path, encoding="utf-8"))
        item = report.get("benchmarks", {}).get(name, {})
        return {
            "checked_files_exist": bool(item.get("checked_files_exist", False)),
            "num_missing_files": int(item.get("num_missing_files", 0)) if item.get("num_missing_files") is not None else None,
            "sample_read_success": bool(item.get("sample_read_success", False)),
        }
    except Exception:
        return default


def base_manifest(name, license_or_terms, urls):
    root = BENCHMARK_ROOT / name
    marker = read_status_marker(name)
    return {
        "name": name,
        "status": marker.get("status", "failed"),
        "version_or_commit": None,
        "source_urls": urls,
        "local_path": str(root),
        "download_time": marker.get("time", now()),
        "license_or_terms": license_or_terms,
        "num_images": None,
        "num_texts": None,
        "num_instances": None,
        "required_external_data": [],
        "image_root": None,
        "annotation_files": [],
        "split_files": [],
        "notes": marker.get("note", ""),
        "verification": verification_for(name),
    }


def write_manifest(item):
    root = Path(item["local_path"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, ensure_ascii=True)
        f.write("\n")


def coco():
    item = base_manifest(
        "coco",
        "COCO annotations: Creative Commons Attribution 4.0; images retain Flickr image licenses/Flickr Terms of Use.",
        [
            "https://cocodataset.org/#download",
            "http://images.cocodataset.org/zips/train2014.zip",
            "http://images.cocodataset.org/zips/val2014.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
        ],
    )
    root = BENCHMARK_ROOT / "coco"
    train = count_files(root / "images" / "train2014", {".jpg", ".jpeg"})
    val = count_files(root / "images" / "val2014", {".jpg", ".jpeg"})
    val2017 = count_files(root / "images" / "val2017", {".jpg", ".jpeg"})
    item["num_images"] = train + val
    item["image_root"] = str(root / "images")
    item["annotation_files"] = file_list([
        root / "annotations" / "captions_train2014.json",
        root / "annotations" / "captions_val2014.json",
        root / "annotations" / "captions_val2017.json",
        root / "annotations" / "dataset_coco_karpathy.json",
    ])
    item["split_files"] = file_list(list((root / "splits").glob("*")) if (root / "splits").exists() else [])
    item["required_external_data"] = []
    if train >= 80000 and val >= 40000 and len(item["annotation_files"]) >= 2:
        item["status"] = "ready"
    elif train or val or item["annotation_files"]:
        item["status"] = "partial"
    else:
        item["status"] = "failed"
    item["notes"] += f" Counts: train2014={train}, val2014={val}, val2017={val2017}. Karpathy retrieval normally uses COCO 2014 train/val; COCO 2017 val is also prepared for modern val-retrieval/SugarCrepe reuse."
    return item


def flickr30k():
    item = base_manifest(
        "flickr30k",
        "Captions are Creative Commons Attribution-ShareAlike per the UIUC page; images are from Flickr and provided for non-commercial research/education, subject to Flickr Terms of Use.",
        [
            "https://shannon.cs.illinois.edu/DenotationGraph/",
            "https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html",
            "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
        ],
    )
    root = BENCHMARK_ROOT / "Flickr30k"
    images = count_files(root / "images", {".jpg", ".jpeg", ".png"})
    item["num_images"] = images
    item["image_root"] = str(root / "images")
    item["annotation_files"] = file_list(list((root / "annotations").glob("*")) if (root / "annotations").exists() else [])
    item["split_files"] = file_list(list((root / "splits").glob("*")) if (root / "splits").exists() else [])
    if images >= 31000 and item["annotation_files"] and item["split_files"]:
        item["status"] = "ready"
    elif images or item["annotation_files"]:
        item["status"] = "partial"
        item["required_external_data"] = [] if images else ["Authorized Flickr30k image download from UIUC request form"]
    else:
        item["status"] = "failed"
        item["required_external_data"] = ["Authorized Flickr30k image download from UIUC request form"]
    item["notes"] += f" Images={images}. Official images are not automatically redistributable."
    return item


def aro():
    item = base_manifest(
        "aro",
        "Official ARO code is MIT; VG subset, COCO, and Flickr30k underlying assets keep their original dataset terms.",
        [
            "https://github.com/mertyg/vision-language-models-are-bows",
            "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
            "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json",
        ],
    )
    root = BENCHMARK_ROOT / "aro"
    repo = BENCHMARK_ROOT / "_raw" / "aro" / "vision-language-models-are-bows"
    item["version_or_commit"] = git_commit(repo)
    vg_images = count_files(root / "vg_relation" / "images", {".jpg", ".jpeg", ".png"})
    anns = [
        root / "annotations" / "visual_genome_relation.json",
        root / "annotations" / "visual_genome_attribution.json",
        root / "coco_order" / "coco_karpathy_val.json",
        root / "coco_order" / "coco_karpathy_test.json",
        root / "flickr30k_order" / "flickr30k_val.json",
        root / "flickr30k_order" / "flickr30k_test.json",
    ]
    item["num_images"] = vg_images
    item["image_root"] = str(root)
    item["annotation_files"] = file_list(anns)
    item["split_files"] = file_list(anns[2:])
    item["required_external_data"] = ["COCO images from benchmarks/coco", "Flickr30k images from benchmarks/flickr30k"]
    item["status"] = "ready" if len(item["annotation_files"]) == len(anns) and vg_images > 0 else ("partial" if item["annotation_files"] else "failed")
    item["notes"] += f" VG images={vg_images}; COCO/Flickr order image roots are symlinks when available."
    return item


def sugarcrepe():
    item = base_manifest(
        "sugarcrepe",
        "Official SugarCrepe work/code is MIT; examples are derived from COCO and prior public benchmark data under their original terms.",
        ["https://github.com/RAIVNLab/sugar-crepe", "https://arxiv.org/abs/2306.14610"],
    )
    root = BENCHMARK_ROOT / "sugarcrepe"
    repo = BENCHMARK_ROOT / "_raw" / "sugarcrepe" / "sugar-crepe"
    anns = list((root / "annotations").glob("*.json")) if (root / "annotations").exists() else []
    val2017 = count_files(BENCHMARK_ROOT / "coco" / "images" / "val2017", {".jpg", ".jpeg"})
    item["version_or_commit"] = git_commit(repo)
    item["num_images"] = val2017
    item["num_instances"] = sum(count_json_records(p) for p in anns)
    item["image_root"] = str(root / "coco_val2017_images")
    item["annotation_files"] = file_list(anns)
    item["split_files"] = file_list([root / "splits" / "categories.txt"])
    item["required_external_data"] = ["COCO 2017 val images, referenced via benchmarks/coco/images/val2017"]
    item["status"] = "ready" if len(anns) >= 7 and val2017 >= 5000 else ("partial" if anns else "failed")
    item["notes"] += f" Categories={len(anns)}, COCO val2017 images={val2017}."
    return item


def count_json_records(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
        return len(data)
    except Exception:
        return 0


def count_jsonl(path):
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except Exception:
        return 0


def winoground():
    item = base_manifest(
        "winoground",
        "Hugging Face gated dataset; user must accept terms, share contact info, and use solely for research. Full license agreement is in dataset files.",
        ["https://huggingface.co/datasets/facebook/winoground", "https://arxiv.org/abs/2204.03162"],
    )
    root = BENCHMARK_ROOT / "winoground"
    images = count_files(root / "images", {".jpg", ".jpeg", ".png", ".webp"})
    examples = root / "examples.jsonl"
    marker = read_status_marker("winoground")
    item["num_images"] = images
    item["num_instances"] = count_jsonl(examples)
    item["image_root"] = str(root / "images")
    item["annotation_files"] = file_list([examples])
    if marker.get("status") == "gated_or_auth_required" and not examples.exists():
        item["status"] = "gated_or_auth_required"
    elif images and examples.exists():
        item["status"] = "ready"
    elif marker:
        item["status"] = marker.get("status", "partial")
    else:
        item["status"] = "failed"
    item["notes"] += f" Images={images}, examples={item['num_instances']}."
    return item


def svo_probes():
    item = base_manifest(
        "svo_probes",
        "Data is CC BY 4.0; repository code is Apache-2.0. Images are external URLs from Google Image Search and keep upstream rights.",
        ["https://github.com/google-deepmind/svo_probes", "https://deepmind.google/blog/probing-image-language-transformers-for-verb-understanding/"],
    )
    root = BENCHMARK_ROOT / "svo_probes"
    csv_path = root / "annotations" / "svo_probes.csv"
    rows = 0
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = sum(1 for _ in csv.DictReader(f))
        except Exception:
            rows = 0
    item["num_instances"] = rows
    item["annotation_files"] = file_list([csv_path])
    item["image_root"] = str(root / "images_or_links")
    item["required_external_data"] = ["External image URLs listed in images_or_links/image_urls.txt"]
    item["status"] = "ready" if rows else ("partial" if csv_path.exists() else "failed")
    return item


def bivlc():
    item = base_manifest(
        "bivlc",
        "Dataset card states MIT License; COCO-derived positive images and generated images should be used with upstream dataset terms in mind.",
        ["https://huggingface.co/datasets/imirandam/BiVLC", "https://imirandam.github.io/BiVLC_project_page", "https://arxiv.org/abs/2406.09952"],
    )
    root = BENCHMARK_ROOT / "bivlc"
    anns = list((root / "annotations").glob("*")) if (root / "annotations").exists() else []
    item["annotation_files"] = file_list(anns)
    item["image_root"] = str(root / "images_or_links")
    marker = read_status_marker("bivlc")
    item["status"] = "ready" if anns else marker.get("status", "failed")
    item["notes"] += " BiVLC is downloaded directly from Hugging Face parquet URLs exposed by /api/datasets/imirandam/BiVLC/parquet; no huggingface_hub dependency is required."
    return item


BUILDERS = [coco, flickr30k, aro, sugarcrepe, winoground, svo_probes, bivlc]


def main():
    manifests = []
    for builder in BUILDERS:
        item = builder()
        write_manifest(item)
        manifests.append(item)
    out = {
        "generated_at": now(),
        "benchmark_root": str(BENCHMARK_ROOT),
        "benchmarks": manifests,
    }
    with open(MANIFEST_ROOT / "benchmarks_manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
        f.write("\n")
    summary = {}
    for item in manifests:
        summary.setdefault(item["status"], []).append(item["name"])
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
