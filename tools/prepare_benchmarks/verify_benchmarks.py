#!/usr/bin/env python3

import csv
import glob
import json
import os
import random
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/vepfs/dataset/benchmark"))
MANIFEST_ROOT = BENCHMARK_ROOT / "manifests"
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(7)


def exists(path):
    return Path(path).exists()


def sample(items, n=8):
    items = list(items)
    if len(items) <= n:
        return items
    return RNG.sample(items, n)


def load_json_records(path):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "images" in data and isinstance(data["images"], list):
            return data["images"]
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("id", key)
                rows.append(row)
            else:
                rows.append({"id": key, "value": value})
        return rows
    return []


def jsonl_records(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def image_candidates(root, value, coco_prefix=None):
    if value is None:
        return []
    s = str(value)
    candidates = [Path(root) / s]
    stem = Path(s).name
    candidates.append(Path(root) / stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        n = int(digits)
        candidates.append(Path(root) / f"{n:012d}.jpg")
        if coco_prefix:
            candidates.append(Path(root) / f"{coco_prefix}_{n:012d}.jpg")
    return candidates


def any_exists(paths):
    return any(p.exists() for p in paths)


def any_image_in_roots(roots, value, coco_prefix=None):
    for root in roots:
        if any_exists(image_candidates(root, value, coco_prefix)):
            return True
    return False


def result_template(status_hint="unknown"):
    return {
        "status_hint": status_hint,
        "checked_files_exist": False,
        "num_missing_files": 0,
        "sample_read_success": False,
        "messages": [],
    }


def add_missing(res, path):
    res["num_missing_files"] += 1
    res["messages"].append(f"missing: {path}")


def verify_coco():
    res = result_template()
    root = BENCHMARK_ROOT / "coco"
    required = [
        root / "images" / "val2014",
        root / "annotations" / "captions_val2014.json",
        root / "images" / "val2017",
        root / "annotations" / "captions_val2017.json",
    ]
    for p in required:
        if not p.exists():
            add_missing(res, p)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    ann = root / "annotations" / "captions_val2014.json"
    if ann.exists():
        data = json.load(open(ann, encoding="utf-8"))
        image_by_id = {im["id"]: im["file_name"] for im in data.get("images", [])}
        rows = sample(data.get("annotations", []), 8)
        ok = 0
        for row in rows:
            fname = image_by_id.get(row.get("image_id"))
            caption_ok = bool(row.get("caption"))
            image_ok = any_exists(image_candidates(root / "images" / "val2014", fname))
            ok += int(caption_ok and image_ok)
        res["sample_read_success"] = ok == len(rows) and bool(rows)
        if rows and ok < len(rows):
            res["messages"].append(f"COCO sample image/caption checks passed {ok}/{len(rows)}")
    ann2017 = root / "annotations" / "captions_val2017.json"
    if ann2017.exists():
        data = json.load(open(ann2017, encoding="utf-8"))
        image_by_id = {im["id"]: im["file_name"] for im in data.get("images", [])}
        rows = sample(data.get("annotations", []), 8)
        ok = 0
        for row in rows:
            fname = image_by_id.get(row.get("image_id"))
            caption_ok = bool(row.get("caption"))
            image_ok = any_exists(image_candidates(root / "images" / "val2017", fname))
            ok += int(caption_ok and image_ok)
        res["sample_read_success"] = res["sample_read_success"] and ok == len(rows) and bool(rows)
        if rows and ok < len(rows):
            res["messages"].append(f"COCO2017 sample image/caption checks passed {ok}/{len(rows)}")
    return res


def verify_flickr30k():
    res = result_template()
    root = BENCHMARK_ROOT / "Flickr30k"
    images = root / "images"
    ann = root / "annotations" / "captions_karpathy.jsonl"
    fallback = root / "annotations" / "flickr30k_test.json"
    if not images.exists():
        add_missing(res, images)
    if not ann.exists() and not fallback.exists():
        add_missing(res, ann)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    rows = []
    if ann.exists():
        rows = jsonl_records(ann, 200)
    elif fallback.exists():
        rows = load_json_records(fallback)
    rows = sample(rows, 8)
    ok = 0
    for row in rows:
        image = row.get("image") or row.get("filename")
        caps = row.get("captions") or row.get("caption") or row.get("sentences")
        ok += int(bool(caps) and any_exists(image_candidates(images, image)))
    res["sample_read_success"] = ok == len(rows) and bool(rows)
    if rows and ok < len(rows):
        res["messages"].append(f"Flickr30k sample image/caption checks passed {ok}/{len(rows)}")
    return res


def verify_aro():
    res = result_template()
    root = BENCHMARK_ROOT / "aro"
    required = [
        root / "annotations" / "visual_genome_relation.json",
        root / "annotations" / "visual_genome_attribution.json",
        root / "coco_order" / "coco_karpathy_test.json",
        root / "flickr30k_order" / "flickr30k_test.json",
    ]
    for p in required:
        if not p.exists():
            add_missing(res, p)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    checks = []
    for ann, image_root in [
        (root / "annotations" / "visual_genome_relation.json", root / "vg_relation" / "images"),
        (root / "annotations" / "visual_genome_attribution.json", root / "vg_attribution" / "images"),
    ]:
        if ann.exists():
            for row in sample(load_json_records(ann), 4):
                checks.append(bool(row.get("true_caption")) and bool(row.get("false_caption")) and any_image_in_roots([BENCHMARK_ROOT / "vg" / "VG_100K", BENCHMARK_ROOT / "vg" / "VG_100K_2"], row.get("image_path")))
    for ann, image_root in [
        (root / "coco_order" / "coco_karpathy_test.json", root / "coco_order"),
        (root / "flickr30k_order" / "flickr30k_test.json", root / "flickr30k_order"),
    ]:
        if ann.exists():
            roots = [BENCHMARK_ROOT / "coco" / "images" / "val2014", BENCHMARK_ROOT / "coco" / "images" / "test2014", BENCHMARK_ROOT / "Flickr30k" / "images"]
            for row in sample(load_json_records(ann), 4):
                caps = row.get("caption") or row.get("captions")
                checks.append(bool(caps) and any_image_in_roots(roots, row.get("image"), "COCO_val2014"))
    res["sample_read_success"] = bool(checks) and all(checks)
    if checks and not all(checks):
        res["messages"].append(f"ARO sample checks passed {sum(checks)}/{len(checks)}")
    return res


def verify_sugarcrepe():
    res = result_template()
    root = BENCHMARK_ROOT / "sugarcrepe"
    anns = sorted((root / "annotations").glob("*.json")) if (root / "annotations").exists() else []
    image_root = BENCHMARK_ROOT / "coco" / "images" / "val2017"
    if not anns:
        add_missing(res, root / "annotations")
    if not image_root.exists():
        add_missing(res, image_root)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    checks = []
    for ann in anns[:7]:
        for row in sample(load_json_records(ann), 2):
            fname = row.get("filename") or row.get("image") or row.get("image_id")
            cap = row.get("caption") or row.get("positive_caption")
            neg = row.get("negative_caption") or row.get("hard_negative")
            checks.append(bool(cap) and bool(neg) and any_exists(image_candidates(image_root, fname, "COCO_val2017")))
    res["sample_read_success"] = bool(checks) and all(checks)
    if checks and not all(checks):
        res["messages"].append(f"SugarCrepe sample checks passed {sum(checks)}/{len(checks)}")
    return res


def verify_winoground():
    res = result_template()
    root = BENCHMARK_ROOT / "winoground"
    status = root / "metadata" / "download_status.json"
    marker = {}
    if status.exists():
        marker = json.load(open(status, encoding="utf-8"))
    if marker.get("status") == "gated_or_auth_required":
        res["status_hint"] = "gated_or_auth_required"
        res["checked_files_exist"] = True
        res["sample_read_success"] = False
        res["messages"].append(marker.get("note", "Winoground is gated/auth-required."))
        return res
    examples = root / "examples.jsonl"
    images = root / "images"
    if not examples.exists():
        add_missing(res, examples)
    if not images.exists():
        add_missing(res, images)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    if examples.exists():
        rows = sample(jsonl_records(examples, 50), 8)
        cap_ok = [bool(r.get("caption_0") or r.get("caption0")) and bool(r.get("caption_1") or r.get("caption1")) for r in rows]
        image_count = len(list(images.glob("*"))) if images.exists() else 0
        res["sample_read_success"] = bool(rows) and all(cap_ok) and image_count > 0
        if not res["sample_read_success"]:
            res["messages"].append(f"Winoground captions ok {sum(cap_ok)}/{len(cap_ok)}; image_count={image_count}")
    return res


def verify_svo_probes():
    res = result_template()
    path = BENCHMARK_ROOT / "svo_probes" / "annotations" / "svo_probes.csv"
    if not path.exists():
        add_missing(res, path)
    res["checked_files_exist"] = res["num_missing_files"] == 0
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = sample(list(csv.DictReader(f))[:200], 8)
        checks = [bool(r.get("sentence")) and bool(r.get("pos_url")) and bool(r.get("neg_url")) for r in rows]
        res["sample_read_success"] = bool(checks) and all(checks)
    return res


def verify_bivlc():
    res = result_template()
    root = BENCHMARK_ROOT / "bivlc"
    anns = glob.glob(str(root / "annotations" / "*"))
    status = root / "metadata" / "download_status.json"
    if not anns:
        add_missing(res, root / "annotations")
    res["checked_files_exist"] = res["num_missing_files"] == 0
    res["sample_read_success"] = bool(anns)
    if status.exists():
        try:
            marker = json.load(open(status, encoding="utf-8"))
            res["messages"].append(marker.get("note", ""))
        except Exception:
            pass
    return res


VERIFY = {
    "coco": verify_coco,
    "flickr30k": verify_flickr30k,
    "aro": verify_aro,
    "sugarcrepe": verify_sugarcrepe,
    "winoground": verify_winoground,
    "svo_probes": verify_svo_probes,
    "bivlc": verify_bivlc,
}


def main():
    report = {"benchmark_root": str(BENCHMARK_ROOT), "benchmarks": {}}
    for name, fn in VERIFY.items():
        try:
            report["benchmarks"][name] = fn()
        except Exception as exc:
            report["benchmarks"][name] = result_template("error")
            report["benchmarks"][name]["messages"].append(f"verification exception: {exc}")
    path = MANIFEST_ROOT / "verify_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"verify report: {path}")
    for name, item in report["benchmarks"].items():
        state = "ok" if item["checked_files_exist"] and item["sample_read_success"] else "needs_attention"
        if item.get("status_hint") == "gated_or_auth_required":
            state = "gated_or_auth_required"
        print(f"{name}: {state}; missing={item['num_missing_files']}; messages={'; '.join(item['messages'][:2])}")


if __name__ == "__main__":
    main()
