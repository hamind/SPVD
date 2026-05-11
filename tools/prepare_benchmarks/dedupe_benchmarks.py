#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/vepfs/dataset/benchmark"))
MANIFEST_ROOT = BENCHMARK_ROOT / "manifests"


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_pairs():
    root = BENCHMARK_ROOT
    pairs = [
        (Path("/vepfs/dataset/aro/ready/vg/visual_genome_relation.json"), root / "aro/annotations/visual_genome_relation.json"),
        (Path("/vepfs/dataset/aro/ready/vg/visual_genome_attribution.json"), root / "aro/annotations/visual_genome_attribution.json"),
        (Path("/vepfs/dataset/aro/ready/coco2014/coco_karpathy_val.json"), root / "coco/splits/coco_karpathy_val.json"),
        (Path("/vepfs/dataset/aro/ready/coco2014/coco_karpathy_test.json"), root / "coco/splits/coco_karpathy_test.json"),
        (Path("/vepfs/dataset/aro/ready/coco2014/coco_karpathy_val.json"), root / "aro/coco_order/coco_karpathy_val.json"),
        (Path("/vepfs/dataset/aro/ready/coco2014/coco_karpathy_test.json"), root / "aro/coco_order/coco_karpathy_test.json"),
        (Path("/vepfs/dataset/aro/ready/flickr30k/flickr30k_val.json"), root / "flickr30k/annotations/flickr30k_val.json"),
        (Path("/vepfs/dataset/aro/ready/flickr30k/flickr30k_test.json"), root / "flickr30k/annotations/flickr30k_test.json"),
        (Path("/vepfs/dataset/aro/ready/flickr30k/flickr30k_val.json"), root / "aro/flickr30k_order/flickr30k_val.json"),
        (Path("/vepfs/dataset/aro/ready/flickr30k/flickr30k_test.json"), root / "aro/flickr30k_order/flickr30k_test.json"),
    ]
    sugar_src = Path("/vepfs/dataset/sugarcrepe/ready/data")
    sugar_dst = root / "sugarcrepe/annotations"
    for name in ["replace_rel.json", "swap_att.json", "add_obj.json", "replace_obj.json", "replace_att.json", "add_att.json", "swap_obj.json"]:
        pairs.append((sugar_src / name, sugar_dst / name))
    return pairs


def raw_archive_report():
    raw = BENCHMARK_ROOT / "_raw"
    archives = []
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
            archives.append({"path": str(path), "size_bytes": path.stat().st_size})
    return sorted(archives, key=lambda item: item["size_bytes"], reverse=True)


def dedupe_pair(src: Path, dst: Path, apply: bool):
    result = {
        "source": str(src),
        "target": str(dst),
        "action": "skip",
        "reason": "",
        "size_bytes": None,
    }
    if not src.exists():
        result["reason"] = "source_missing"
        return result
    if not dst.exists() and not dst.is_symlink():
        result["reason"] = "target_missing"
        return result
    if dst.is_symlink():
        target = os.readlink(dst)
        result["action"] = "already_symlink"
        result["reason"] = target
        return result
    if not dst.is_file() or not src.is_file():
        result["reason"] = "not_regular_file"
        return result
    result["size_bytes"] = dst.stat().st_size
    if src.stat().st_size != dst.stat().st_size:
        result["reason"] = "size_differs"
        return result
    if sha256(src) != sha256(dst):
        result["reason"] = "sha256_differs"
        return result
    result["action"] = "would_symlink"
    result["reason"] = "identical"
    if apply:
        dst.unlink()
        os.symlink(src, dst)
        result["action"] = "symlinked"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Replace verified duplicate files with symlinks.")
    args = parser.parse_args()

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    results = [dedupe_pair(src, dst, args.apply) for src, dst in candidate_pairs()]
    reclaimed = sum(item.get("size_bytes") or 0 for item in results if item["action"] == "symlinked")
    report = {
        "generated_at": now(),
        "mode": "apply" if args.apply else "dry-run",
        "benchmark_root": str(BENCHMARK_ROOT),
        "reclaimed_duplicate_file_bytes": reclaimed,
        "file_dedupe": results,
        "raw_archives_kept": raw_archive_report(),
        "notes": [
            "Only byte-identical annotation copies are replaced with symlinks.",
            "Large raw archives are reported but kept by default; remove them manually only after deciding that redownload/resume archives are not needed.",
        ],
    }
    out = MANIFEST_ROOT / "dedupe_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"dedupe report: {out}")
    for item in results:
        print(f"{item['action']}: {item['target']} <- {item['source']} ({item['reason']})")
    if args.apply:
        print(f"reclaimed duplicate regular-file bytes: {reclaimed}")


if __name__ == "__main__":
    main()

