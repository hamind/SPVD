#!/usr/bin/env python3

import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/vepfs/dataset/benchmark"))
RAW_ROOT = Path(os.environ.get("RAW_ROOT", str(BENCHMARK_ROOT / "_raw")))
ROOT = BENCHMARK_ROOT / "winoground"
RAW = RAW_ROOT / "winoground"
SOURCE_URL = "https://huggingface.co/datasets/facebook/winoground"


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def ensure_dirs():
    for p in [ROOT, ROOT / "metadata", RAW]:
        p.mkdir(parents=True, exist_ok=True)


def write_status(status, note):
    payload = {"benchmark": "winoground", "status": status, "note": note, "source_url": SOURCE_URL, "time": now()}
    with open(ROOT / "metadata" / "download_status.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def link_once(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return True
    os.symlink(src, dst, target_is_directory=src.is_dir())
    log(f"linked {dst} -> {src}")
    return True


def try_existing():
    for src_root in [
        Path("/vepfs/dataset/winoground/ready/facebook-winoground"),
        Path("/vepfs/dataset/winoground/ready/asphyxia-flattened"),
    ]:
        if not src_root.exists():
            continue
        image_src = src_root / "images"
        ann_src = src_root / "examples.jsonl"
        if not ann_src.exists():
            ann_src = src_root / "annotations.jsonl"
        ok = False
        if image_src.exists():
            ok = link_once(image_src, ROOT / "images") or ok
        if ann_src.exists():
            ok = link_once(ann_src, ROOT / "examples.jsonl") or ok
        if ok:
            write_status("ready", f"Reused existing local authorized copy from {src_root}.")
            return True
    return False


def try_huggingface():
    token = os.environ.get("HF_TOKEN")
    if not token:
        write_status("gated_or_auth_required", "HF_TOKEN is not set, and facebook/winoground requires accepted Hugging Face access terms.")
        return False
    def download(url: str, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            return
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dst)

    try:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        examples_url = f"{endpoint}/datasets/facebook/winoground/resolve/main/data/examples.jsonl"
        images_url = f"{endpoint}/datasets/facebook/winoground/resolve/main/data/images.zip"
        examples = RAW / "facebook-winoground" / "examples.jsonl"
        images_zip = RAW / "facebook-winoground" / "images.zip"
        download(examples_url, examples)
        download(images_url, images_zip)
    except Exception as exc:
        write_status("gated_or_auth_required", f"Direct Hugging Face download failed, likely gated/auth-related: {exc}")
        return False

    shutil.copy2(examples, ROOT / "examples.jsonl")
    if not (ROOT / "images").exists():
        (ROOT / "images").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(ROOT / "images")

    write_status("ready", "Downloaded facebook/winoground directly from Hugging Face file URLs with HF_TOKEN.")
    return True


def main():
    ensure_dirs()
    if try_existing():
        return 0
    if try_huggingface():
        return 0
    log("Winoground is gated or unavailable; continuing without failing the full benchmark preparation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
