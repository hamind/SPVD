#!/usr/bin/env bash

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ROOT="${BENCHMARK_ROOT}/bivlc"
RAW="${RAW_ROOT}/bivlc"
mkdir -p "${ROOT}/annotations" "${ROOT}/images_or_links" "${ROOT}/metadata" "${RAW}"

log "prepare BiVLC under ${ROOT}"

python3 - "${ROOT}" "${RAW}" <<'PY'
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
raw = Path(sys.argv[2])
status_path = root / "metadata" / "download_status.json"

def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def status(value, note):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark": "bivlc", "status": value, "note": note, "time": now()}, f, indent=2)
        f.write("\n")

def request_json(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dst)

try:
    endpoints = []
    if os.environ.get("HF_ENDPOINT"):
        endpoints.append(os.environ["HF_ENDPOINT"].rstrip("/"))
    endpoints.extend(["https://huggingface.co", "https://hf-mirror.com"])
    seen = set()
    endpoints = [x for x in endpoints if not (x in seen or seen.add(x))]
    parquet_index = None
    api = None
    endpoint = None
    errors = []
    for endpoint in endpoints:
        api = f"{endpoint}/api/datasets/imirandam/BiVLC/parquet"
        try:
            parquet_index = request_json(api)
            break
        except Exception as exc:
            errors.append(f"{api}: {exc}")
            parquet_index = None
    if parquet_index is None:
        raise RuntimeError("; ".join(errors))
    urls = []
    def collect(obj):
        if isinstance(obj, str) and obj.startswith("https://"):
            urls.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                collect(item)
    collect(parquet_index)
    urls = sorted(set(url.replace("https://huggingface.co", endpoint) for url in urls))
    if not urls:
        raise RuntimeError(f"no parquet URLs returned from {api}: {parquet_index}")
    index_path = root / "annotations" / "parquet_urls.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"source": api, "urls": urls}, f, indent=2)
        f.write("\n")
    for idx, url in enumerate(urls):
        name = url.rsplit("/", 1)[-1] or f"{idx:04d}.parquet"
        if not name.endswith(".parquet"):
            name = f"{idx:04d}.parquet"
        download(url, root / "annotations" / name)
    status("attempted", f"Downloaded BiVLC parquet files directly from Hugging Face API ({len(urls)} files).")
except Exception as exc:
    status("partial", f"BiVLC direct Hugging Face parquet download failed: {exc}. Manual source: https://huggingface.co/datasets/imirandam/BiVLC")
    raise SystemExit(0)
PY

log "BiVLC done"
