from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml


LOGGER_NAME = "zero_train_diagnostics"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def write_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def append_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
        if existing_header:
            for key in existing_header:
                if key not in seen:
                    fieldnames.insert(0, key)
            for key in fieldnames:
                if key not in existing_header:
                    # Rewriting keeps CSV rectangular if a later row has a new column.
                    old = pd.read_csv(path)
                    for new_key in fieldnames:
                        if new_key not in old.columns:
                            old[new_key] = np.nan
                    new = pd.DataFrame(rows)
                    combined = pd.concat([old[fieldnames], new.reindex(columns=fieldnames)], ignore_index=True)
                    combined.to_csv(path, index=False)
                    return
            fieldnames = existing_header
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def setup_logging(log_path: Path, verbose: bool = True) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:  # pragma: no cover - environment probe
        return {"available": False, "error": repr(exc)}


def collect_environment_info(dataset_root: Path, model_root: Path, output_dir: Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - environment probe
            gpu_name = f"unavailable: {exc!r}"
    return {
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "open_clip": import_status("open_clip"),
        "transformers": import_status("transformers"),
        "dataset_root": str(dataset_root),
        "model_root": str(model_root),
        "output_dir": str(output_dir),
    }


def list_candidate_files(root: Path, exts: Iterable[str] = (".json", ".jsonl", ".csv", ".tsv")) -> list[Path]:
    if not root.exists():
        return []
    wanted = {ext.lower() for ext in exts}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted)


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = []
            for key, value in data.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("sample_id", str(key))
                    records.append(row)
                else:
                    records.append({"sample_id": str(key), "value": value})
            return records
        if isinstance(data, list):
            return [dict(item) if isinstance(item, dict) else {"value": item} for item in data]
        raise ValueError(f"Unsupported JSON root in {path}: {type(data)}")
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).to_dict("records")
    raise ValueError(f"Unsupported data file extension: {path}")


def resolve_existing_path(path: str | Path, base_dirs: Iterable[Path]) -> Path | None:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for base in base_dirs:
        candidate = base / p
        if candidate.exists():
            return candidate
    name = p.name
    for base in base_dirs:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def balanced_limit(items: list[Any], limit: int | None, key_fn) -> list[Any]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    buckets: dict[Any, list[Any]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    keys = sorted(buckets, key=lambda k: str(k))
    selected: list[Any] = []
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
        keys = [k for k in keys if buckets[k]]
        cursor += 1
    return selected


def clean_text(text: Any, max_words: int | None = None) -> str:
    value = " ".join(str(text).replace("\n", " ").split())
    if max_words is not None and max_words > 0:
        words = value.split()
        value = " ".join(words[:max_words])
    return value
