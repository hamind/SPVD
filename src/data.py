"""Image-text datasets and dataloaders."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import random
import re
import sqlite3
import tarfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, get_worker_info

from open_clip_train.data import ResampledShards2, SharedEpoch, detshuffle2, get_dataset_size
from open_clip_train.data import group_by_keys_nothrow, tar_file_expander, url_opener

from distributed import get_rank, get_world_size


_NUMERIC_BRACE_RE = re.compile(r"\{(\d+)\.\.(\d+)(?:\.\.(-?\d+))?\}")


@dataclass
class DataInfo:
    """Dataloader wrapper with epoch coordination hooks."""

    dataloader: DataLoader[Any]
    sampler: DistributedSampler[Any] | None = None
    shared_epoch: Any | None = None

    def set_epoch(self, epoch: int) -> None:
        if self.shared_epoch is not None and hasattr(self.shared_epoch, "set_value"):
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:</s>|\r?\n+|[.!?。！？]+)")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{12,}")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_FLAIR_CAPTION_GROUP_KEYS = (
    "raw",
    "shortIB",
    "longIB",
    "shortSV",
    "longSV",
    "shortLLA",
    "longLLA",
)
_FLAIR_TOP_LEVEL_CAPTION_KEYS = (
    "raw_caption",
    "shortIB_captions",
    "longIB_captions",
    "shortSV_captions",
    "longSV_captions",
    "shortLLA_captions",
    "longLLA_captions",
)


def expand_urls(urls: str | list[str], weights: str | list[float] | None = None) -> tuple[list[str], list[float] | None]:
    """Expand WebDataset URLs, copied from FLAIR's data pipeline."""
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        import braceexpand

        urllist = urls.split("::")
        weight_list = str(weights).split("::")
        assert len(weight_list) == len(urllist), (
            f"Expected the number of data components ({len(urllist)}) and weights({len(weight_list)}) to match."
        )
        float_weights = [float(weight) for weight in weight_list]
        all_urls: list[str] = []
        all_weights: list[float] = []
        for url, weight in zip(urllist, float_weights, strict=True):
            expanded_url = list(braceexpand.braceexpand(url))
            all_urls.extend(expanded_url)
            all_weights.extend([weight for _ in expanded_url])
        return all_urls, all_weights
    return list(urls), weights


def filter_no_caption_or_no_image(sample: dict[str, Any]) -> bool:
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


def filter_no_caption_or_no_image_json(sample: dict[str, Any]) -> bool:
    has_caption = "json" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


def _clean_sentence(sentence: Any) -> str:
    text = str(sentence or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'`")
    return text


def _looks_like_valid_sentence(sentence: str, min_chars: int = 8, min_words: int = 3, max_chars: int = 320) -> bool:
    if not sentence:
        return False
    if len(sentence) < min_chars or len(sentence) > max_chars:
        return False
    if _URL_RE.search(sentence):
        return False
    if _REPEATED_CHAR_RE.search(sentence):
        return False
    alnum = sum(ch.isalnum() for ch in sentence)
    if alnum < max(3, min_chars // 2):
        return False
    if alnum / max(len(sentence), 1) < 0.35:
        return False
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", sentence)
    return len(words) >= min_words or sum("\u4e00" <= ch <= "\u9fff" for ch in sentence) >= 4


def split_caption(text: str | list[Any] | tuple[Any, ...]) -> list[str]:
    """Split and filter long captions, following FLAIR's dynamic caption sampling setup."""
    if isinstance(text, (list, tuple)):
        raw_candidates = [_clean_sentence(item) for item in text]
    else:
        raw_candidates = [_clean_sentence(item) for item in _SENTENCE_BOUNDARY_RE.split(str(text or ""))]
    cleaned = [item for item in raw_candidates if item]
    filtered = [item for item in cleaned if _looks_like_valid_sentence(item)]
    if filtered:
        return filtered
    if cleaned:
        return cleaned
    fallback = _clean_sentence(text)
    return [fallback] if fallback else []


def _draw_sentence_indices(num_sentences: int, count: int) -> list[int]:
    population = list(range(num_sentences))
    if num_sentences >= count:
        return random.sample(population, count)
    return random.choices(population, k=count)


def _sample_sentence_span(sentences: list[str], span_size: int) -> list[str]:
    if len(sentences) >= span_size:
        start = random.randint(0, len(sentences) - span_size)
        return sentences[start : start + span_size]
    return random.choices(sentences, k=span_size)


def sample_subcaptions(
    caption_or_sentences: str | list[Any] | tuple[Any, ...],
    k: int = 4,
    max_merged_num: int = 3,
) -> list[str]:
    """Dynamically sample K sub-captions from one long caption or sentence list."""
    sentences = [sentence for sentence in split_caption(caption_or_sentences) if sentence]
    if not sentences:
        sentences = [""]
    k = max(int(k), 1)
    max_merged_num = max(int(max_merged_num), 1)
    subcaptions: list[str] = []
    for _ in range(k):
        span_size = random.randint(1, max_merged_num)
        if span_size == 1:
            selected = [random.choice(sentences)]
        elif random.random() < 0.5:
            selected = _sample_sentence_span(sentences, span_size)
        else:
            selected = [sentences[index] for index in _draw_sentence_indices(len(sentences), span_size)]
        subcaptions.append(".\n".join(selected))
    return subcaptions


def log_and_continue(exn: Exception) -> bool:
    """Ignore a WebDataset decoding/tar error and continue, matching FLAIR."""
    logging.warning("Handling webdataset error (%r). Ignoring.", exn)
    return True


def tarfile_to_samples_nothrow(src: Any, handler: Callable[[Exception], bool] = log_and_continue) -> Any:
    """OpenCLIP/FLAIR tar expansion with EOF markers disabled for this WebDataset version."""
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler, eof_value=None)
    return group_by_keys_nothrow(files, handler=handler)


def _fallback_braceexpand(pattern: str) -> list[str]:
    """Expand simple numeric braces without requiring the braceexpand package."""
    match = _NUMERIC_BRACE_RE.search(pattern)
    if match is None:
        return [pattern]
    start_s, end_s, step_s = match.groups()
    start, end = int(start_s), int(end_s)
    step = int(step_s) if step_s is not None else (1 if end >= start else -1)
    if step == 0 or (end - start) * step < 0:
        return [pattern]
    width = max(len(start_s), len(end_s))
    stop = end + (1 if step > 0 else -1)
    expanded: list[str] = []
    for value in range(start, stop, step):
        replacement = f"{value:0{width}d}" if width > 1 else str(value)
        expanded.extend(_fallback_braceexpand(pattern[: match.start()] + replacement + pattern[match.end() :]))
    return expanded


def _expand_shards(pattern: str | list[str]) -> list[str]:
    """Expand WebDataset brace and glob shard patterns."""
    if isinstance(pattern, list):
        out: list[str] = []
        for item in pattern:
            out.extend(_expand_shards(item))
        return sorted(dict.fromkeys(out))
    try:
        from braceexpand import braceexpand
        expanded = list(braceexpand(pattern))
    except ImportError:
        expanded = _fallback_braceexpand(pattern)
    paths: list[str] = []
    for item in expanded:
        import glob
        matches = sorted(glob.glob(item))
        if not matches and not Path(item).is_absolute():
            matches = sorted(str(path) for path in Path().glob(item))
        paths.extend(matches or [item])
    return sorted(dict.fromkeys(paths))


def _split_caption_columns(value: Any, fallback: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _sqlite_uri(path: Path, mode: str = "ro") -> str:
    return f"file:{path.as_posix()}?mode={mode}"


class CaptionRelabeler:
    """Fast URL-to-caption lookup backed by a one-time SQLite index."""

    def __init__(
        self,
        caption_file: str | Path,
        caption_key: str = "longSV_captions",
        index_path: str | Path | None = None,
        file_key: str = "Image Path",
        fallback_keys: list[str] | tuple[str, ...] | str | None = None,
        sample_key: str = "url",
        missing: str = "fallback",
        build_index: bool = True,
        metadata_key: str = "json",
        lock_timeout: int = 3600,
    ) -> None:
        self.caption_file = Path(caption_file)
        self.file_key = str(file_key)
        fallback = [
            "longSV_captions",
            "longIB_captions",
            "longLLA_captions",
            "shortSV_captions",
            "shortIB_captions",
            "shortLLA_captions",
            "raw_caption",
        ]
        columns = _split_caption_columns(caption_key)
        for column in _split_caption_columns(fallback_keys, fallback):
            if column not in columns:
                columns.append(column)
        self.caption_columns = columns
        self.sample_key = str(sample_key).lower()
        if self.sample_key not in {"url", "key", "caption"}:
            raise ValueError(f"Unsupported caption relabel sample key: {sample_key}")
        self.missing = str(missing).lower()
        if self.missing not in {"fallback", "skip", "error"}:
            raise ValueError(f"Unsupported caption relabel missing policy: {missing}")
        self.metadata_key = metadata_key.lower().lstrip(".")
        self.index_path = Path(index_path) if index_path else self._default_index_path()
        self._conn: sqlite3.Connection | None = None
        if build_index:
            self.ensure_index(lock_timeout=lock_timeout)
        elif not self.index_path.exists():
            raise FileNotFoundError(f"Caption relabel index not found: {self.index_path}")

    def _default_index_path(self) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.caption_columns[0] if self.caption_columns else "caption")
        return self.caption_file.with_name(f"{self.caption_file.stem}.{safe_name}.sqlite")

    @property
    def needs_metadata(self) -> bool:
        return self.sample_key == "url"

    def ensure_index(self, lock_timeout: int = 3600) -> None:
        ready_path = self.index_path.with_suffix(self.index_path.suffix + ".ready")
        if self.index_path.exists() and ready_path.exists():
            return
        if not self.caption_file.exists():
            raise FileNotFoundError(f"Caption relabel CSV not found: {self.caption_file}")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.index_path.with_suffix(self.index_path.suffix + ".lock")
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if self.index_path.exists() and ready_path.exists():
                    return
                if time.time() - start > lock_timeout:
                    raise TimeoutError(f"Timed out waiting for caption relabel index lock: {lock_path}")
                time.sleep(5)

        tmp_path = self.index_path.with_suffix(self.index_path.suffix + f".tmp.{os.getpid()}")
        try:
            self._build_index(tmp_path)
            os.replace(tmp_path, self.index_path)
            ready_path.write_text("ready\n", encoding="utf-8")
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _select_caption(self, row: dict[str, str]) -> str:
        for column in self.caption_columns:
            value = row.get(column)
            if value and value.strip():
                return value.strip()
        return ""

    def _build_index(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("CREATE TABLE captions (lookup_key TEXT PRIMARY KEY, caption TEXT NOT NULL)")
            batch: list[tuple[str, str]] = []
            with self.caption_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if self.file_key not in (reader.fieldnames or []):
                    raise KeyError(f"Caption relabel key column not found: {self.file_key}")
                missing_columns = [column for column in self.caption_columns if column not in (reader.fieldnames or [])]
                if len(missing_columns) == len(self.caption_columns):
                    raise KeyError(f"None of the caption columns were found: {self.caption_columns}")
                for row in reader:
                    lookup_key = (row.get(self.file_key) or "").strip()
                    caption = self._select_caption(row)
                    if not lookup_key or not caption:
                        continue
                    batch.append((lookup_key, caption))
                    if len(batch) >= 50000:
                        conn.executemany("INSERT OR REPLACE INTO captions VALUES (?, ?)", batch)
                        conn.commit()
                        batch.clear()
            if batch:
                conn.executemany("INSERT OR REPLACE INTO captions VALUES (?, ?)", batch)
                conn.commit()
            conn.execute("PRAGMA optimize")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Caption relabel index not found: {self.index_path}")
            self._conn = sqlite3.connect(_sqlite_uri(self.index_path, "ro"), uri=True, check_same_thread=False)
            self._conn.execute("PRAGMA query_only=ON")
        return self._conn

    def lookup(self, lookup_key: str | None) -> str | None:
        if not lookup_key:
            return None
        row = self._connect().execute("SELECT caption FROM captions WHERE lookup_key = ?", (lookup_key,)).fetchone()
        return str(row[0]) if row else None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_conn"] = None
        return state


class ManifestImageTextDataset(Dataset[dict[str, Any]]):
    """Load image-text pairs from CSV, TSV, JSON, JSONL, or Parquet manifests."""

    def __init__(
        self,
        manifest_file: str | Path,
        image_root: str | Path | None,
        image_key: str,
        caption_key: str,
        transform: Callable[[Image.Image], torch.Tensor],
        file_format: str | None = None,
        id_key: str | None = None,
    ) -> None:
        self.manifest_file = Path(manifest_file)
        self.image_root = Path(image_root) if image_root else None
        self.image_key = image_key
        self.caption_key = caption_key
        self.id_key = id_key
        self.transform = transform
        self.rows = self._read_rows(file_format)

    def _read_rows(self, file_format: str | None) -> list[dict[str, Any]]:
        suffix = (file_format or self.manifest_file.suffix.lstrip(".")).lower()
        if suffix == "csv":
            with self.manifest_file.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        if suffix == "tsv":
            with self.manifest_file.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle, delimiter="\t"))
        if suffix == "jsonl":
            with self.manifest_file.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if suffix == "json":
            payload = json.loads(self.manifest_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else payload.get("data", [])
        if suffix == "parquet":
            import pandas as pd
            return pd.read_parquet(self.manifest_file).to_dict("records")
        raise ValueError(f"Unsupported manifest format: {suffix}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve_image(str(row[self.image_key]))
        caption = str(row[self.caption_key])
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        sample_id = str(row.get(self.id_key, index)) if self.id_key else str(index)
        return {
            "image": image_tensor,
            "caption": caption,
            "image_id": str(row.get(self.image_key, image_path)),
            "sample_id": sample_id,
        }


class SyntheticImageTextDataset(Dataset[dict[str, Any]]):
    """Tiny synthetic dataset for dry tests."""

    def __init__(self, image_size: int = 224, length: int = 128) -> None:
        self.image_size = image_size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "caption": f"synthetic sample {index}",
            "image_id": str(index),
            "sample_id": str(index),
        }


def _collate(samples: list[dict[str, Any]], tokenizer: Callable[[list[str]], torch.Tensor]) -> dict[str, Any]:
    images = torch.stack([sample["image"] for sample in samples], dim=0)
    captions = [str(sample["caption"]) for sample in samples]
    return {
        "image": images,
        "text": tokenizer(captions),
        "caption": captions,
        "image_id": [sample.get("image_id") for sample in samples],
        "sample_id": [sample.get("sample_id") for sample in samples],
    }


def _get_first_attr(args: object, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = getattr(args, name, None)
        if value not in (None, ""):
            return value
    return default


def _build_caption_relabeler(args: object, is_train: bool) -> CaptionRelabeler | None:
    caption_file = _get_first_attr(args, ("caption_relabel_file", "long_caption_file"))
    if not caption_file:
        return None
    split_enabled = bool(getattr(args, "caption_relabel_train", True)) if is_train else bool(getattr(args, "caption_relabel_val", True))
    if not split_enabled:
        return None
    return CaptionRelabeler(
        caption_file=caption_file,
        caption_key=str(_get_first_attr(args, ("caption_relabel_caption_key", "long_caption_caption_key"), "longSV_captions")),
        index_path=_get_first_attr(args, ("caption_relabel_index", "long_caption_index")),
        file_key=str(_get_first_attr(args, ("caption_relabel_file_key", "long_caption_file_key"), "Image Path")),
        fallback_keys=_get_first_attr(args, ("caption_relabel_fallback_keys", "long_caption_fallback_keys"), None),
        sample_key=str(_get_first_attr(args, ("caption_relabel_sample_key", "long_caption_sample_key"), "url")),
        missing=str(_get_first_attr(args, ("caption_relabel_missing", "long_caption_missing"), "fallback")),
        build_index=bool(_get_first_attr(args, ("caption_relabel_build_index", "long_caption_build_index"), True)),
    )


def _filter_relabel_success_json(sample: dict[str, Any], relabel_success_key: str = "longSV", strict_caption_match: bool = True) -> bool:
    metadata = sample.get("json")
    if not isinstance(metadata, dict):
        return False
    captions = metadata.get("captions")
    if not isinstance(captions, dict):
        return False
    expected = captions.get(relabel_success_key)
    if not split_caption(expected):
        return False
    default_key = metadata.get("default_caption_key")
    if default_key is not None and str(default_key) != relabel_success_key:
        return False
    if strict_caption_match and isinstance(expected, str):
        raw_text = sample.get("txt")
        if not isinstance(raw_text, str) or raw_text.strip() != expected.strip():
            return False
    return True


def _extend_caption_pool(pool: list[str], value: Any) -> None:
    pool.extend(sentence for sentence in split_caption(value) if sentence)


def _flair_caption_pool_from_json(text: dict[str, Any], relabel_success_key: str = "longSV") -> list[str]:
    pool: list[str] = []
    captions = text.get("captions")
    if isinstance(captions, dict):
        for key in _FLAIR_CAPTION_GROUP_KEYS:
            if key in captions:
                _extend_caption_pool(pool, captions.get(key))

    for key in _FLAIR_TOP_LEVEL_CAPTION_KEYS:
        if key in text:
            _extend_caption_pool(pool, text.get(key))

    if not pool and isinstance(captions, dict) and captions.get(relabel_success_key):
        _extend_caption_pool(pool, captions.get(relabel_success_key))
    return pool


def _wds_caption_source_from_sample(sample: dict[str, Any], relabel_success_key: str = "longSV") -> str | list[Any]:
    text = sample.get("text")
    if isinstance(text, dict):
        caption_pool = _flair_caption_pool_from_json(text, relabel_success_key)
        if caption_pool:
            return caption_pool
        caption = text.get("caption")
        if isinstance(caption, (str, list, tuple)):
            return caption
    if isinstance(text, (str, list, tuple)):
        return text
    raw_text = sample.get("raw_text")
    return str(raw_text or "").strip()


def _prepare_wds_sample(
    sample: dict[str, Any],
    preprocess_img: Callable[[Image.Image], torch.Tensor],
    tokenizer: Callable[[list[str]], torch.Tensor],
    relabel_success_key: str,
    num_sampled_captions: int,
    max_merged_num: int,
) -> dict[str, Any]:
    caption_source = _wds_caption_source_from_sample(sample, relabel_success_key)
    subcaptions = sample_subcaptions(caption_source, k=num_sampled_captions, max_merged_num=max_merged_num)
    text = tokenizer(subcaptions)
    sample_id = str(sample.get("__key__", ""))
    return {
        "image": preprocess_img(sample["image"]),
        "text": text,
        "caption": subcaptions,
        "image_id": sample_id,
        "sample_id": sample_id,
    }


def _tuple_to_batch_dict(batch: tuple[Any, ...]) -> dict[str, Any]:
    images, texts, captions, image_ids, sample_ids = batch
    return {
        "image": images,
        "text": texts,
        "caption": list(captions),
        "image_id": list(image_ids),
        "sample_id": list(sample_ids),
    }


def _is_webdataset_source(data: object) -> bool:
    return ".tar" in str(data) or "{" in str(data)


def get_wds_dataset(
    args: object,
    preprocess_img: Callable[[Image.Image], torch.Tensor],
    is_train: bool,
    epoch: int = 0,
    floor: bool = False,
    tokenizer: Callable[[list[str]], torch.Tensor] | None = None,
) -> DataInfo:
    """FLAIR-style WebDataset pipeline, adapted only at the final project batch mapping."""
    if tokenizer is None:
        raise ValueError("get_wds_dataset requires a tokenizer.")
    input_shards = getattr(args, "train_data", None) if is_train else getattr(args, "val_data", None)
    assert input_shards is not None
    resampled = bool(getattr(args, "dataset_resampled", False)) and is_train
    workers = int(getattr(args, "workers", 8))
    world_size = int(getattr(args, "world_size", get_world_size()))
    batch_size = int(getattr(args, "batch_size", 256))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    persistent_workers = bool(getattr(args, "persistent_workers", workers > 0))
    pin_memory = bool(getattr(args, "pin_memory", True))

    num_shards = None
    if is_train:
        if getattr(args, "train_num_samples", None) is not None:
            num_samples = int(getattr(args, "train_num_samples"))
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        num_samples = int(getattr(args, "val_num_samples", 0) or 0)

    shared_epoch = SharedEpoch(epoch=epoch)
    upsampling_factors = getattr(args, "train_data_upsampling_factors", None)
    if is_train and upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported with --dataset-resampled."

    if resampled:
        pipeline: list[Any] = [
            ResampledShards2(
                input_shards,
                weights=upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=int(getattr(args, "shard_shuffle_size", _SHARD_SHUFFLE_SIZE)),
                        initial=int(getattr(args, "shard_shuffle_initial", _SHARD_SHUFFLE_INITIAL)),
                        seed=int(getattr(args, "seed", 0)),
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                tarfile_to_samples_nothrow,
                wds.shuffle(
                    bufsize=int(getattr(args, "sample_shuffle_size", _SAMPLE_SHUFFLE_SIZE)),
                    initial=int(getattr(args, "sample_shuffle_initial", _SAMPLE_SHUFFLE_INITIAL)),
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )

    filter_relabel_success = bool(getattr(args, "filter_relabel_success", False))
    relabel_success_key = str(getattr(args, "relabel_success_key", "longSV"))
    strict_caption_match = bool(getattr(args, "strict_caption_match", True))
    num_sampled_captions = int(getattr(args, "num_sampled_captions", 4))
    max_merged_num = int(getattr(args, "max_merged_num", 3))
    if filter_relabel_success:
        pipeline.extend(
            [
                wds.select(filter_no_caption_or_no_image_json),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.select(lambda sample: _filter_relabel_success_json(sample, relabel_success_key, strict_caption_match)),
                wds.rename(image="jpg;png;jpeg;webp", text="json", raw_text="txt"),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            ]
        )

    pipeline.extend(
        [
            wds.map(
                lambda sample: _prepare_wds_sample(
                    sample,
                    preprocess_img,
                    tokenizer,
                    relabel_success_key,
                    num_sampled_captions,
                    max_merged_num,
                )
            ),
            wds.to_tuple("image", "text", "caption", "image_id", "sample_id"),
            wds.batched(batch_size, partial=not is_train),
            wds.map(_tuple_to_batch_dict),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= workers * world_size, "number of shards must be >= total workers"
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, workers)
        num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        num_batches = math.ceil(num_samples / batch_size) if num_samples else 0

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=persistent_workers if workers > 0 else False,
        pin_memory=pin_memory,
        **({"prefetch_factor": prefetch_factor} if workers > 0 else {}),
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def build_dataset(args: object, transform: Callable[[Image.Image], torch.Tensor], is_train: bool) -> Dataset[Any] | IterableDataset[Any]:
    """Build the configured dataset."""
    data = getattr(args, "train_data", None) if is_train else getattr(args, "val_data", None)
    dataset_type = str(getattr(args, "dataset_type", "auto"))
    if dataset_type == "synthetic":
        return SyntheticImageTextDataset(image_size=int(getattr(args, "image_size", 224)))
    if not data:
        split = "train" if is_train else "validation"
        raise ValueError(f"No {split} data configured.")
    if dataset_type in {"webdataset", "auto"} and (".tar" in str(data) or "{" in str(data)):
        raise ValueError("WebDataset sources are built by build_data_info/get_wds_dataset because tokenization happens inside the FLAIR-style pipeline.")
    return ManifestImageTextDataset(
        manifest_file=str(data),
        image_root=getattr(args, "image_root", None),
        image_key=str(getattr(args, "csv_img_key", "filepath")),
        caption_key=str(getattr(args, "csv_caption_key", "title")),
        transform=transform,
        file_format=dataset_type if dataset_type != "auto" else None,
    )


def build_dataloader(
    args: object,
    transform: Callable[[Image.Image], torch.Tensor],
    tokenizer: Callable[[list[str]], torch.Tensor],
    is_train: bool,
) -> DataLoader[Any]:
    """Build a dataloader with DDP-aware sampling."""
    return build_data_info(args, transform, tokenizer, is_train=is_train).dataloader


def build_data_info(
    args: object,
    transform: Callable[[Image.Image], torch.Tensor],
    tokenizer: Callable[[list[str]], torch.Tensor],
    is_train: bool,
) -> DataInfo:
    """Build data metadata around the dataloader."""
    data = getattr(args, "train_data", None) if is_train else getattr(args, "val_data", None)
    dataset_type = str(getattr(args, "dataset_type", "auto"))
    if data and dataset_type in {"webdataset", "auto"} and _is_webdataset_source(data):
        return get_wds_dataset(args, transform, is_train=is_train, tokenizer=tokenizer)

    dataset = build_dataset(args, transform, is_train=is_train)
    is_iterable = isinstance(dataset, IterableDataset)
    sampler: DistributedSampler[Any] | None = None
    shuffle = is_train and not is_iterable
    if is_train and get_world_size() > 1 and not is_iterable:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(getattr(args, "batch_size", 256)),
        "num_workers": int(getattr(args, "workers", 8)),
        "pin_memory": bool(getattr(args, "pin_memory", True)),
        "drop_last": is_train,
        "shuffle": shuffle,
        "sampler": sampler,
        "collate_fn": lambda samples: _collate(samples, tokenizer),
    }
    if int(getattr(args, "workers", 8)) > 0:
        loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))
    loader = DataLoader(dataset, **loader_kwargs)
    return DataInfo(dataloader=loader, sampler=sampler)


def get_data(
    args: object,
    preprocess_fns: tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]],
    epoch: int = 0,
    tokenizer: Callable[[list[str]], torch.Tensor] | None = None,
) -> dict[str, DataInfo]:
    """Return split metadata for the requested training and validation loaders."""
    if tokenizer is None:
        raise ValueError("get_data requires a tokenizer.")
    preprocess_train, preprocess_val = preprocess_fns
    data: dict[str, DataInfo] = {}
    if getattr(args, "train_data", None):
        data["train"] = build_data_info(args, preprocess_train, tokenizer, is_train=True)
        data["train"].set_epoch(epoch)
    if getattr(args, "val_data", None):
        data["val"] = build_data_info(args, preprocess_val, tokenizer, is_train=False)
    return data
