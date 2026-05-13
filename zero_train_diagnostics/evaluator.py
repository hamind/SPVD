from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from .benchmarks import load_benchmarks
from .metrics import (
    summarize_aro,
    summarize_2x2_by_benchmark,
    summarize_margins,
    summarize_pairwise_by_benchmark,
    summarize_ssr,
    summarize_sugarcrepe,
    summarize_winoground,
    summary_all_models,
)
from .models import FrozenVLMWrapper, load_model
from .plotting import generate_figures
from .schema import BenchmarkLoadResult, PairwiseSample, WinogroundSample
from .utils import (
    append_csv_rows,
    collect_environment_info,
    ensure_dir,
    now_utc_iso,
    read_yaml,
    set_seed,
    setup_logging,
    write_dataframe,
    write_json,
    write_yaml,
)


class ImageCache:
    def __init__(self, max_items: int = 512) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[tuple[str, tuple[int, int, int, int] | None], Image.Image] = OrderedDict()

    def get(self, path: Path, crop_box: tuple[int, int, int, int] | None = None) -> Image.Image:
        key = (str(path), crop_box)
        if key in self._cache:
            image = self._cache.pop(key)
            self._cache[key] = image
            return image.copy()
        with Image.open(path) as img:
            image = img.convert("RGB")
            if crop_box is not None:
                left, top, right, bottom = crop_box
                left = max(0, min(left, image.width - 1))
                top = max(0, min(top, image.height - 1))
                right = max(left + 1, min(right, image.width))
                bottom = max(top + 1, min(bottom, image.height))
                image = image.crop((left, top, right, bottom))
            image.load()
        self._cache[key] = image.copy()
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return image.copy()


@dataclass(frozen=True)
class DistributedEvalContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    initialized: bool = False
    use_shards: bool = False

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class PipelineConfig:
    batch_size: int
    pair_chunk_size: int
    num_workers_per_gpu: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    perf_log_interval: int = 20
    write_perf_jsonl: bool = True


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def init_distributed_eval(device_arg: str | None, distributed_eval: bool) -> DistributedEvalContext:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    initialized = False
    device = device_arg or "cuda"
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed evaluation requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        initialized = True
        device = f"cuda:{local_rank}"
    elif device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return DistributedEvalContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        initialized=initialized,
        use_shards=bool(distributed_eval or world_size > 1),
    )


def _dist_barrier(ctx: DistributedEvalContext) -> None:
    if ctx.initialized:
        dist.barrier()


def _dist_all_reduce_sum(values: list[int | float], ctx: DistributedEvalContext) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=ctx.device if str(ctx.device).startswith("cuda") else "cpu")
    if ctx.initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(v) for v in tensor.cpu().tolist()]


def _broadcast_run_id(run_id: str, ctx: DistributedEvalContext) -> str:
    if not ctx.initialized:
        return run_id
    payload = [run_id if ctx.is_rank0 else ""]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def _cleanup_distributed_eval(ctx: DistributedEvalContext) -> None:
    if ctx.initialized and dist.is_initialized():
        dist.destroy_process_group()


def _pipeline_config(config: dict[str, Any], eval_mode: dict[str, Any], batch_size: int, score_chunk: int) -> PipelineConfig:
    return PipelineConfig(
        batch_size=batch_size,
        pair_chunk_size=int(eval_mode.get("pair_chunk_size", config.get("pair_chunk_size", score_chunk))),
        num_workers_per_gpu=int(eval_mode.get("num_workers_per_gpu", config.get("num_workers_per_gpu", config.get("num_workers", 0)))),
        pin_memory=bool(eval_mode.get("pin_memory", config.get("pin_memory", True))),
        persistent_workers=bool(eval_mode.get("persistent_workers", config.get("persistent_workers", False))),
        prefetch_factor=eval_mode.get("prefetch_factor", config.get("prefetch_factor", 2)),
        perf_log_interval=int(eval_mode.get("perf_log_interval", config.get("perf_log_interval", 20))),
        write_perf_jsonl=bool(eval_mode.get("write_perf_jsonl", config.get("write_perf_jsonl", True))),
    )


class ImageTensorDataset(Dataset):
    def __init__(self, keys: list[tuple[str, tuple[int, int, int, int] | None]], preprocess: Any) -> None:
        self.keys = keys
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self.keys[index]
        path_text, crop_box = key
        try:
            with Image.open(path_text) as img:
                image = img.convert("RGB")
                if crop_box is not None:
                    left, top, right, bottom = crop_box
                    left = max(0, min(left, image.width - 1))
                    top = max(0, min(top, image.height - 1))
                    right = max(left + 1, min(right, image.width))
                    bottom = max(top + 1, min(bottom, image.height))
                    image = image.crop((left, top, right, bottom))
                tensor = self.preprocess(image)
            return {"index": index, "key": key, "image": tensor, "error": None}
        except Exception as exc:
            return {"index": index, "key": key, "image": None, "error": repr(exc)}


def _collate_image_tensors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in batch if item.get("image") is not None]
    errors = [item for item in batch if item.get("image") is None]
    return {
        "indices": [item["index"] for item in valid],
        "keys": [item["key"] for item in valid],
        "images": torch.stack([item["image"] for item in valid]) if valid else torch.empty(0),
        "errors": errors,
    }


class TextTokenDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any) -> None:
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text = self.texts[index]
        try:
            tokens = self.tokenizer([text])
            if torch.is_tensor(tokens) and tokens.ndim > 1:
                tokens = tokens[0]
            return {"index": index, "text": text, "tokens": tokens, "error": None}
        except Exception as exc:
            return {"index": index, "text": text, "tokens": None, "error": repr(exc)}


def _collate_text_tokens(batch: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in batch if item.get("tokens") is not None]
    errors = [item for item in batch if item.get("tokens") is None]
    return {
        "indices": [item["index"] for item in valid],
        "texts": [item["text"] for item in valid],
        "tokens": torch.stack([item["tokens"] for item in valid]) if valid else torch.empty(0, dtype=torch.long),
        "errors": errors,
    }


def _make_loader(dataset: Dataset, cfg: PipelineConfig, batch_size: int, collate_fn: Any) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": max(1, batch_size),
        "shuffle": False,
        "num_workers": max(0, cfg.num_workers_per_gpu),
        "pin_memory": bool(cfg.pin_memory),
        "collate_fn": collate_fn,
    }
    if cfg.num_workers_per_gpu > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        if cfg.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _shard_list(items: list[Any], ctx: DistributedEvalContext) -> list[Any]:
    if not ctx.use_shards:
        return items
    return items[ctx.rank :: ctx.world_size]


def _rank_jsonl_path(run_dir: Path, model_name: str, benchmark_name: str, rank: int) -> Path:
    return run_dir / "raw_results" / f"{model_name}_{benchmark_name}_rank{rank}.jsonl"


def _merged_csv_path(run_dir: Path, model_name: str, benchmark_name: str) -> Path:
    return run_dir / "raw_results" / f"{model_name}_{benchmark_name}.csv"


def _write_jsonl_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_rank_jsonl(run_dir: Path, model_name: str, benchmark_name: str, ctx: DistributedEvalContext) -> pd.DataFrame:
    _dist_barrier(ctx)
    if not ctx.is_rank0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for rank in range(ctx.world_size):
        rows.extend(_read_jsonl_rows(_rank_jsonl_path(run_dir, model_name, benchmark_name, rank)))
    df = pd.DataFrame(rows)
    write_dataframe(df, _merged_csv_path(run_dir, model_name, benchmark_name))
    return df


def _write_perf_event(run_dir: Path, model_name: str, benchmark_name: str, ctx: DistributedEvalContext, cfg: PipelineConfig, event: dict[str, Any]) -> None:
    if not cfg.write_perf_jsonl:
        return
    path = run_dir / "perf" / f"{model_name}_{benchmark_name}_perf_rank{ctx.rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rank": ctx.rank, "model": model_name, "benchmark": benchmark_name, **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def _score_pairs_with_retry(
    model: FrozenVLMWrapper,
    images: list[Image.Image],
    texts: list[str],
    batch_size: int,
    logger: logging.Logger,
) -> list[float]:
    scores: list[float] = []
    index = 0
    current = max(1, min(batch_size, len(images)))
    while index < len(images):
        end = min(index + current, len(images))
        try:
            values = model.score_pairs(images[index:end], texts[index:end])
            scores.extend(float(v) for v in values.detach().cpu().reshape(-1).tolist())
            index = end
        except RuntimeError as exc:
            if _is_oom(exc) and current > 1:
                logger.warning("OOM while scoring %s; reducing pair batch size from %d to %d", model.name, current, max(1, current // 2))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current = max(1, current // 2)
                continue
            raise
    return scores


def _safe_float(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _random_caption_map(samples: list[PairwiseSample], seed: int) -> dict[str, str | None]:
    rng = np.random.default_rng(seed)
    captions = [sample.positive_caption for sample in samples]
    mapping: dict[str, str | None] = {}
    if len(captions) < 2:
        for sample in samples:
            mapping[sample.sample_id] = None
        return mapping
    for idx, sample in enumerate(samples):
        other = int(rng.integers(0, len(captions) - 1))
        if other >= idx:
            other += 1
        mapping[sample.sample_id] = captions[other]
    return mapping


def _build_pairwise_specs(samples: list[PairwiseSample], random_map: dict[str, str | None]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sample in samples:
        random_caption = random_map.get(sample.sample_id)
        for neg_idx, negative_caption in enumerate(sample.negative_captions):
            specs.append(
                {
                    "benchmark": sample.benchmark,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "subcategory": sample.subcategory,
                    "image_path": str(sample.image_path),
                    "image_id": sample.image_id,
                    "crop_box": sample.crop_box,
                    "positive_caption": sample.positive_caption,
                    "negative_caption": negative_caption,
                    "negative_index": neg_idx,
                    "negative_type": sample.negative_types[neg_idx] if neg_idx < len(sample.negative_types) else "hard_negative",
                    "random_negative_caption": random_caption,
                    "source_file": str(sample.source_file) if sample.source_file else "",
                }
            )
    return specs


def _image_key(spec: dict[str, Any]) -> tuple[str, tuple[int, int, int, int] | None]:
    crop = spec.get("crop_box")
    crop_tuple = tuple(crop) if crop is not None else None
    return str(spec["image_path"]), crop_tuple  # type: ignore[return-value]


def _encode_with_retry(model: FrozenVLMWrapper, kind: str, items: list[Any], batch_size: int, logger: logging.Logger) -> torch.Tensor:
    current = max(1, min(batch_size, len(items)))
    outputs: list[torch.Tensor] = []
    index = 0
    while index < len(items):
        end = min(index + current, len(items))
        try:
            if kind == "image":
                encoded = model.encode_images(items[index:end])
            else:
                encoded = model.encode_texts(items[index:end])
            outputs.append(encoded.detach().cpu())
            index = end
        except RuntimeError as exc:
            if _is_oom(exc) and current > 1:
                logger.warning("OOM while encoding %s features for %s; reducing batch size from %d to %d", kind, model.name, current, max(1, current // 2))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current = max(1, current // 2)
                continue
            raise
    if not outputs:
        return torch.empty(0)
    return torch.cat(outputs, dim=0)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _tensor_tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(item) for item in value)
    return 0


def _encode_image_tokens_with_retry(
    model: FrozenVLMWrapper,
    images: list[Image.Image],
    batch_size: int,
    logger: logging.Logger,
) -> torch.Tensor:
    current = max(1, min(batch_size, len(images)))
    outputs: list[torch.Tensor] = []
    index = 0
    while index < len(images):
        end = min(index + current, len(images))
        try:
            encoded = model.encode_image_tokens(images[index:end])
            outputs.append(encoded.detach().cpu())
            index = end
        except RuntimeError as exc:
            if _is_oom(exc) and current > 1:
                logger.warning(
                    "OOM while encoding image tokens for %s; reducing batch size from %d to %d",
                    model.name,
                    current,
                    max(1, current // 2),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current = max(1, current // 2)
                continue
            raise
    if not outputs:
        return torch.empty(0)
    return torch.cat(outputs, dim=0)


def _encode_text_cues_with_retry(
    model: FrozenVLMWrapper,
    texts: list[str],
    batch_size: int,
    logger: logging.Logger,
) -> dict[str, torch.Tensor]:
    current = max(1, min(batch_size, len(texts)))
    outputs: dict[str, list[torch.Tensor]] = {}
    index = 0
    while index < len(texts):
        end = min(index + current, len(texts))
        try:
            encoded = model.encode_text_cues(texts[index:end])
            for key, value in encoded.items():
                if torch.is_tensor(value):
                    outputs.setdefault(key, []).append(value.detach().cpu())
            index = end
        except RuntimeError as exc:
            if _is_oom(exc) and current > 1:
                logger.warning(
                    "OOM while encoding text cues for %s; reducing batch size from %d to %d",
                    model.name,
                    current,
                    max(1, current // 2),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current = max(1, current // 2)
                continue
            raise
    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


def _score_conditioned_pairs_with_retry(
    model: FrozenVLMWrapper,
    image_tokens: torch.Tensor,
    soft_cues: torch.Tensor,
    text_features: torch.Tensor,
    batch_size: int,
    logger: logging.Logger,
) -> torch.Tensor:
    total = int(image_tokens.shape[0])
    current = max(1, min(batch_size, total))
    outputs: list[torch.Tensor] = []
    index = 0
    while index < total:
        end = min(index + current, total)
        try:
            scores = model.score_cached_conditioned_pairs(
                image_tokens=image_tokens[index:end],
                soft_cues=soft_cues[index:end],
                text_features=text_features[index:end],
            )
            outputs.append(scores.detach().cpu())
            index = end
        except RuntimeError as exc:
            if _is_oom(exc) and current > 1:
                logger.warning(
                    "OOM while scoring text-conditioned pairs for %s; reducing score batch from %d to %d",
                    model.name,
                    current,
                    max(1, current // 2),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current = max(1, current // 2)
                continue
            raise
    if not outputs:
        return torch.empty(0)
    return torch.cat(outputs, dim=0)


def _score_conditioned_pairs_pipeline(
    model: FrozenVLMWrapper,
    pair_image_tokens: list[torch.Tensor],
    pair_soft_cues: list[torch.Tensor],
    pair_text_features: list[torch.Tensor],
    cfg: PipelineConfig,
    logger: logging.Logger,
    run_dir: Path,
    benchmark_name: str,
    ctx: DistributedEvalContext,
    step_base: int,
) -> tuple[list[float], dict[str, float]]:
    scores_out: list[float] = []
    total_h2d = 0.0
    total_gpu = 0.0
    total_pairs = len(pair_image_tokens)
    chunk_size = max(1, cfg.pair_chunk_size)
    device = torch.device(model.device)
    for offset in range(0, total_pairs, chunk_size):
        h2d_start = time.perf_counter()
        image_tokens = torch.stack(pair_image_tokens[offset : offset + chunk_size])
        soft_cues = torch.stack(pair_soft_cues[offset : offset + chunk_size])
        text_features = torch.stack(pair_text_features[offset : offset + chunk_size])
        if cfg.pin_memory and device.type == "cuda":
            image_tokens = image_tokens.pin_memory()
            soft_cues = soft_cues.pin_memory()
            text_features = text_features.pin_memory()
        image_tokens = image_tokens.to(device, non_blocking=True)
        soft_cues = soft_cues.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_time = time.perf_counter() - h2d_start

        gpu_start = time.perf_counter()
        scores = model.score_cached_conditioned_pairs(image_tokens, soft_cues, text_features)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_time = time.perf_counter() - gpu_start

        scores_out.extend(float(v) for v in scores.detach().cpu().float().reshape(-1).tolist())
        total_h2d += h2d_time
        total_gpu += gpu_time
        step = step_base + offset // chunk_size
        if cfg.perf_log_interval > 0 and (step == 0 or step % cfg.perf_log_interval == 0):
            pairs = min(chunk_size, total_pairs - offset)
            throughput = pairs / max(gpu_time, 1e-9)
            memory = float(torch.cuda.memory_allocated(device) if device.type == "cuda" else 0)
            event = {
                "step": step,
                "pairs": pairs,
                "data_time": 0.0,
                "h2d_time": h2d_time,
                "gpu_time": gpu_time,
                "throughput_pairs_per_s": throughput,
                "gpu_memory_allocated": memory,
            }
            if ctx.is_rank0:
                logger.info(
                    "%s/%s rank=%d step=%d h2d=%.4fs gpu=%.4fs throughput=%.2f pairs/s mem=%.2f MiB",
                    model.name,
                    benchmark_name,
                    ctx.rank,
                    step,
                    h2d_time,
                    gpu_time,
                    throughput,
                    memory / (1024**2),
                )
            _write_perf_event(run_dir, model.name, benchmark_name, ctx, cfg, event)
    return scores_out, {"h2d_time": total_h2d, "gpu_time": total_gpu}


def _cache_image_tokens(
    model: FrozenVLMWrapper,
    benchmark_name: str,
    image_keys: list[tuple[str, tuple[int, int, int, int] | None]],
    image_cache: ImageCache,
    batch_size: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
    pipeline_cfg: PipelineConfig | None = None,
) -> tuple[dict[tuple[str, tuple[int, int, int, int] | None], torch.Tensor], set[tuple[str, tuple[int, int, int, int] | None]]]:
    image_tokens: dict[tuple[str, tuple[int, int, int, int] | None], torch.Tensor] = {}
    bad_image_keys: set[tuple[str, tuple[int, int, int, int] | None]] = set()
    if pipeline_cfg is not None and getattr(model, "preprocess", None) is not None:
        loader = _make_loader(
            ImageTensorDataset(image_keys, model.preprocess),
            pipeline_cfg,
            batch_size,
            _collate_image_tensors,
        )
        seen = 0
        for batch in loader:
            for error in batch["errors"]:
                key = error["key"]
                bad_image_keys.add(key)
                skipped_samples.append(
                    {
                        "benchmark": benchmark_name,
                        "source_file": "",
                        "sample_id": "",
                        "reason": f"image load failed during SPVD token cache: {key[0]} {key[1]} {error['error']}",
                    }
                )
            if batch["images"].numel() > 0:
                encoded = model.encode_image_tokens_from_tensor(batch["images"])
                for key, tokens in zip(batch["keys"], encoded):
                    image_tokens[key] = tokens.detach().cpu()
            seen += len(batch["indices"]) + len(batch["errors"])
            if seen == len(image_keys) or seen <= batch_size or seen % max(batch_size * 20, 1) == 0:
                logger.info("%s/%s image tokens: %d/%d encoded", model.name, benchmark_name, seen, len(image_keys))
        return image_tokens, bad_image_keys

    image_batch = max(1, batch_size)
    for start in range(0, len(image_keys), image_batch):
        keys = image_keys[start : start + image_batch]
        valid_keys: list[tuple[str, tuple[int, int, int, int] | None]] = []
        images: list[Image.Image] = []
        for key in keys:
            path_text, crop_box = key
            try:
                images.append(image_cache.get(Path(path_text), crop_box))
                valid_keys.append(key)
            except Exception as exc:
                bad_image_keys.add(key)
                skipped_samples.append(
                    {
                        "benchmark": benchmark_name,
                        "source_file": "",
                        "sample_id": "",
                        "reason": f"image load failed during SPVD token cache: {path_text} {crop_box} {exc!r}",
                    }
                )
        if images:
            encoded = _encode_image_tokens_with_retry(model, images, image_batch, logger)
            for key, tokens in zip(valid_keys, encoded):
                image_tokens[key] = tokens.detach().cpu()
        if start == 0 or (start + image_batch) % max(image_batch * 20, 1) == 0 or start + image_batch >= len(image_keys):
            logger.info(
                "%s/%s image tokens: %d/%d encoded",
                model.name,
                benchmark_name,
                min(start + image_batch, len(image_keys)),
                len(image_keys),
            )
    return image_tokens, bad_image_keys


def _cache_text_cues(
    model: FrozenVLMWrapper,
    benchmark_name: str,
    unique_texts: list[str],
    batch_size: int,
    logger: logging.Logger,
    pipeline_cfg: PipelineConfig | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    text_cache: dict[str, dict[str, torch.Tensor]] = {}
    tokenizer = getattr(model, "tokenizer", None)
    if pipeline_cfg is not None and tokenizer is not None:
        loader = _make_loader(
            TextTokenDataset(unique_texts, tokenizer),
            pipeline_cfg,
            max(1, batch_size * 4),
            _collate_text_tokens,
        )
        seen = 0
        for batch in loader:
            if batch["tokens"].numel() > 0:
                encoded = model.encode_text_cues_from_tokens(batch["tokens"])
                for idx, text in enumerate(batch["texts"]):
                    text_cache[text] = {key: value[idx].detach().cpu() for key, value in encoded.items()}
            for error in batch["errors"]:
                logger.warning("Text cue tokenization failed for %s/%s: %s", model.name, benchmark_name, error["error"])
            seen += len(batch["texts"]) + len(batch["errors"])
            if seen == len(unique_texts) or seen <= batch_size or seen % max(batch_size * 20, 1) == 0:
                logger.info("%s/%s text cues: %d/%d encoded", model.name, benchmark_name, seen, len(unique_texts))
        return text_cache

    text_batch = max(1, batch_size * 4)
    for start in range(0, len(unique_texts), text_batch):
        texts = unique_texts[start : start + text_batch]
        encoded = _encode_text_cues_with_retry(model, texts, text_batch, logger)
        for idx, text in enumerate(texts):
            text_cache[text] = {key: value[idx].detach().cpu() for key, value in encoded.items()}
        if start == 0 or (start + text_batch) % max(text_batch * 20, 1) == 0 or start + text_batch >= len(unique_texts):
            logger.info(
                "%s/%s text cues: %d/%d encoded",
                model.name,
                benchmark_name,
                min(start + text_batch, len(unique_texts)),
                len(unique_texts),
            )
    return text_cache


def _evaluate_pairwise_with_text_conditioned_cache(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    batch_size: int,
    compute_random_negative: bool,
    seed: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
    score_chunk: int,
    dry_run: bool,
    dist_ctx: DistributedEvalContext | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    dist_ctx = dist_ctx or DistributedEvalContext()
    all_samples = list(benchmark.samples)
    samples = _shard_list(all_samples, dist_ctx)
    random_map = _random_caption_map(all_samples, seed) if compute_random_negative else {}
    specs = _build_pairwise_specs(samples, random_map)
    raw_path = _rank_jsonl_path(run_dir, model.name, benchmark.name, dist_ctx.rank) if dist_ctx.use_shards else _merged_csv_path(run_dir, model.name, benchmark.name)
    if raw_path.exists():
        raw_path.unlink()
    if dist_ctx.use_shards and dist_ctx.is_rank0:
        merged_path = _merged_csv_path(run_dir, model.name, benchmark.name)
        if merged_path.exists():
            merged_path.unlink()

    image_keys = list(dict.fromkeys(_image_key(spec) for spec in specs))
    text_values: list[str] = []
    pair_count = 0
    for spec in specs:
        text_values.append(spec["positive_caption"])
        text_values.append(spec["negative_caption"])
        pair_count += 2
        random_caption = spec.get("random_negative_caption")
        if random_caption:
            text_values.append(random_caption)
            pair_count += 1
    unique_texts = list(dict.fromkeys(text_values))
    logger.info(
        "SPVD exact_cached %splan for %s/%s rank=%d/%d: total_samples=%d shard_samples=%d unique_images=%d unique_texts=%d pair_count=%d workers=%d pair_chunk_size=%d cache_dtype=%s",
        "dry-run " if dry_run else "",
        model.name,
        benchmark.name,
        dist_ctx.rank,
        dist_ctx.world_size,
        len(all_samples),
        len(samples),
        len(image_keys),
        len(unique_texts),
        pair_count,
        pipeline_cfg.num_workers_per_gpu if pipeline_cfg else 0,
        pipeline_cfg.pair_chunk_size if pipeline_cfg else score_chunk,
        model.dtype,
    )

    image_cache = ImageCache(max_items=64)
    image_tokens, bad_image_keys = _cache_image_tokens(
        model,
        benchmark.name,
        image_keys,
        image_cache,
        batch_size,
        logger,
        skipped_samples,
        pipeline_cfg,
    )
    text_cache = _cache_text_cues(model, benchmark.name, unique_texts, batch_size, logger, pipeline_cfg)
    image_cache_bytes = _tensor_tree_nbytes(image_tokens)
    text_cache_bytes = _tensor_tree_nbytes(text_cache)
    logger.info(
        "SPVD exact_cached cache memory for %s/%s: image_tokens=%s text_cues=%s total=%s",
        model.name,
        benchmark.name,
        _format_bytes(image_cache_bytes),
        _format_bytes(text_cache_bytes),
        _format_bytes(image_cache_bytes + text_cache_bytes),
    )

    all_rows: list[dict[str, Any]] = []
    score_chunk = max(1, int(score_chunk))
    for start in range(0, len(specs), score_chunk):
        data_start = time.perf_counter()
        chunk_specs = specs[start : start + score_chunk]
        valid_specs: list[dict[str, Any]] = []
        has_random: list[bool] = []
        pair_image_tokens: list[torch.Tensor] = []
        pair_soft_cues: list[torch.Tensor] = []
        pair_text_features: list[torch.Tensor] = []
        for spec in chunk_specs:
            key = _image_key(spec)
            if key in bad_image_keys or key not in image_tokens:
                continue
            pos_cache = text_cache.get(spec["positive_caption"])
            neg_cache = text_cache.get(spec["negative_caption"])
            if pos_cache is None or neg_cache is None:
                continue
            valid_specs.append(spec)
            base_tokens = image_tokens[key]
            for text_entry in (pos_cache, neg_cache):
                pair_image_tokens.append(base_tokens)
                pair_soft_cues.append(text_entry["soft_cues"])
                pair_text_features.append(text_entry["text_features"])
            random_caption = spec.get("random_negative_caption")
            random_cache = text_cache.get(random_caption) if random_caption else None
            if random_cache is not None:
                has_random.append(True)
                pair_image_tokens.append(base_tokens)
                pair_soft_cues.append(random_cache["soft_cues"])
                pair_text_features.append(random_cache["text_features"])
            else:
                has_random.append(False)
        if not valid_specs:
            continue
        data_done = time.perf_counter()
        if pipeline_cfg is not None:
            score_values, timing = _score_conditioned_pairs_pipeline(
                model,
                pair_image_tokens,
                pair_soft_cues,
                pair_text_features,
                pipeline_cfg,
                logger,
                run_dir,
                benchmark.name,
                dist_ctx,
                start // max(1, score_chunk),
            )
            data_time = data_done - data_start
        else:
            scores = _score_conditioned_pairs_with_retry(
                model,
                torch.stack(pair_image_tokens),
                torch.stack(pair_soft_cues),
                torch.stack(pair_text_features),
                score_chunk,
                logger,
            )
            score_values = scores.detach().cpu().float().reshape(-1).tolist()
            timing = {"h2d_time": 0.0, "gpu_time": 0.0}
            data_time = data_done - data_start
        rows: list[dict[str, Any]] = []
        cursor = 0
        for idx, spec in enumerate(valid_specs):
            pos = float(score_values[cursor])
            neg = float(score_values[cursor + 1])
            cursor += 2
            random_score = None
            if has_random[idx]:
                random_score = float(score_values[cursor])
                cursor += 1
            hard_margin = pos - neg
            random_margin = pos - random_score if random_score is not None else float("nan")
            ssr = hard_margin / (random_margin + 1e-6) if random_score is not None else float("nan")
            rows.append(
                {
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "benchmark": spec["benchmark"],
                    "sample_id": spec["sample_id"],
                    "category": spec["category"],
                    "subcategory": spec.get("subcategory"),
                    "image_path": spec["image_path"],
                    "image_id": spec.get("image_id"),
                    "positive_caption": spec["positive_caption"],
                    "negative_caption": spec["negative_caption"],
                    "negative_index": spec.get("negative_index"),
                    "negative_type": spec.get("negative_type"),
                    "random_negative_caption": spec.get("random_negative_caption"),
                    "score_pos": pos,
                    "score_neg": neg,
                    "score_random_neg": _safe_float(random_score),
                    "hard_margin": hard_margin,
                    "random_margin": random_margin,
                    "ssr": ssr,
                    "correct": int(pos > neg),
                    "source_file": spec.get("source_file"),
                }
            )
        if dist_ctx.use_shards:
            _write_jsonl_rows(rows, raw_path)
        else:
            append_csv_rows(rows, raw_path)
        all_rows.extend(rows)
        if start == 0 or start + score_chunk >= len(specs) or (start + score_chunk) % (score_chunk * 10) == 0:
            pairs_scored = len(score_values)
            throughput = pairs_scored / max(timing["gpu_time"], 1e-9) if pipeline_cfg is not None else 0.0
            logger.info(
                "%s/%s rank=%d raw rows scored: %d/%d data=%.4fs h2d=%.4fs gpu=%.4fs throughput=%.2f pairs/s",
                model.name,
                benchmark.name,
                dist_ctx.rank,
                min(start + score_chunk, len(specs)),
                len(specs),
                data_time,
                timing["h2d_time"],
                timing["gpu_time"],
                throughput,
            )
    local_correct = sum(int(row["correct"]) for row in all_rows)
    local_total = len(all_rows)
    global_correct, global_total = _dist_all_reduce_sum([local_correct, local_total], dist_ctx)
    if dist_ctx.is_rank0:
        logger.info(
            "%s/%s metric aggregate: correct=%d total=%d acc=%.6f",
            model.name,
            benchmark.name,
            int(global_correct),
            int(global_total),
            global_correct / max(global_total, 1.0),
        )
    logger.info("Wrote %d raw rows to %s", len(all_rows), raw_path)
    if dist_ctx.use_shards:
        return _merge_rank_jsonl(run_dir, model.name, benchmark.name, dist_ctx)
    return pd.DataFrame(all_rows)


def _evaluate_pairwise_with_feature_cache(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    batch_size: int,
    compute_random_negative: bool,
    seed: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
) -> pd.DataFrame:
    samples = list(benchmark.samples)
    random_map = _random_caption_map(samples, seed) if compute_random_negative else {}
    specs = _build_pairwise_specs(samples, random_map)
    raw_path = run_dir / "raw_results" / f"{model.name}_{benchmark.name}.csv"
    if raw_path.exists():
        raw_path.unlink()
    logger.info(
        "Using dual-encoder feature-cache path for %s on %s: %d rows",
        model.name,
        benchmark.name,
        len(specs),
    )

    image_cache = ImageCache(max_items=64)
    image_keys = list(dict.fromkeys(_image_key(spec) for spec in specs))
    image_features: dict[tuple[str, tuple[int, int, int, int] | None], torch.Tensor] = {}
    bad_image_keys: set[tuple[str, tuple[int, int, int, int] | None]] = set()
    image_batch = max(1, batch_size)
    for start in range(0, len(image_keys), image_batch):
        keys = image_keys[start : start + image_batch]
        valid_keys: list[tuple[str, tuple[int, int, int, int] | None]] = []
        images: list[Image.Image] = []
        for key in keys:
            path_text, crop_box = key
            try:
                images.append(image_cache.get(Path(path_text), crop_box))
                valid_keys.append(key)
            except Exception as exc:
                bad_image_keys.add(key)
                skipped_samples.append(
                    {
                        "benchmark": benchmark.name,
                        "source_file": "",
                        "sample_id": "",
                        "reason": f"image load failed during feature cache: {path_text} {crop_box} {exc!r}",
                    }
                )
        if images:
            feats = _encode_with_retry(model, "image", images, image_batch, logger)
            for key, feat in zip(valid_keys, feats):
                image_features[key] = feat.detach().cpu()
        if start == 0 or (start + image_batch) % max(image_batch * 20, 1) == 0 or start + image_batch >= len(image_keys):
            logger.info(
                "%s/%s image features: %d/%d encoded",
                model.name,
                benchmark.name,
                min(start + image_batch, len(image_keys)),
                len(image_keys),
            )

    text_values: list[str] = []
    for spec in specs:
        text_values.append(spec["positive_caption"])
        text_values.append(spec["negative_caption"])
        random_caption = spec.get("random_negative_caption")
        if random_caption:
            text_values.append(random_caption)
    unique_texts = list(dict.fromkeys(text_values))
    text_features: dict[str, torch.Tensor] = {}
    text_batch = max(1, batch_size * 4)
    for start in range(0, len(unique_texts), text_batch):
        texts = unique_texts[start : start + text_batch]
        feats = _encode_with_retry(model, "text", texts, text_batch, logger)
        for text, feat in zip(texts, feats):
            text_features[text] = feat.detach().cpu()
        if start == 0 or (start + text_batch) % max(text_batch * 20, 1) == 0 or start + text_batch >= len(unique_texts):
            logger.info(
                "%s/%s text features: %d/%d encoded",
                model.name,
                benchmark.name,
                min(start + text_batch, len(unique_texts)),
                len(unique_texts),
            )

    all_rows: list[dict[str, Any]] = []
    score_chunk = 8192
    for start in range(0, len(specs), score_chunk):
        chunk_specs = specs[start : start + score_chunk]
        valid_specs: list[dict[str, Any]] = []
        img_pos_feats: list[torch.Tensor] = []
        pos_text_feats: list[torch.Tensor] = []
        neg_text_feats: list[torch.Tensor] = []
        rand_text_feats: list[torch.Tensor] = []
        rand_mask: list[bool] = []
        for spec in chunk_specs:
            key = _image_key(spec)
            if key in bad_image_keys or key not in image_features:
                continue
            random_caption = spec.get("random_negative_caption")
            if spec["positive_caption"] not in text_features or spec["negative_caption"] not in text_features:
                continue
            valid_specs.append(spec)
            image_feat = image_features[key]
            img_pos_feats.append(image_feat)
            pos_text_feats.append(text_features[spec["positive_caption"]])
            neg_text_feats.append(text_features[spec["negative_caption"]])
            if random_caption and random_caption in text_features:
                rand_mask.append(True)
                rand_text_feats.append(text_features[random_caption])
            else:
                rand_mask.append(False)
                rand_text_feats.append(torch.zeros_like(image_feat))
        if not valid_specs:
            continue
        image_tensor = torch.stack(img_pos_feats)
        pos_tensor = torch.stack(pos_text_feats)
        neg_tensor = torch.stack(neg_text_feats)
        rand_tensor = torch.stack(rand_text_feats)
        pos_scores = model.score_encoded_pairs(image_tensor, pos_tensor).detach().cpu().float()
        neg_scores = model.score_encoded_pairs(image_tensor, neg_tensor).detach().cpu().float()
        rand_scores = model.score_encoded_pairs(image_tensor, rand_tensor).detach().cpu().float()
        rows: list[dict[str, Any]] = []
        for idx, spec in enumerate(valid_specs):
            pos = float(pos_scores[idx].item())
            neg = float(neg_scores[idx].item())
            random_score = float(rand_scores[idx].item()) if rand_mask[idx] else None
            hard_margin = pos - neg
            random_margin = pos - random_score if random_score is not None else float("nan")
            ssr = hard_margin / (random_margin + 1e-6) if random_score is not None else float("nan")
            rows.append(
                {
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "benchmark": spec["benchmark"],
                    "sample_id": spec["sample_id"],
                    "category": spec["category"],
                    "subcategory": spec.get("subcategory"),
                    "image_path": spec["image_path"],
                    "image_id": spec.get("image_id"),
                    "positive_caption": spec["positive_caption"],
                    "negative_caption": spec["negative_caption"],
                    "negative_index": spec.get("negative_index"),
                    "negative_type": spec.get("negative_type"),
                    "random_negative_caption": spec.get("random_negative_caption"),
                    "score_pos": pos,
                    "score_neg": neg,
                    "score_random_neg": _safe_float(random_score),
                    "hard_margin": hard_margin,
                    "random_margin": random_margin,
                    "ssr": ssr,
                    "correct": int(pos > neg),
                    "source_file": spec.get("source_file"),
                }
            )
        append_csv_rows(rows, raw_path)
        all_rows.extend(rows)
        if start == 0 or start + score_chunk >= len(specs) or (start + score_chunk) % (score_chunk * 10) == 0:
            logger.info("%s/%s raw rows scored: %d/%d", model.name, benchmark.name, min(start + score_chunk, len(specs)), len(specs))
    logger.info("Wrote %d raw rows to %s", len(all_rows), raw_path)
    return pd.DataFrame(all_rows)


def _flush_pairwise_rows(
    pending: list[dict[str, Any]],
    model: FrozenVLMWrapper,
    image_cache: ImageCache,
    batch_size: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_specs: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for spec in pending:
        try:
            images.append(image_cache.get(Path(spec["image_path"]), spec.get("crop_box")))
            valid_specs.append(spec)
        except Exception as exc:
            skipped_samples.append(
                {
                    "benchmark": spec.get("benchmark"),
                    "source_file": spec.get("source_file", ""),
                    "sample_id": spec.get("sample_id"),
                    "reason": f"image load failed during evaluation: {exc!r}",
                }
            )
    if not valid_specs:
        return []
    has_random = any(spec.get("random_negative_caption") for spec in valid_specs)
    pair_images: list[Image.Image] = []
    pair_texts: list[str] = []
    for image, spec in zip(images, valid_specs):
        pair_images.append(image)
        pair_texts.append(spec["positive_caption"])
        pair_images.append(image)
        pair_texts.append(spec["negative_caption"])
        if has_random:
            pair_images.append(image)
            pair_texts.append(spec.get("random_negative_caption") or spec["negative_caption"])
    flat_scores = _score_pairs_with_retry(model, pair_images, pair_texts, batch_size, logger)
    stride = 3 if has_random else 2
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(valid_specs):
        pos = flat_scores[idx * stride]
        neg = flat_scores[idx * stride + 1]
        random_score = flat_scores[idx * stride + 2] if has_random and spec.get("random_negative_caption") else None
        hard_margin = pos - neg
        random_margin = pos - random_score if random_score is not None else float("nan")
        ssr = hard_margin / (random_margin + 1e-6) if random_score is not None else float("nan")
        rows.append(
            {
                "model_name": model.name,
                "model_type": model.model_type,
                "benchmark": spec["benchmark"],
                "sample_id": spec["sample_id"],
                "category": spec["category"],
                "subcategory": spec.get("subcategory"),
                "image_path": spec["image_path"],
                "image_id": spec.get("image_id"),
                "positive_caption": spec["positive_caption"],
                "negative_caption": spec["negative_caption"],
                "negative_index": spec.get("negative_index"),
                "negative_type": spec.get("negative_type"),
                "random_negative_caption": spec.get("random_negative_caption"),
                "score_pos": pos,
                "score_neg": neg,
                "score_random_neg": _safe_float(random_score),
                "hard_margin": hard_margin,
                "random_margin": random_margin,
                "ssr": ssr,
                "correct": int(pos > neg),
                "source_file": spec.get("source_file"),
            }
        )
    return rows


def evaluate_pairwise_benchmark(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    batch_size: int,
    compute_random_negative: bool,
    seed: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
    eval_mode: dict[str, Any] | None = None,
    score_chunk: int = 8192,
    dry_run: bool = False,
    dist_ctx: DistributedEvalContext | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    dist_ctx = dist_ctx or DistributedEvalContext()
    samples = list(benchmark.samples)
    eval_mode = eval_mode or {}
    spvd_pairwise_mode = str(eval_mode.get("spvd_pairwise_mode", "exact_cached"))
    if model.supports_text_conditioned_pair_cache and spvd_pairwise_mode == "exact_cached":
        return _evaluate_pairwise_with_text_conditioned_cache(
            model,
            benchmark,
            run_dir,
            batch_size,
            compute_random_negative,
            seed,
            logger,
            skipped_samples,
            score_chunk,
            dry_run,
            dist_ctx,
            pipeline_cfg,
        )
    if dist_ctx.use_shards:
        all_samples = samples
        shard_samples = _shard_list(all_samples, dist_ctx)
        random_map = _random_caption_map(all_samples, seed) if compute_random_negative else {}
        specs = _build_pairwise_specs(shard_samples, random_map)
        raw_path = _rank_jsonl_path(run_dir, model.name, benchmark.name, dist_ctx.rank)
        if raw_path.exists():
            raw_path.unlink()
        if dist_ctx.is_rank0:
            merged_path = _merged_csv_path(run_dir, model.name, benchmark.name)
            if merged_path.exists():
                merged_path.unlink()
            logger.info(
                "%s/%s distributed naive path: total_samples=%d per_rank=%s num_workers=%d pair_chunk_size=%d",
                model.name,
                benchmark.name,
                len(all_samples),
                [len(all_samples[r :: dist_ctx.world_size]) for r in range(dist_ctx.world_size)],
                pipeline_cfg.num_workers_per_gpu if pipeline_cfg else 0,
                pipeline_cfg.pair_chunk_size if pipeline_cfg else score_chunk,
            )
        image_cache = ImageCache()
        all_rows: list[dict[str, Any]] = []
        flush_size = max(1, batch_size)
        for start in range(0, len(specs), flush_size):
            rows = _flush_pairwise_rows(specs[start : start + flush_size], model, image_cache, batch_size, logger, skipped_samples)
            _write_jsonl_rows(rows, raw_path)
            all_rows.extend(rows)
        local_correct = sum(int(row["correct"]) for row in all_rows)
        local_total = len(all_rows)
        global_correct, global_total = _dist_all_reduce_sum([local_correct, local_total], dist_ctx)
        if dist_ctx.is_rank0:
            logger.info(
                "%s/%s metric aggregate: correct=%d total=%d acc=%.6f",
                model.name,
                benchmark.name,
                int(global_correct),
                int(global_total),
                global_correct / max(global_total, 1.0),
            )
        logger.info("Wrote %d raw rows to %s", len(all_rows), raw_path)
        return _merge_rank_jsonl(run_dir, model.name, benchmark.name, dist_ctx)
    if model.supports_feature_cache:
        return _evaluate_pairwise_with_feature_cache(
            model,
            benchmark,
            run_dir,
            batch_size,
            compute_random_negative,
            seed,
            logger,
            skipped_samples,
        )
    random_map = _random_caption_map(samples, seed) if compute_random_negative else {}
    image_cache = ImageCache()
    raw_path = run_dir / "raw_results" / f"{model.name}_{benchmark.name}.csv"
    if raw_path.exists():
        raw_path.unlink()
    pending: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    flush_size = max(1, batch_size)
    for spec in _build_pairwise_specs(samples, random_map):
        pending.append(spec)
        if len(pending) >= flush_size:
            rows = _flush_pairwise_rows(pending, model, image_cache, batch_size, logger, skipped_samples)
            append_csv_rows(rows, raw_path)
            all_rows.extend(rows)
            pending = []
    if pending:
        rows = _flush_pairwise_rows(pending, model, image_cache, batch_size, logger, skipped_samples)
        append_csv_rows(rows, raw_path)
        all_rows.extend(rows)
    logger.info("Wrote %d raw rows to %s", len(all_rows), raw_path)
    return pd.DataFrame(all_rows)


def _evaluate_winoground_with_text_conditioned_cache(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    batch_size: int,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
    score_chunk: int,
    dry_run: bool,
    dist_ctx: DistributedEvalContext | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    dist_ctx = dist_ctx or DistributedEvalContext()
    benchmark_name = benchmark.name
    all_samples = [sample for sample in benchmark.samples if isinstance(sample, WinogroundSample)]
    samples = _shard_list(all_samples, dist_ctx)
    raw_path = _rank_jsonl_path(run_dir, model.name, benchmark_name, dist_ctx.rank) if dist_ctx.use_shards else _merged_csv_path(run_dir, model.name, benchmark_name)
    if raw_path.exists():
        raw_path.unlink()
    if dist_ctx.use_shards and dist_ctx.is_rank0:
        merged_path = _merged_csv_path(run_dir, model.name, benchmark_name)
        if merged_path.exists():
            merged_path.unlink()

    image_keys = list(
        dict.fromkeys(
            [(str(sample.image_0_path), None) for sample in samples]
            + [(str(sample.image_1_path), None) for sample in samples]
        )
    )
    unique_texts = list(
        dict.fromkeys(
            [sample.caption_0 for sample in samples]
            + [sample.caption_1 for sample in samples]
        )
    )
    logger.info(
        "SPVD exact_cached %splan for %s/%s rank=%d/%d: total_samples=%d shard_samples=%d unique_images=%d unique_texts=%d pair_count=%d workers=%d pair_chunk_size=%d cache_dtype=%s",
        "dry-run " if dry_run else "",
        model.name,
        benchmark_name,
        dist_ctx.rank,
        dist_ctx.world_size,
        len(all_samples),
        len(samples),
        len(image_keys),
        len(unique_texts),
        len(samples) * 4,
        pipeline_cfg.num_workers_per_gpu if pipeline_cfg else 0,
        pipeline_cfg.pair_chunk_size if pipeline_cfg else score_chunk,
        model.dtype,
    )

    image_cache = ImageCache(max_items=64)
    image_tokens, bad_image_keys = _cache_image_tokens(
        model,
        benchmark_name,
        image_keys,
        image_cache,
        batch_size,
        logger,
        skipped_samples,
        pipeline_cfg,
    )
    text_cache = _cache_text_cues(model, benchmark_name, unique_texts, batch_size, logger, pipeline_cfg)
    image_cache_bytes = _tensor_tree_nbytes(image_tokens)
    text_cache_bytes = _tensor_tree_nbytes(text_cache)
    logger.info(
        "SPVD exact_cached cache memory for %s/%s: image_tokens=%s text_cues=%s total=%s",
        model.name,
        benchmark_name,
        _format_bytes(image_cache_bytes),
        _format_bytes(text_cache_bytes),
        _format_bytes(image_cache_bytes + text_cache_bytes),
    )

    rows: list[dict[str, Any]] = []
    score_chunk = max(1, int(score_chunk))
    sample_chunk = max(1, score_chunk // 4)
    for start in range(0, len(samples), sample_chunk):
        data_start = time.perf_counter()
        chunk_samples = samples[start : start + sample_chunk]
        valid_samples: list[WinogroundSample] = []
        pair_image_tokens: list[torch.Tensor] = []
        pair_soft_cues: list[torch.Tensor] = []
        pair_text_features: list[torch.Tensor] = []
        for sample in chunk_samples:
            key_0 = (str(sample.image_0_path), None)
            key_1 = (str(sample.image_1_path), None)
            text_0 = text_cache.get(sample.caption_0)
            text_1 = text_cache.get(sample.caption_1)
            if key_0 in bad_image_keys or key_1 in bad_image_keys or key_0 not in image_tokens or key_1 not in image_tokens:
                skipped_samples.append(
                    {
                        "benchmark": benchmark_name,
                        "source_file": str(sample.source_file) if sample.source_file else "",
                        "sample_id": sample.sample_id,
                        "reason": "image token cache unavailable",
                    }
                )
                continue
            if text_0 is None or text_1 is None:
                skipped_samples.append(
                    {
                        "benchmark": benchmark_name,
                        "source_file": str(sample.source_file) if sample.source_file else "",
                        "sample_id": sample.sample_id,
                        "reason": "text cue cache unavailable",
                    }
                )
                continue
            valid_samples.append(sample)
            for tokens, text_entry in (
                (image_tokens[key_0], text_0),
                (image_tokens[key_0], text_1),
                (image_tokens[key_1], text_0),
                (image_tokens[key_1], text_1),
            ):
                pair_image_tokens.append(tokens)
                pair_soft_cues.append(text_entry["soft_cues"])
                pair_text_features.append(text_entry["text_features"])
        if not valid_samples:
            continue
        data_done = time.perf_counter()
        if pipeline_cfg is not None:
            score_values, timing = _score_conditioned_pairs_pipeline(
                model,
                pair_image_tokens,
                pair_soft_cues,
                pair_text_features,
                pipeline_cfg,
                logger,
                run_dir,
                benchmark_name,
                dist_ctx,
                start // max(1, sample_chunk),
            )
            data_time = data_done - data_start
        else:
            scores = _score_conditioned_pairs_with_retry(
                model,
                torch.stack(pair_image_tokens),
                torch.stack(pair_soft_cues),
                torch.stack(pair_text_features),
                score_chunk,
                logger,
            )
            score_values = scores.detach().cpu().float().reshape(-1).tolist()
            timing = {"h2d_time": 0.0, "gpu_time": 0.0}
            data_time = data_done - data_start
        cursor = 0
        for sample in valid_samples:
            s00 = float(score_values[cursor])
            s01 = float(score_values[cursor + 1])
            s10 = float(score_values[cursor + 2])
            s11 = float(score_values[cursor + 3])
            cursor += 4
            text_margin_0 = s00 - s01
            text_margin_1 = s11 - s10
            image_margin_0 = s00 - s10
            image_margin_1 = s11 - s01
            text_score = int(text_margin_0 > 0 and text_margin_1 > 0)
            image_score = int(image_margin_0 > 0 and image_margin_1 > 0)
            group_score = int(text_score == 1 and image_score == 1)
            rows.append(
                {
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "benchmark": benchmark_name,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "image_0_path": str(sample.image_0_path),
                    "image_1_path": str(sample.image_1_path),
                    "caption_0": sample.caption_0,
                    "caption_1": sample.caption_1,
                    "S00": s00,
                    "S01": s01,
                    "S10": s10,
                    "S11": s11,
                    "text_score": text_score,
                    "image_score": image_score,
                    "group_score": group_score,
                    "text_margin_0": text_margin_0,
                    "text_margin_1": text_margin_1,
                    "image_margin_0": image_margin_0,
                    "image_margin_1": image_margin_1,
                    "group_min_margin": min(text_margin_0, text_margin_1, image_margin_0, image_margin_1),
                    "source_file": str(sample.source_file) if sample.source_file else "",
                }
            )
        if start == 0 or start + sample_chunk >= len(samples) or (start + sample_chunk) % (sample_chunk * 10) == 0:
            throughput = len(score_values) / max(timing["gpu_time"], 1e-9) if pipeline_cfg is not None else 0.0
            logger.info(
                "%s/%s rank=%d raw rows scored: %d/%d data=%.4fs h2d=%.4fs gpu=%.4fs throughput=%.2f pairs/s",
                model.name,
                benchmark_name,
                dist_ctx.rank,
                min(start + sample_chunk, len(samples)),
                len(samples),
                data_time,
                timing["h2d_time"],
                timing["gpu_time"],
                throughput,
            )
    if dist_ctx.use_shards:
        _write_jsonl_rows(rows, raw_path)
    else:
        write_dataframe(pd.DataFrame(rows), raw_path)
    local_text = sum(int(row["text_score"]) for row in rows)
    local_image = sum(int(row["image_score"]) for row in rows)
    local_group = sum(int(row["group_score"]) for row in rows)
    local_total = len(rows)
    global_text, global_image, global_group, global_total = _dist_all_reduce_sum([local_text, local_image, local_group, local_total], dist_ctx)
    if dist_ctx.is_rank0:
        logger.info(
            "%s/%s metric aggregate: text=%d image=%d group=%d total=%d",
            model.name,
            benchmark_name,
            int(global_text),
            int(global_image),
            int(global_group),
            int(global_total),
        )
    logger.info("Wrote %d raw rows to %s", len(rows), raw_path)
    if dist_ctx.use_shards:
        return _merge_rank_jsonl(run_dir, model.name, benchmark_name, dist_ctx)
    return pd.DataFrame(rows)


def evaluate_winoground(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
    batch_size: int = 64,
    eval_mode: dict[str, Any] | None = None,
    score_chunk: int = 8192,
    dry_run: bool = False,
    dist_ctx: DistributedEvalContext | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    dist_ctx = dist_ctx or DistributedEvalContext()
    benchmark_name = benchmark.name
    eval_mode = eval_mode or {}
    spvd_pairwise_mode = str(eval_mode.get("spvd_pairwise_mode", "exact_cached"))
    if model.supports_text_conditioned_pair_cache and spvd_pairwise_mode == "exact_cached":
        return _evaluate_winoground_with_text_conditioned_cache(
            model,
            benchmark,
            run_dir,
            batch_size,
            logger,
            skipped_samples,
            score_chunk,
            dry_run,
            dist_ctx,
            pipeline_cfg,
        )
    samples_for_rank = _shard_list(list(benchmark.samples), dist_ctx) if dist_ctx.use_shards else list(benchmark.samples)
    raw_path = _rank_jsonl_path(run_dir, model.name, benchmark_name, dist_ctx.rank) if dist_ctx.use_shards else _merged_csv_path(run_dir, model.name, benchmark_name)
    if raw_path.exists():
        raw_path.unlink()
    if dist_ctx.use_shards and dist_ctx.is_rank0:
        merged_path = _merged_csv_path(run_dir, model.name, benchmark_name)
        if merged_path.exists():
            merged_path.unlink()
        total = len(benchmark.samples)
        logger.info(
            "%s/%s distributed naive path: total_samples=%d per_rank=%s num_workers=%d pair_chunk_size=%d",
            model.name,
            benchmark_name,
            total,
            [len(list(benchmark.samples)[r :: dist_ctx.world_size]) for r in range(dist_ctx.world_size)],
            pipeline_cfg.num_workers_per_gpu if pipeline_cfg else 0,
            pipeline_cfg.pair_chunk_size if pipeline_cfg else score_chunk,
        )
    image_cache = ImageCache()
    rows: list[dict[str, Any]] = []
    for sample in samples_for_rank:
        assert isinstance(sample, WinogroundSample)
        try:
            image_0 = image_cache.get(sample.image_0_path)
            image_1 = image_cache.get(sample.image_1_path)
            scores = model.score_batch([image_0, image_1], [sample.caption_0, sample.caption_1]).detach().cpu().float()
            s00, s01 = float(scores[0, 0]), float(scores[0, 1])
            s10, s11 = float(scores[1, 0]), float(scores[1, 1])
            text_margin_0 = s00 - s01
            text_margin_1 = s11 - s10
            image_margin_0 = s00 - s10
            image_margin_1 = s11 - s01
            text_score = int(text_margin_0 > 0 and text_margin_1 > 0)
            image_score = int(image_margin_0 > 0 and image_margin_1 > 0)
            group_score = int(text_score == 1 and image_score == 1)
            rows.append(
                {
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "benchmark": benchmark_name,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "image_0_path": str(sample.image_0_path),
                    "image_1_path": str(sample.image_1_path),
                    "caption_0": sample.caption_0,
                    "caption_1": sample.caption_1,
                    "S00": s00,
                    "S01": s01,
                    "S10": s10,
                    "S11": s11,
                    "text_score": text_score,
                    "image_score": image_score,
                    "group_score": group_score,
                    "text_margin_0": text_margin_0,
                    "text_margin_1": text_margin_1,
                    "image_margin_0": image_margin_0,
                    "image_margin_1": image_margin_1,
                    "group_min_margin": min(text_margin_0, text_margin_1, image_margin_0, image_margin_1),
                    "source_file": str(sample.source_file) if sample.source_file else "",
                }
            )
        except Exception as exc:
            skipped_samples.append(
                {
                    "benchmark": benchmark_name,
                    "source_file": str(sample.source_file) if sample.source_file else "",
                    "sample_id": sample.sample_id,
                    "reason": f"evaluation failed: {exc!r}",
                }
            )
            logger.exception("%s sample failed: %s", benchmark_name, sample.sample_id)
    if dist_ctx.use_shards:
        _write_jsonl_rows(rows, raw_path)
    else:
        write_dataframe(pd.DataFrame(rows), raw_path)
    local_text = sum(int(row["text_score"]) for row in rows)
    local_image = sum(int(row["image_score"]) for row in rows)
    local_group = sum(int(row["group_score"]) for row in rows)
    local_total = len(rows)
    global_text, global_image, global_group, global_total = _dist_all_reduce_sum([local_text, local_image, local_group, local_total], dist_ctx)
    if dist_ctx.is_rank0:
        logger.info(
            "%s/%s metric aggregate: text=%d image=%d group=%d total=%d",
            model.name,
            benchmark_name,
            int(global_text),
            int(global_image),
            int(global_group),
            int(global_total),
        )
    logger.info("Wrote %d raw rows to %s", len(rows), raw_path)
    if dist_ctx.use_shards:
        return _merge_rank_jsonl(run_dir, model.name, benchmark_name, dist_ctx)
    return pd.DataFrame(rows)


def write_summaries(
    run_dir: Path,
    model_info: dict[str, dict[str, Any]],
    pairwise_df: pd.DataFrame,
    winoground_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summaries_dir = ensure_dir(run_dir / "summaries")
    aro_summary = summarize_aro(pairwise_df)
    sugar_summary = summarize_sugarcrepe(pairwise_df)
    winoground_summary = summarize_winoground(winoground_df)
    pairwise_summary = summarize_pairwise_by_benchmark(pairwise_df)
    two_by_two_summary = summarize_2x2_by_benchmark(winoground_df)
    margins_summary = summarize_margins(pairwise_df)
    ssr_summary = summarize_ssr(pairwise_df)
    all_summary = summary_all_models(
        model_info,
        aro_summary,
        sugar_summary,
        winoground_summary,
        margins_summary,
        ssr_summary,
        pairwise_summary,
        two_by_two_summary,
    )
    outputs = {
        "summary_all_models": all_summary,
        "summary_aro_by_category": aro_summary,
        "summary_sugarcrepe_by_category": sugar_summary,
        "summary_winoground": winoground_summary,
        "summary_pairwise_by_benchmark": pairwise_summary,
        "summary_2x2_by_benchmark": two_by_two_summary,
        "summary_margins": margins_summary,
        "summary_ssr": ssr_summary,
    }
    for name, df in outputs.items():
        write_dataframe(df, summaries_dir / f"{name}.csv")
    return outputs


def _fmt(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NaN"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_analysis_notes(run_dir: Path, summaries: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> None:
    all_summary = summaries["summary_all_models"]
    margins = summaries["summary_margins"]
    ssr = summaries["summary_ssr"]
    lines: list[str] = []
    lines.append("# Zero-Training Diagnostic Analysis Notes")
    lines.append("")
    lines.append("This run evaluates frozen pretrained image-text models only. It performs scoring, benchmark evaluation, summary statistics, and plotting; it does not train, backpropagate, update model parameters, or write model checkpoints.")
    lines.append("")
    lines.append("## Environment")
    env = manifest.get("environment", {})
    for key in ["python_executable", "conda_environment", "torch_version", "cuda_available", "gpu_name", "dataset_root", "model_root", "output_dir"]:
        lines.append(f"- {key}: {env.get(key)}")
    lines.append("")
    lines.append("## Main Results")
    if all_summary.empty:
        lines.append("No successful model-benchmark results were produced.")
    else:
        for _, row in all_summary.iterrows():
            lines.append(
                f"- {row['model_name']}: ARO overall={_fmt(row.get('ARO_overall_acc'))}, "
                f"SugarCrepe overall={_fmt(row.get('SugarCrepe_overall_acc'))}, "
                f"Winoground text/image/group={_fmt(row.get('Winoground_text_score'))}/{_fmt(row.get('Winoground_image_score'))}/{_fmt(row.get('Winoground_group_score'))}, "
                f"hard margin mean={_fmt(row.get('Hard_margin_mean'))}, random margin mean={_fmt(row.get('Random_margin_mean'))}, SSR mean={_fmt(row.get('SSR_mean'))}."
            )
    lines.append("")
    lines.append("## Hard vs Random Negative Margins")
    if margins.empty:
        lines.append("Random-negative margin summaries were not available.")
    else:
        gap = margins.copy()
        gap = gap.sort_values("hard_vs_random_gap", ascending=False)
        for _, row in gap.head(12).iterrows():
            lines.append(
                f"- {row['model_name']} / {row['benchmark']} / {row['category']}: "
                f"hard={_fmt(row['hard_margin_mean'])}, random={_fmt(row['random_margin_mean'])}, "
                f"random-minus-hard={_fmt(row['hard_vs_random_gap'])}."
            )
    lines.append("")
    lines.append("## Category-Level Failure Signals")
    for summary_name, label in [("summary_aro_by_category", "ARO"), ("summary_sugarcrepe_by_category", "SugarCrepe")]:
        df = summaries[summary_name]
        if df.empty:
            continue
        df = df[df["category"] != "overall"].sort_values("accuracy", ascending=True)
        lines.append(f"{label} lowest-accuracy slices:")
        for _, row in df.head(10).iterrows():
            sub = row.get("subcategory")
            sub_text = f" / {sub}" if isinstance(sub, str) and sub else ""
            lines.append(f"- {row['model_name']} / {row['category']}{sub_text}: acc={_fmt(row['accuracy'])}, mean_margin={_fmt(row['mean_margin'])}, n={int(row['sample_count'])}.")
        lines.append("")
    lines.append("## Semantic Sensitivity Ratio")
    if ssr.empty:
        lines.append("SSR summaries were not available.")
    else:
        for _, row in ssr.sort_values("SSR_mean", ascending=True).head(12).iterrows():
            lines.append(
                f"- {row['model_name']} / {row['benchmark']} / {row['category']}: "
                f"SSR mean={_fmt(row['SSR_mean'])}, median={_fmt(row['SSR_median'])}, "
                f"invalid random margin ratio={_fmt(row['invalid_random_margin_ratio'])}."
            )
    lines.append("")
    lines.append("## Interpretation Guidance")
    lines.append("- Interpret raw scores and margins only within the same model; do not compare raw score magnitudes across models.")
    lines.append("- The cautious claim supported by this diagnostic is limited sensitivity to certain language-relevant semantic perturbations, especially when hard-negative margins are much smaller than random-negative margins.")
    lines.append("- Avoid claims such as \"CLIP does not understand language\" or \"this proves semantic absence\". Prefer phrasing that coarse image-text alignment may obscure weaknesses in caption-level semantic discrimination.")
    lines.append("- Comparisons between OpenAI CLIP, OpenCLIP, SigLIP, and BLIP ITM should be framed as diagnostic trends under frozen scoring rather than broad conclusions about model families.")
    lines.append("")
    lines.append("## Introduction Draft")
    lines.append("Before introducing our method, we conduct a zero-training diagnostic study on frozen image-text models using ARO, SugarCrepe, and Winoground. These benchmarks probe attribute binding, relation understanding, word order, and hard-negative caption discrimination. We observe that several pretrained models can often separate matched image-caption pairs from random negatives, yet their margins become substantially smaller when the negative captions involve minimal semantic perturbations. This suggests that coarse image-text alignment may obscure weaknesses in caption-level semantic discrimination, motivating our study of semantic preservation during visual decomposition.")
    lines.append("")
    lines.append("## Missing Or Skipped Items")
    for warning in manifest.get("warnings", []):
        lines.append(f"- warning: {warning}")
    for model_name in manifest.get("skipped_models", []):
        lines.append(f"- skipped model: {model_name}")
    for benchmark_name in manifest.get("skipped_benchmarks", []):
        lines.append(f"- skipped benchmark: {benchmark_name}")
    (run_dir / "analysis_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(args: argparse.Namespace) -> Path:
    config = read_yaml(Path(args.config))
    eval_mode = config.get("eval_mode", {}) or {}
    distributed_eval = bool(config.get("distributed_eval", eval_mode.get("distributed_eval", False)))
    dist_ctx = init_distributed_eval(args.device or config.get("device", "cuda"), distributed_eval)
    dataset_root = Path(args.dataset_root or config.get("dataset_root") or "/vepfs/dataset")
    model_root = Path(args.model_root or config.get("model_root") or "/vepfs/model")
    output_base = Path(args.output_dir or config.get("output_dir") or "outputs/zero_train_diagnostics")
    seed = int(args.seed if args.seed is not None else config.get("random_seed", 42))
    set_seed(seed)
    timestamp = _broadcast_run_id(datetime.now().strftime("run_%Y%m%d_%H%M%S"), dist_ctx)
    run_dir = ensure_dir(output_base / timestamp)
    ensure_dir(run_dir / "raw_results")
    ensure_dir(run_dir / "summaries")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "perf")
    logger = setup_logging(run_dir / "logs" / (f"eval_rank{dist_ctx.rank}.log" if dist_ctx.use_shards else "eval.log"), verbose=dist_ctx.is_rank0)
    logger.info("Starting zero-training diagnostics in %s rank=%d/%d local_rank=%d device=%s distributed_eval=%s", run_dir, dist_ctx.rank, dist_ctx.world_size, dist_ctx.local_rank, dist_ctx.device, dist_ctx.use_shards)
    expanded_config = dict(config)
    expanded_config.update(
        {
            "dataset_root": str(dataset_root),
            "model_root": str(model_root),
            "output_dir": str(output_base),
            "device": dist_ctx.device,
            "dtype": args.dtype or config.get("dtype", "fp32"),
            "batch_size": int(args.batch_size or config.get("batch_size", 64)),
            "num_workers": int(args.num_workers or config.get("num_workers", 0)),
            "random_seed": seed,
            "eval_mode": eval_mode,
            "distributed": {
                "rank": dist_ctx.rank,
                "world_size": dist_ctx.world_size,
                "local_rank": dist_ctx.local_rank,
                "use_shards": dist_ctx.use_shards,
            },
        }
    )
    if dist_ctx.is_rank0:
        write_yaml(expanded_config, run_dir / "config.yaml")
    environment = collect_environment_info(dataset_root, model_root, output_base)
    manifest: dict[str, Any] = {
        "run_time": now_utc_iso(),
        "command": " ".join(sys.argv),
        "environment": environment,
        "python_executable": environment.get("python_executable"),
        "conda_environment": environment.get("conda_environment"),
        "torch_version": environment.get("torch_version"),
        "cuda_available": environment.get("cuda_available"),
        "gpu_name": environment.get("gpu_name"),
        "dataset_root": str(dataset_root),
        "model_root": str(model_root),
        "output_dir": str(run_dir),
        "model_loading_status": {},
        "benchmark_loading_status": {},
        "skipped_models": [],
        "skipped_benchmarks": [],
        "error_messages": [],
        "warnings": [],
        "distributed": {
            "rank": dist_ctx.rank,
            "world_size": dist_ctx.world_size,
            "local_rank": dist_ctx.local_rank,
            "use_shards": dist_ctx.use_shards,
        },
    }
    expected_env = str(config.get("conda_env") or "openclip")
    if environment.get("conda_environment") != expected_env:
        warning = f"Expected conda env {expected_env}, got {environment.get('conda_environment')}"
        logger.warning(warning)
        manifest["warnings"].append(warning)
    for key, value in environment.items():
        logger.info("ENV %s=%s", key, value)
    if dist_ctx.is_rank0:
        write_json(manifest, run_dir / "manifest.json")

    limit = int(args.limit) if args.limit is not None else None
    if args.dry_run and limit is None:
        limit = int(config.get("dry_run_limit", 8))
    benchmarks = load_benchmarks(dataset_root, limit, expanded_config, seed, logger)
    skipped_samples: list[dict[str, Any]] = []
    for name, result in benchmarks.items():
        manifest["benchmark_loading_status"][name] = {
            "status": result.status,
            "sample_count": len(result.samples),
            "data_files": [str(p) for p in result.data_files],
            "candidates": [str(p) for p in result.candidates],
            "warnings": result.warnings,
            "metadata": result.metadata,
        }
        manifest["warnings"].extend(result.warnings)
        skipped_samples.extend(result.skipped_samples)
        if result.status != "ok":
            manifest["skipped_benchmarks"].append(name)
    if dist_ctx.is_rank0:
        write_json(manifest, run_dir / "manifest.json")

    models_cfg = list(config.get("models") or [])
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        models_cfg = [cfg for cfg in models_cfg if cfg.get("name") in wanted]
    device = dist_ctx.device
    dtype = args.dtype or config.get("dtype", "fp32")
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    score_chunk = int(eval_mode.get("score_chunk", config.get("score_chunk", 8192)))
    pipeline_cfg = _pipeline_config(config, eval_mode, batch_size, score_chunk)
    if dist_ctx.is_rank0:
        logger.info(
            "Evaluation pipeline: total_models=%d world_size=%d batch_size=%d num_workers_per_gpu=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s pair_chunk_size=%d dtype=%s",
            len(models_cfg) if "models_cfg" in locals() else len(config.get("models") or []),
            dist_ctx.world_size,
            pipeline_cfg.batch_size,
            pipeline_cfg.num_workers_per_gpu,
            pipeline_cfg.pin_memory,
            pipeline_cfg.persistent_workers,
            pipeline_cfg.prefetch_factor,
            pipeline_cfg.pair_chunk_size,
            dtype,
        )
    compute_random_negative = bool(config.get("compute_random_negative", True))
    pairwise_frames: list[pd.DataFrame] = []
    winoground_frames: list[pd.DataFrame] = []
    model_info: dict[str, dict[str, Any]] = {}

    for model_cfg in models_cfg:
        model_name = str(model_cfg.get("name"))
        logger.info("Evaluating model %s", model_name)
        loaded = load_model(model_cfg, model_root, device, dtype, logger)
        manifest["model_loading_status"][model_name] = {
            "status": loaded.status,
            "model_type": loaded.model_type,
            "checkpoint_path": str(loaded.checkpoint_path) if loaded.checkpoint_path else None,
            "local_dir": str(loaded.local_dir) if loaded.local_dir else None,
            "parameter_count": loaded.parameter_count,
            "device": loaded.device,
            "dtype": loaded.dtype,
            "dummy_forward_success": loaded.dummy_forward_success,
            "error": loaded.error,
        }
        if loaded.status != "ok" or loaded.wrapper is None:
            manifest["skipped_models"].append(model_name)
            if loaded.error:
                manifest["error_messages"].append({model_name: loaded.error})
            if dist_ctx.is_rank0:
                write_json(manifest, run_dir / "manifest.json")
            continue
        model_info[model_name] = {"model_type": loaded.model_type}
        try:
            for benchmark_name, benchmark in benchmarks.items():
                if benchmark.status != "ok":
                    logger.warning("Skipping benchmark %s for %s due to benchmark status=%s", benchmark_name, model_name, benchmark.status)
                    continue
                logger.info("Scoring %s on %s (%d samples)", model_name, benchmark_name, len(benchmark.samples))
                if benchmark_name in {"aro", "sugarcrepe", "sugarcrepe_pp"}:
                    df = evaluate_pairwise_benchmark(
                        loaded.wrapper,
                        benchmark,
                        run_dir,
                        batch_size,
                        compute_random_negative,
                        seed,
                        logger,
                        skipped_samples,
                        eval_mode,
                        score_chunk,
                        bool(args.dry_run),
                        dist_ctx,
                        pipeline_cfg,
                    )
                    if dist_ctx.is_rank0 and not df.empty:
                        pairwise_frames.append(df)
                elif benchmark_name in {"winoground", "bivlc"}:
                    df = evaluate_winoground(
                        loaded.wrapper,
                        benchmark,
                        run_dir,
                        logger,
                        skipped_samples,
                        batch_size,
                        eval_mode,
                        score_chunk,
                        bool(args.dry_run),
                        dist_ctx,
                        pipeline_cfg,
                    )
                    if dist_ctx.is_rank0 and not df.empty:
                        winoground_frames.append(df)
                if dist_ctx.is_rank0:
                    write_json(manifest, run_dir / "manifest.json")
        except Exception as exc:
            logger.exception("Model evaluation failed for %s", model_name)
            manifest["error_messages"].append({model_name: repr(exc)})
        finally:
            loaded.wrapper.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if dist_ctx.is_rank0:
                write_json(manifest, run_dir / "manifest.json")

    _dist_barrier(dist_ctx)
    if not dist_ctx.is_rank0:
        _cleanup_distributed_eval(dist_ctx)
        return run_dir
    pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    winoground_df = pd.concat(winoground_frames, ignore_index=True) if winoground_frames else pd.DataFrame()
    summaries = write_summaries(run_dir, model_info, pairwise_df, winoground_df)
    if skipped_samples:
        write_dataframe(pd.DataFrame(skipped_samples), run_dir / "skipped_samples.csv")
    else:
        write_dataframe(pd.DataFrame(columns=["benchmark", "source_file", "sample_id", "reason"]), run_dir / "skipped_samples.csv")
    if bool(config.get("save_figures", True)) and not args.dry_run:
        generate_figures(
            pairwise_df,
            summaries["summary_aro_by_category"],
            summaries["summary_sugarcrepe_by_category"],
            summaries["summary_winoground"],
            summaries["summary_margins"],
            summaries["summary_ssr"],
            run_dir / "figures",
        )
    else:
        logger.info("Skipping full figure generation (dry_run=%s save_figures=%s)", args.dry_run, config.get("save_figures", True))
    write_analysis_notes(run_dir, summaries, manifest)
    manifest["completed_at"] = now_utc_iso()
    manifest["raw_result_files"] = [str(p) for p in sorted((run_dir / "raw_results").glob("*.csv"))]
    manifest["summary_files"] = [str(p) for p in sorted((run_dir / "summaries").glob("*.csv"))]
    manifest["figure_files"] = [str(p) for p in sorted((run_dir / "figures").glob("*"))]
    write_json(manifest, run_dir / "manifest.json")
    logger.info("Finished zero-training diagnostics. Run dir: %s", run_dir)
    _cleanup_distributed_eval(dist_ctx)
    return run_dir
