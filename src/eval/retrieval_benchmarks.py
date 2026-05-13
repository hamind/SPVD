"""COCO/Flickr30k retrieval benchmarks for frozen VLM wrappers."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from PIL import Image

from zero_train_diagnostics.models import load_model


@dataclass(frozen=True)
class RetrievalDist:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    initialized: bool = False

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _init_dist(device_arg: str) -> RetrievalDist:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    initialized = False
    device = device_arg
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun retrieval requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        initialized = True
        device = f"cuda:{local_rank}"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return RetrievalDist(rank=rank, world_size=world_size, local_rank=local_rank, device=device, initialized=initialized)


def _barrier(ctx: RetrievalDist) -> None:
    if ctx.initialized:
        dist.barrier()


def _cleanup_dist(ctx: RetrievalDist) -> None:
    if ctx.initialized and dist.is_initialized():
        dist.destroy_process_group()


def _shard_indices(n: int, ctx: RetrievalDist) -> list[int]:
    return list(range(n))[ctx.rank :: ctx.world_size]


def _gather_indexed_tensor(indices: list[int], features: torch.Tensor, total: int, ctx: RetrievalDist) -> torch.Tensor | None:
    if not ctx.initialized:
        out = torch.empty((total, features.shape[1]), dtype=features.dtype)
        out[torch.tensor(indices, dtype=torch.long)] = features.cpu()
        return out
    payload = [(indices, features.cpu())]
    gathered: list[Any] = [None for _ in range(ctx.world_size)]
    dist.gather_object(payload[0], gathered if ctx.is_rank0 else None, dst=0)
    if not ctx.is_rank0:
        return None
    nonempty = [item for item in gathered if int(item[1].shape[0]) > 0]
    if not nonempty:
        return torch.empty(total, 0)
    feat_dim = int(nonempty[0][1].shape[1])
    out = torch.empty((total, feat_dim), dtype=gathered[0][1].dtype)
    for rank_indices, rank_features in gathered:
        out[torch.tensor(rank_indices, dtype=torch.long)] = rank_features
    return out


def _gather_pair_scores(rows: list[tuple[int, int, float]], ctx: RetrievalDist) -> list[tuple[int, int, float]]:
    if not ctx.initialized:
        return rows
    gathered: list[Any] = [None for _ in range(ctx.world_size)]
    dist.gather_object(rows, gathered if ctx.is_rank0 else None, dst=0)
    if not ctx.is_rank0:
        return []
    out: list[tuple[int, int, float]] = []
    for part in gathered:
        out.extend(part)
    return out


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    image_paths: list[Path]
    image_cues: list[str]
    captions: list[str]
    caption_image_indices: list[int]
    split: str


def _setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("retrieval_benchmarks")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "retrieval_benchmarks.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _caption_text(sentence: Any) -> str:
    if isinstance(sentence, dict):
        return str(sentence.get("raw") or " ".join(sentence.get("tokens") or [])).strip()
    return str(sentence).strip()


def _limit_records(records: list[dict[str, Any]], limit_images: int | None) -> list[dict[str, Any]]:
    if limit_images is None or limit_images <= 0:
        return records
    return records[:limit_images]


def load_coco(dataset_root: Path, split: str, limit_images: int | None) -> RetrievalDataset:
    if split == "val2017":
        return load_coco_captions(dataset_root, split, limit_images)

    split_file = dataset_root / "coco" / "splits" / f"karpathy_{split}.json"
    rows = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"COCO split must be a list: {split_file}")

    image_paths: list[Path] = []
    image_cues: list[str] = []
    captions: list[str] = []
    caption_image_indices: list[int] = []
    for row in _limit_records(rows, limit_images):
        filepath = str(row.get("filepath") or "val2014")
        filename = str(row.get("filename"))
        image_path = dataset_root / "coco" / "images" / filepath / filename
        if not image_path.exists():
            raise FileNotFoundError(f"missing COCO image: {image_path}")
        image_idx = len(image_paths)
        image_paths.append(image_path)
        sentence_rows = row.get("sentences") or []
        cue_caption = _caption_text(sentence_rows[0]) if sentence_rows else ""
        image_cues.append(cue_caption)
        for sentence in sentence_rows:
            caption = _caption_text(sentence)
            if caption:
                captions.append(caption)
                caption_image_indices.append(image_idx)
    return RetrievalDataset("coco", image_paths, image_cues, captions, caption_image_indices, split)


def load_coco_captions(dataset_root: Path, split: str, limit_images: int | None) -> RetrievalDataset:
    annotation_file = dataset_root / "coco" / "annotations" / f"captions_{split}.json"
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    image_rows = payload.get("images", []) if isinstance(payload, dict) else []
    annotation_rows = payload.get("annotations", []) if isinstance(payload, dict) else []
    if not isinstance(image_rows, list) or not isinstance(annotation_rows, list):
        raise ValueError(f"COCO captions file must contain images and annotations: {annotation_file}")

    selected_images = sorted(image_rows, key=lambda row: (str(row.get("file_name", "")), int(row.get("id", 0))))
    selected_images = _limit_records(selected_images, limit_images)
    image_ids = {int(row["id"]) for row in selected_images if "id" in row}
    captions_by_image: dict[int, list[str]] = {image_id: [] for image_id in image_ids}
    for row in annotation_rows:
        image_id = int(row.get("image_id", -1))
        caption = str(row.get("caption") or "").strip()
        if image_id in captions_by_image and caption:
            captions_by_image[image_id].append(caption)

    image_root = dataset_root / "coco" / "images" / split
    image_paths: list[Path] = []
    image_cues: list[str] = []
    captions: list[str] = []
    caption_image_indices: list[int] = []
    for row in selected_images:
        image_id = int(row.get("id"))
        image_captions = captions_by_image.get(image_id, [])
        if not image_captions:
            continue
        image_path = image_root / str(row.get("file_name"))
        if not image_path.exists():
            raise FileNotFoundError(f"missing COCO image: {image_path}")
        image_idx = len(image_paths)
        image_paths.append(image_path)
        image_cues.append(image_captions[0])
        for caption in image_captions:
            captions.append(caption)
            caption_image_indices.append(image_idx)
    return RetrievalDataset("coco", image_paths, image_cues, captions, caption_image_indices, split)


def load_flickr30k(dataset_root: Path, split: str, limit_images: int | None) -> RetrievalDataset:
    annotation_file = dataset_root / "flickr30k" / "annotations" / "dataset_flickr30k_karpathy.json"
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    rows = payload.get("images", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Flickr30k annotations must contain an image list: {annotation_file}")

    filtered = [row for row in rows if str(row.get("split")) == split]
    image_root = dataset_root / "flickr30k" / "images"
    image_paths: list[Path] = []
    image_cues: list[str] = []
    captions: list[str] = []
    caption_image_indices: list[int] = []
    for row in _limit_records(filtered, limit_images):
        image_path = image_root / str(row.get("filename"))
        if not image_path.exists():
            raise FileNotFoundError(f"missing Flickr30k image: {image_path}")
        image_idx = len(image_paths)
        image_paths.append(image_path)
        sentence_rows = row.get("sentences") or []
        cue_caption = _caption_text(sentence_rows[0]) if sentence_rows else ""
        image_cues.append(cue_caption)
        for sentence in sentence_rows:
            caption = _caption_text(sentence)
            if caption:
                captions.append(caption)
                caption_image_indices.append(image_idx)
    return RetrievalDataset("flickr30k", image_paths, image_cues, captions, caption_image_indices, split)



def _normalize_retrieval_features(features: torch.Tensor) -> torch.Tensor:
    """Return 2D L2-normalized features before retrieval dot products."""
    if features.ndim == 3:
        features = features.mean(dim=1)
    if features.ndim != 2:
        raise ValueError(f"Retrieval features must be 2D or 3D, got shape={tuple(features.shape)}.")
    return torch.nn.functional.normalize(features.float(), dim=-1)

def _open_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


@torch.no_grad()
def encode_dataset(wrapper: Any, dataset: RetrievalDataset, batch_size: int, logger: logging.Logger) -> tuple[torch.Tensor, torch.Tensor]:
    image_features: list[torch.Tensor] = []
    for start in range(0, len(dataset.image_paths), batch_size):
        end = min(start + batch_size, len(dataset.image_paths))
        images = _open_images(dataset.image_paths[start:end])
        image_features.append(wrapper.encode_images(images))
        if start == 0 or end == len(dataset.image_paths) or end % max(batch_size * 20, 1) == 0:
            logger.info("%s image features: %d/%d encoded", dataset.name, end, len(dataset.image_paths))

    text_features: list[torch.Tensor] = []
    for start in range(0, len(dataset.captions), batch_size):
        end = min(start + batch_size, len(dataset.captions))
        text_features.append(wrapper.encode_texts(dataset.captions[start:end]))
        if start == 0 or end == len(dataset.captions) or end % max(batch_size * 20, 1) == 0:
            logger.info("%s text features: %d/%d encoded", dataset.name, end, len(dataset.captions))

    return (
        _normalize_retrieval_features(torch.cat(image_features, dim=0)),
        _normalize_retrieval_features(torch.cat(text_features, dim=0)),
    )


@torch.inference_mode()
def encode_dataset_distributed(wrapper: Any, dataset: RetrievalDataset, batch_size: int, logger: logging.Logger, ctx: RetrievalDist) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_indices = _shard_indices(len(dataset.image_paths), ctx)
    text_indices = _shard_indices(len(dataset.captions), ctx)

    image_features: list[torch.Tensor] = []
    for start in range(0, len(image_indices), batch_size):
        batch_indices = image_indices[start : start + batch_size]
        images = _open_images([dataset.image_paths[idx] for idx in batch_indices])
        image_features.append(wrapper.encode_images(images))
        if ctx.is_rank0 and (start == 0 or start + batch_size >= len(image_indices)):
            logger.info("%s image shard rank=%d encoded %d/%d", dataset.name, ctx.rank, min(start + batch_size, len(image_indices)), len(image_indices))

    text_features: list[torch.Tensor] = []
    for start in range(0, len(text_indices), batch_size):
        batch_indices = text_indices[start : start + batch_size]
        text_features.append(wrapper.encode_texts([dataset.captions[idx] for idx in batch_indices]))
        if ctx.is_rank0 and (start == 0 or start + batch_size >= len(text_indices)):
            logger.info("%s text shard rank=%d encoded %d/%d", dataset.name, ctx.rank, min(start + batch_size, len(text_indices)), len(text_indices))

    local_image_features = _normalize_retrieval_features(torch.cat(image_features, dim=0)) if image_features else torch.empty(0, 1)
    local_text_features = _normalize_retrieval_features(torch.cat(text_features, dim=0)) if text_features else torch.empty(0, 1)
    all_image_features = _gather_indexed_tensor(image_indices, local_image_features, len(dataset.image_paths), ctx)
    all_text_features = _gather_indexed_tensor(text_indices, local_text_features, len(dataset.captions), ctx)
    _barrier(ctx)
    return all_image_features, all_text_features


@torch.inference_mode()
def rerank_topk_spvd(
    wrapper: Any,
    dataset: RetrievalDataset,
    similarity: torch.Tensor,
    topk: int,
    batch_size: int,
    logger: logging.Logger,
    ctx: RetrievalDist,
) -> torch.Tensor | None:
    if not getattr(wrapper, "supports_text_conditioned_pair_cache", False):
        return similarity if ctx.is_rank0 else None
    if ctx.is_rank0:
        topk = max(1, min(int(topk), similarity.shape[1], similarity.shape[0]))
        pairs: set[tuple[int, int]] = set()
        image_topk = torch.topk(similarity, k=min(topk, similarity.shape[1]), dim=1).indices
        for image_idx in range(image_topk.shape[0]):
            for text_idx in image_topk[image_idx].tolist():
                pairs.add((image_idx, int(text_idx)))
        text_topk = torch.topk(similarity, k=min(topk, similarity.shape[0]), dim=0).indices
        for text_idx in range(text_topk.shape[1]):
            for image_idx in text_topk[:, text_idx].tolist():
                pairs.add((int(image_idx), text_idx))
        pair_list = sorted(pairs)
    else:
        pair_list = []
    if ctx.initialized:
        payload = [pair_list]
        dist.broadcast_object_list(payload, src=0)
        pair_list = payload[0]

    local_pairs = pair_list[ctx.rank :: ctx.world_size]
    unique_images = sorted({image_idx for image_idx, _ in local_pairs})
    unique_texts = sorted({text_idx for _, text_idx in local_pairs})
    image_cache: dict[int, torch.Tensor] = {}
    text_cache: dict[int, dict[str, torch.Tensor]] = {}

    for start in range(0, len(unique_images), batch_size):
        batch_indices = unique_images[start : start + batch_size]
        images = _open_images([dataset.image_paths[idx] for idx in batch_indices])
        tokens = wrapper.encode_image_tokens(images)
        for idx, token in zip(batch_indices, tokens):
            image_cache[idx] = token.cpu()
    for start in range(0, len(unique_texts), batch_size):
        batch_indices = unique_texts[start : start + batch_size]
        encoded = wrapper.encode_text_cues([dataset.captions[idx] for idx in batch_indices])
        for offset, idx in enumerate(batch_indices):
            text_cache[idx] = {key: value[offset].cpu() for key, value in encoded.items()}

    local_rows: list[tuple[int, int, float]] = []
    for start in range(0, len(local_pairs), batch_size):
        chunk = local_pairs[start : start + batch_size]
        image_tokens = torch.stack([image_cache[image_idx] for image_idx, _ in chunk])
        soft_cues = torch.stack([text_cache[text_idx]["soft_cues"] for _, text_idx in chunk])
        text_features = torch.stack([text_cache[text_idx]["text_features"] for _, text_idx in chunk])
        scores = wrapper.score_cached_conditioned_pairs(image_tokens, soft_cues, text_features).detach().cpu().float().tolist()
        local_rows.extend((image_idx, text_idx, float(score)) for (image_idx, text_idx), score in zip(chunk, scores))
    gathered = _gather_pair_scores(local_rows, ctx)
    if not ctx.is_rank0:
        return None
    reranked = similarity.clone()
    for image_idx, text_idx, score in gathered:
        reranked[image_idx, text_idx] = score
    logger.info("%s rerank_topk exact SPVD pairs scored: %d", wrapper.name, len(gathered))
    return reranked


@torch.no_grad()
def score_soft_cue_conditioned_dataset(
    wrapper: Any,
    dataset: RetrievalDataset,
    batch_size: int,
    logger: logging.Logger,
    ctx: RetrievalDist,
) -> torch.Tensor | None:
    if not getattr(wrapper, "supports_text_conditioned_pair_cache", False):
        raise ValueError(f"{wrapper.name} does not support SPVD full text-conditioned retrieval")

    image_cache_batch = max(1, min(batch_size, 128))
    text_cache_batch = max(1, batch_size)
    image_score_batch = max(1, min(batch_size // 4 if batch_size >= 4 else batch_size, 64))
    text_score_batch = max(1, batch_size)
    image_indices = _shard_indices(len(dataset.image_paths), ctx)

    if ctx.is_rank0:
        logger.info(
            "%s uses full SPVD soft-cue-conditioned retrieval: world_size=%d image_cache_bs=%d text_cache_bs=%d image_score_bs=%d text_score_bs=%d",
            wrapper.name,
            ctx.world_size,
            image_cache_batch,
            text_cache_batch,
            image_score_batch,
            text_score_batch,
        )

    image_token_chunks: list[torch.Tensor] = []
    for start in range(0, len(image_indices), image_cache_batch):
        batch_indices = image_indices[start : start + image_cache_batch]
        images = _open_images([dataset.image_paths[idx] for idx in batch_indices])
        image_token_chunks.append(wrapper.encode_image_tokens(images))
        if ctx.is_rank0 and (start == 0 or start + image_cache_batch >= len(image_indices)):
            logger.info(
                "%s conditioned image tokens rank=%d: %d/%d encoded",
                dataset.name,
                ctx.rank,
                min(start + image_cache_batch, len(image_indices)),
                len(image_indices),
            )
    image_tokens = torch.cat(image_token_chunks, dim=0) if image_token_chunks else torch.empty(0, 1)

    soft_cue_chunks: list[torch.Tensor] = []
    text_feature_chunks: list[torch.Tensor] = []
    for start in range(0, len(dataset.captions), text_cache_batch):
        end = min(start + text_cache_batch, len(dataset.captions))
        text_cache = wrapper.encode_text_cues(dataset.captions[start:end])
        soft_cue_chunks.append(text_cache["soft_cues"])
        text_feature_chunks.append(text_cache["text_features"])
        if ctx.is_rank0 and (start == 0 or end == len(dataset.captions) or end % max(text_cache_batch * 20, 1) == 0):
            logger.info("%s conditioned text cues: %d/%d encoded", dataset.name, end, len(dataset.captions))
    soft_cues = torch.cat(soft_cue_chunks, dim=0)
    text_features = torch.cat(text_feature_chunks, dim=0)

    local_similarity = torch.empty((len(image_indices), len(dataset.captions)), dtype=torch.float32)
    for image_start in range(0, len(image_indices), image_score_batch):
        image_end = min(image_start + image_score_batch, len(image_indices))
        for text_start in range(0, len(dataset.captions), text_score_batch):
            text_end = min(text_start + text_score_batch, len(dataset.captions))
            scores = wrapper.score_conditioned_retrieval_chunk(
                image_tokens[image_start:image_end],
                soft_cues[text_start:text_end],
                text_features[text_start:text_end],
            )
            local_similarity[image_start:image_end, text_start:text_end] = scores
        if ctx.is_rank0 and (image_start == 0 or image_end == len(image_indices) or image_end % max(image_score_batch * 20, 1) == 0):
            logger.info("%s full SPVD retrieval rows rank=%d: %d/%d scored", dataset.name, ctx.rank, image_end, len(image_indices))

    similarity = _gather_indexed_tensor(image_indices, local_similarity, len(dataset.image_paths), ctx)
    _barrier(ctx)
    return similarity



def _recall_metrics(ranks: torch.Tensor, prefix: str) -> dict[str, float]:
    ranks = ranks.float()
    return {
        f"{prefix}_r@1": float((ranks <= 1).float().mean().item() * 100),
        f"{prefix}_r@5": float((ranks <= 5).float().mean().item() * 100),
        f"{prefix}_r@10": float((ranks <= 10).float().mean().item() * 100),
        f"{prefix}_mean_rank": float(ranks.mean().item()),
        f"{prefix}_median_rank": float(ranks.median().item()),
    }


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_image_indices: list[int],
) -> dict[str, float]:
    image_features = _normalize_retrieval_features(image_features)
    text_features = _normalize_retrieval_features(text_features)
    similarity = image_features @ text_features.t()
    return compute_retrieval_metrics_from_similarity(similarity, caption_image_indices)


def compute_retrieval_metrics_from_similarity(
    similarity: torch.Tensor,
    caption_image_indices: list[int],
) -> dict[str, float]:
    caption_targets = torch.tensor(caption_image_indices, dtype=torch.long)

    i2t_ranks: list[torch.Tensor] = []
    for image_idx in range(similarity.shape[0]):
        positive_indices = (caption_targets == image_idx).nonzero(as_tuple=False).flatten()
        positive_score = similarity[image_idx, positive_indices].max()
        i2t_ranks.append((similarity[image_idx] > positive_score).sum() + 1)
    i2t = torch.stack(i2t_ranks)

    text_indices = torch.arange(similarity.shape[1], dtype=torch.long)
    positive_scores = similarity[caption_targets, text_indices]
    t2i = (similarity > positive_scores.unsqueeze(0)).sum(dim=0) + 1

    out = _recall_metrics(i2t, "i2t")
    out.update(_recall_metrics(t2i, "t2i"))
    return out


def _load_datasets(dataset_root: Path, names: list[str], split: str, limit_images: int | None) -> list[RetrievalDataset]:
    datasets: list[RetrievalDataset] = []
    for name in names:
        if name == "coco":
            datasets.append(load_coco(dataset_root, split, limit_images))
        elif name == "flickr30k":
            datasets.append(load_flickr30k(dataset_root, split, limit_images))
        else:
            raise ValueError(f"unsupported retrieval dataset: {name}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_all_models.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model names; default: all config models.")
    parser.add_argument("--datasets", default="coco,flickr30k")
    parser.add_argument("--dataset-root", default="/vepfs/dataset/benchmark")
    parser.add_argument("--model-root", default="/vepfs/model/models")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--output-dir", default="outputs/benchmark_retrieval")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--retrieval-mode", choices=["global", "rerank_topk", "spvd_full"], default=None)
    parser.add_argument("--rerank-topk", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = _setup_logger(output_dir)
    cfg = _read_yaml(Path(args.config))
    eval_mode = cfg.get("eval_mode", {}) if isinstance(cfg.get("eval_mode", {}), dict) else {}
    retrieval_mode = args.retrieval_mode or str(eval_mode.get("retrieval_mode", "global"))
    rerank_topk = int(args.rerank_topk or eval_mode.get("rerank_topk", 100))
    ctx = _init_dist(args.device)
    if ctx.is_rank0:
        logger.info(
            "Retrieval mode=%s rerank_topk=%d world_size=%d batch_size=%d dtype=%s",
            retrieval_mode,
            rerank_topk,
            ctx.world_size,
            args.batch_size,
            args.dtype,
        )
    selected = {name.strip() for name in args.models.split(",")} if args.models else None
    model_cfgs = [m for m in cfg.get("models", []) if selected is None or str(m.get("name")) in selected]
    datasets = _load_datasets(Path(args.dataset_root), [x.strip() for x in args.datasets.split(",") if x.strip()], args.split, args.limit_images)

    rows: list[dict[str, Any]] = []
    try:
        for model_cfg in model_cfgs:
            name = str(model_cfg["name"])
            logger.info("Evaluating retrieval model %s rank=%d/%d", name, ctx.rank, ctx.world_size)
            result = load_model(model_cfg, Path(args.model_root), ctx.device, args.dtype, logger=logger)
            if result.status != "ok" or result.wrapper is None:
                if ctx.is_rank0:
                    rows.append({"model": name, "status": result.status, "reason": result.error or "load failed"})
                continue
            wrapper = result.wrapper
            for dataset in datasets:
                logger.info("Scoring %s on %s/%s: %d images, %d captions", name, dataset.name, dataset.split, len(dataset.image_paths), len(dataset.captions))
                if retrieval_mode == "spvd_full":
                    if not getattr(wrapper, "supports_text_conditioned_pair_cache", False):
                        if ctx.is_rank0:
                            rows.append({
                                "model": name,
                                "dataset": dataset.name,
                                "split": dataset.split,
                                "status": "unsupported",
                                "reason": "model does not support full SPVD text-conditioned retrieval",
                            })
                        _barrier(ctx)
                        continue
                    similarity = score_soft_cue_conditioned_dataset(wrapper, dataset, args.batch_size, logger, ctx)
                    protocol = "full_exact_spvd_conditioned"
                else:
                    image_features, text_features = encode_dataset_distributed(wrapper, dataset, args.batch_size, logger, ctx)
                    if ctx.is_rank0:
                        assert image_features is not None and text_features is not None
                        similarity = image_features @ text_features.t()
                    else:
                        similarity = None
                    if retrieval_mode == "rerank_topk":
                        similarity = rerank_topk_spvd(wrapper, dataset, similarity if similarity is not None else torch.empty(0), rerank_topk, args.batch_size, logger, ctx)
                        protocol = "global_topk_exact_spvd_rerank" if getattr(wrapper, "supports_text_conditioned_pair_cache", False) else "dual_encoder_global"
                    else:
                        protocol = "dual_encoder_global"
                if ctx.is_rank0 and similarity is not None:
                    metrics = compute_retrieval_metrics_from_similarity(similarity, dataset.caption_image_indices)
                    row = {
                        "model": name,
                        "dataset": dataset.name,
                        "split": dataset.split,
                        "status": "ok",
                        "retrieval_protocol": protocol,
                        "retrieval_mode": retrieval_mode,
                        "rerank_topk": rerank_topk if retrieval_mode == "rerank_topk" else None,
                        "num_images": len(dataset.image_paths),
                        "num_captions": len(dataset.captions),
                        **metrics,
                    }
                    rows.append(row)
                    logger.info("%s/%s metrics: %s", name, dataset.name, metrics)
                _barrier(ctx)
            wrapper.unload()

        if ctx.is_rank0:
            (output_dir / "retrieval_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            if rows:
                fieldnames = sorted({key for row in rows for key in row.keys()})
                with (output_dir / "retrieval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info("Wrote retrieval metrics to %s", output_dir)
    finally:
        _cleanup_dist(ctx)


if __name__ == "__main__":
    main()
