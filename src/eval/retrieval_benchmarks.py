"""COCO/Flickr30k retrieval benchmarks for frozen VLM wrappers."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from zero_train_diagnostics.models import load_model


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

    return torch.cat(image_features, dim=0).float(), torch.cat(text_features, dim=0).float()


@torch.no_grad()
def score_soft_cue_conditioned_dataset(wrapper: Any, dataset: RetrievalDataset, batch_size: int, logger: logging.Logger) -> torch.Tensor:
    image_cache_batch = max(1, min(batch_size, 128))
    text_cache_batch = max(1, batch_size)
    image_score_batch = max(1, min(batch_size // 4 if batch_size >= 4 else batch_size, 64))
    text_score_batch = max(1, batch_size)

    logger.info(
        "%s uses SPVD soft-cue-conditioned retrieval: image_cache_bs=%d text_cache_bs=%d image_score_bs=%d text_score_bs=%d",
        wrapper.name,
        image_cache_batch,
        text_cache_batch,
        image_score_batch,
        text_score_batch,
    )

    image_token_chunks: list[torch.Tensor] = []
    for start in range(0, len(dataset.image_paths), image_cache_batch):
        end = min(start + image_cache_batch, len(dataset.image_paths))
        images = _open_images(dataset.image_paths[start:end])
        image_token_chunks.append(wrapper.encode_conditioned_image_tokens(images))
        if start == 0 or end == len(dataset.image_paths) or end % max(image_cache_batch * 20, 1) == 0:
            logger.info("%s conditioned image tokens: %d/%d encoded", dataset.name, end, len(dataset.image_paths))
    image_tokens = torch.cat(image_token_chunks, dim=0)

    soft_cue_chunks: list[torch.Tensor] = []
    text_feature_chunks: list[torch.Tensor] = []
    for start in range(0, len(dataset.captions), text_cache_batch):
        end = min(start + text_cache_batch, len(dataset.captions))
        text_cache = wrapper.encode_conditioned_texts(dataset.captions[start:end])
        soft_cue_chunks.append(text_cache["soft_cues"])
        text_feature_chunks.append(text_cache["text_features"])
        if start == 0 or end == len(dataset.captions) or end % max(text_cache_batch * 20, 1) == 0:
            logger.info("%s conditioned text cues: %d/%d encoded", dataset.name, end, len(dataset.captions))
    soft_cues = torch.cat(soft_cue_chunks, dim=0)
    text_features = torch.cat(text_feature_chunks, dim=0)

    similarity = torch.empty((len(dataset.image_paths), len(dataset.captions)), dtype=torch.float32)
    for image_start in range(0, len(dataset.image_paths), image_score_batch):
        image_end = min(image_start + image_score_batch, len(dataset.image_paths))
        for text_start in range(0, len(dataset.captions), text_score_batch):
            text_end = min(text_start + text_score_batch, len(dataset.captions))
            scores = wrapper.score_conditioned_retrieval_chunk(
                image_tokens[image_start:image_end],
                soft_cues[text_start:text_end],
                text_features[text_start:text_end],
            )
            similarity[image_start:image_end, text_start:text_end] = scores
        if image_start == 0 or image_end == len(dataset.image_paths) or image_end % max(image_score_batch * 20, 1) == 0:
            logger.info("%s conditioned retrieval rows: %d/%d scored", dataset.name, image_end, len(dataset.image_paths))
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
    parser.add_argument("--dataset-root", default="/vepfs/dataset/benchmarks")
    parser.add_argument("--model-root", default="/vepfs/model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--output-dir", default="outputs/benchmark_retrieval")
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = _setup_logger(output_dir)
    cfg = _read_yaml(Path(args.config))
    selected = {name.strip() for name in args.models.split(",")} if args.models else None
    model_cfgs = [m for m in cfg.get("models", []) if selected is None or str(m.get("name")) in selected]
    datasets = _load_datasets(Path(args.dataset_root), [x.strip() for x in args.datasets.split(",") if x.strip()], args.split, args.limit_images)

    rows: list[dict[str, Any]] = []
    for model_cfg in model_cfgs:
        name = str(model_cfg["name"])
        logger.info("Evaluating retrieval model %s", name)
        result = load_model(model_cfg, Path(args.model_root), args.device, args.dtype, logger=logger)
        if result.status != "ok" or result.wrapper is None:
            rows.append({"model": name, "status": result.status, "reason": result.error or "load failed"})
            continue
        wrapper = result.wrapper
        supports_retrieval = bool(getattr(wrapper, "supports_retrieval_cache", wrapper.supports_feature_cache))
        if not supports_retrieval:
            logger.warning("Skipping retrieval for %s: wrapper does not expose cached image/text features", name)
            rows.append({"model": name, "status": "unsupported", "reason": "retrieval requires cached image/text features"})
            wrapper.unload()
            continue
        for dataset in datasets:
            logger.info("Scoring %s on %s/%s: %d images, %d captions", name, dataset.name, dataset.split, len(dataset.image_paths), len(dataset.captions))
            soft_cue_conditioned = bool(getattr(wrapper, "supports_soft_cue_conditioned_retrieval", False))
            if soft_cue_conditioned:
                similarity = score_soft_cue_conditioned_dataset(wrapper, dataset, args.batch_size, logger)
                metrics = compute_retrieval_metrics_from_similarity(similarity, dataset.caption_image_indices)
                protocol = "soft_cue_conditioned_all_candidates"
            else:
                image_features, text_features = encode_dataset(wrapper, dataset, args.batch_size, logger)
                metrics = compute_retrieval_metrics(image_features, text_features, dataset.caption_image_indices)
                protocol = "dual_encoder"
            row = {
                "model": name,
                "dataset": dataset.name,
                "split": dataset.split,
                "status": "ok",
                "retrieval_protocol": protocol,
                "num_images": len(dataset.image_paths),
                "num_captions": len(dataset.captions),
                **metrics,
            }
            rows.append(row)
            logger.info("%s/%s metrics: %s", name, dataset.name, metrics)
        wrapper.unload()

    (output_dir / "retrieval_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (output_dir / "retrieval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote retrieval metrics to %s", output_dir)


if __name__ == "__main__":
    main()
