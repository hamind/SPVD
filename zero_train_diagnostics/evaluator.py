from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .benchmarks import load_benchmarks
from .metrics import (
    summarize_aro,
    summarize_margins,
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
) -> pd.DataFrame:
    samples = list(benchmark.samples)
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


def evaluate_winoground(
    model: FrozenVLMWrapper,
    benchmark: BenchmarkLoadResult,
    run_dir: Path,
    logger: logging.Logger,
    skipped_samples: list[dict[str, Any]],
) -> pd.DataFrame:
    image_cache = ImageCache()
    rows: list[dict[str, Any]] = []
    for sample in benchmark.samples:
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
                    "benchmark": "winoground",
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
                    "benchmark": "winoground",
                    "source_file": str(sample.source_file) if sample.source_file else "",
                    "sample_id": sample.sample_id,
                    "reason": f"evaluation failed: {exc!r}",
                }
            )
            logger.exception("Winoground sample failed: %s", sample.sample_id)
    raw_path = run_dir / "raw_results" / f"{model.name}_winoground.csv"
    write_dataframe(pd.DataFrame(rows), raw_path)
    logger.info("Wrote %d raw rows to %s", len(rows), raw_path)
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
    margins_summary = summarize_margins(pairwise_df)
    ssr_summary = summarize_ssr(pairwise_df)
    all_summary = summary_all_models(model_info, aro_summary, sugar_summary, winoground_summary, margins_summary, ssr_summary)
    outputs = {
        "summary_all_models": all_summary,
        "summary_aro_by_category": aro_summary,
        "summary_sugarcrepe_by_category": sugar_summary,
        "summary_winoground": winoground_summary,
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
    dataset_root = Path(args.dataset_root or config.get("dataset_root") or "/vepfs/dataset")
    model_root = Path(args.model_root or config.get("model_root") or "/vepfs/model")
    output_base = Path(args.output_dir or config.get("output_dir") or "outputs/zero_train_diagnostics")
    seed = int(args.seed if args.seed is not None else config.get("random_seed", 42))
    set_seed(seed)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_base / timestamp)
    ensure_dir(run_dir / "raw_results")
    ensure_dir(run_dir / "summaries")
    ensure_dir(run_dir / "figures")
    logger = setup_logging(run_dir / "logs" / "eval.log", verbose=True)
    logger.info("Starting zero-training diagnostics in %s", run_dir)
    expanded_config = dict(config)
    expanded_config.update(
        {
            "dataset_root": str(dataset_root),
            "model_root": str(model_root),
            "output_dir": str(output_base),
            "device": args.device or config.get("device", "cuda"),
            "batch_size": int(args.batch_size or config.get("batch_size", 64)),
            "num_workers": int(args.num_workers or config.get("num_workers", 0)),
            "random_seed": seed,
        }
    )
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
    }
    expected_env = str(config.get("conda_env") or "openclip")
    if environment.get("conda_environment") != expected_env:
        warning = f"Expected conda env {expected_env}, got {environment.get('conda_environment')}"
        logger.warning(warning)
        manifest["warnings"].append(warning)
    for key, value in environment.items():
        logger.info("ENV %s=%s", key, value)
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
    write_json(manifest, run_dir / "manifest.json")

    models_cfg = list(config.get("models") or [])
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        models_cfg = [cfg for cfg in models_cfg if cfg.get("name") in wanted]
    device = args.device or config.get("device", "cuda")
    dtype = args.dtype or config.get("dtype", "fp32")
    batch_size = int(args.batch_size or config.get("batch_size", 64))
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
            write_json(manifest, run_dir / "manifest.json")
            continue
        model_info[model_name] = {"model_type": loaded.model_type}
        try:
            for benchmark_name, benchmark in benchmarks.items():
                if benchmark.status != "ok":
                    logger.warning("Skipping benchmark %s for %s due to benchmark status=%s", benchmark_name, model_name, benchmark.status)
                    continue
                logger.info("Scoring %s on %s (%d samples)", model_name, benchmark_name, len(benchmark.samples))
                if benchmark_name in {"aro", "sugarcrepe"}:
                    df = evaluate_pairwise_benchmark(
                        loaded.wrapper,
                        benchmark,
                        run_dir,
                        batch_size,
                        compute_random_negative,
                        seed,
                        logger,
                        skipped_samples,
                    )
                    if not df.empty:
                        pairwise_frames.append(df)
                elif benchmark_name == "winoground":
                    df = evaluate_winoground(loaded.wrapper, benchmark, run_dir, logger, skipped_samples)
                    if not df.empty:
                        winoground_frames.append(df)
                write_json(manifest, run_dir / "manifest.json")
        except Exception as exc:
            logger.exception("Model evaluation failed for %s", model_name)
            manifest["error_messages"].append({model_name: repr(exc)})
        finally:
            loaded.wrapper.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_json(manifest, run_dir / "manifest.json")

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
    return run_dir
