from __future__ import annotations

import logging
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .schema import BenchmarkLoadResult, PairwiseSample, WinogroundSample
from .utils import balanced_limit, clean_text, list_candidate_files, load_records, resolve_existing_path


def _logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger(__name__)


def _skip(benchmark: str, source_file: Path | None, sample_id: str, reason: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "source_file": str(source_file) if source_file else "",
        "sample_id": sample_id,
        "reason": reason,
    }


def _word_order_negatives(caption: str, seed: int, max_words: int) -> tuple[list[str], list[str]]:
    words = clean_text(caption, max_words=max_words).split()
    if len(words) < 2:
        return [], []
    rng = random.Random(seed)
    negatives: list[tuple[str, str]] = []
    reverse = " ".join(reversed(words))
    if reverse != caption:
        negatives.append(("reverse_words", reverse))
    adjacent = words[:]
    for i in range(0, len(adjacent) - 1, 2):
        adjacent[i], adjacent[i + 1] = adjacent[i + 1], adjacent[i]
    adjacent_text = " ".join(adjacent)
    if adjacent_text != caption:
        negatives.append(("swap_adjacent_tokens", adjacent_text))
    trigrams = [words[i : i + 3] for i in range(0, len(words), 3)]
    shuffled_trigrams = trigrams[:]
    rng.shuffle(shuffled_trigrams)
    trigram_text = " ".join(token for tri in shuffled_trigrams for token in tri)
    if trigram_text != caption:
        negatives.append(("shuffle_trigrams_seeded", trigram_text))
    all_words = words[:]
    rng.shuffle(all_words)
    all_words_text = " ".join(all_words)
    if all_words_text != caption:
        negatives.append(("shuffle_all_words_seeded", all_words_text))
    deduped: list[tuple[str, str]] = []
    seen = {caption}
    for name, text in negatives:
        if text and text not in seen:
            seen.add(text)
            deduped.append((name, text))
    return [text for _, text in deduped], [name for name, _ in deduped]


def _find_one_by_name(candidates: list[Path], filename: str) -> Path | None:
    matches = [p for p in candidates if p.name == filename]
    return sorted(matches)[0] if matches else None


def _prefer_split(files: list[Path], stem_prefix: str, preferred_splits: list[str]) -> Path | None:
    for split in preferred_splits:
        name = f"{stem_prefix}_{split}.json"
        match = _find_one_by_name(files, name)
        if match:
            return match
    matches = [p for p in files if p.name.startswith(stem_prefix) and p.suffix == ".json"]
    return sorted(matches)[0] if matches else None


def load_aro(dataset_root: Path, limit: int | None, seed: int, config: dict[str, Any], logger: logging.Logger | None = None) -> BenchmarkLoadResult:
    log = _logger(logger)
    root = Path(config.get("root") or dataset_root / "aro")
    candidates = list_candidate_files(root)
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []
    samples: list[PairwiseSample] = []
    data_files: list[Path] = []
    if not root.exists():
        warning = f"ARO root not found: {root}"
        log.warning(warning)
        return BenchmarkLoadResult("aro", [], "missing", candidates, candidates, [warning], skipped)

    attr_file = _find_one_by_name(candidates, "visual_genome_attribution.json")
    rel_file = _find_one_by_name(candidates, "visual_genome_relation.json")
    for file_path, category in [(attr_file, "attr"), (rel_file, "rel")]:
        if not file_path:
            warning = f"ARO {category} file not found under {root}"
            warnings.append(warning)
            log.warning(warning)
            continue
        data_files.append(file_path)
        rows = load_records(file_path)
        image_dirs = [
            root / "vg_relation" / "images",
            root / "vg_attribution" / "images",
            file_path.parent / "images",
            file_path.parent,
            dataset_root / "vg" / "VG_100K",
            dataset_root / "vg" / "VG_100K_2",
            dataset_root / "visual_genome" / "VG_100K",
            dataset_root / "visual_genome" / "VG_100K_2",
            Path("/vepfs/dataset/aro/ready/vg/images"),
        ]
        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or row.get("image_id") or f"{category}_{idx}")
            image_ref = row.get("image_path") or row.get("image") or row.get("filename")
            pos = row.get("true_caption") or row.get("positive_caption") or row.get("caption")
            neg = row.get("false_caption") or row.get("negative_caption")
            if not image_ref or not pos or not neg:
                skipped.append(_skip("aro", file_path, sample_id, "missing image or caption fields"))
                continue
            image_path = resolve_existing_path(str(image_ref), image_dirs)
            if image_path is None:
                skipped.append(_skip("aro", file_path, sample_id, f"image not found: {image_ref}"))
                continue
            crop_box = None
            if all(k in row for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
                x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
                crop_box = (x, y, x + max(1, w), y + max(1, h))
            subcategory = None
            if category == "attr" and row.get("attributes"):
                attrs = row.get("attributes")
                if isinstance(attrs, list):
                    subcategory = "_".join(str(v) for v in attrs)
            elif category == "rel":
                subcategory = str(row.get("relation_name") or "")
            samples.append(
                PairwiseSample(
                    benchmark="aro",
                    sample_id=f"{category}_{sample_id}_{idx}",
                    image_path=image_path,
                    positive_caption=clean_text(pos),
                    negative_captions=[clean_text(neg)],
                    category=category,
                    subcategory=subcategory or None,
                    source_file=file_path,
                    image_id=str(row.get("image_id") or ""),
                    crop_box=crop_box,
                    negative_types=["hard_negative"],
                    metadata={k: v for k, v in row.items() if k not in {"true_caption", "false_caption"}},
                )
            )

    split_order = list(config.get("order_splits") or ["test", "val"])
    max_words = int(config.get("order_max_words") or 30)
    order_sources = [
        ("coco_order", _prefer_split(candidates, "coco_karpathy", split_order)),
        ("flickr_order", _prefer_split(candidates, "flickr30k", split_order)),
    ]
    for subcategory, file_path in order_sources:
        if not file_path:
            warning = f"ARO {subcategory} file not found under {root}"
            warnings.append(warning)
            log.warning(warning)
            continue
        data_files.append(file_path)
        rows = load_records(file_path)
        image_dirs = [
            file_path.parent,
            file_path.parent / "val2014",
            file_path.parent / "test2014",
            file_path.parent / "flickr30k-images",
            dataset_root / "coco" / "images",
            dataset_root / "coco" / "images" / "val2014",
            dataset_root / "coco" / "images" / "test2014",
            dataset_root / "Flickr30k" / "images",
            dataset_root / "flickr30k" / "images",
        ]
        for idx, row in enumerate(rows):
            captions = row.get("caption") or row.get("captions")
            if isinstance(captions, str):
                captions = [captions]
            image_ref = row.get("image") or row.get("image_path") or row.get("filename")
            if not image_ref or not captions:
                skipped.append(_skip("aro", file_path, f"{subcategory}_{idx}", "missing image or caption fields"))
                continue
            image_path = resolve_existing_path(str(image_ref), image_dirs)
            if image_path is None:
                skipped.append(_skip("aro", file_path, f"{subcategory}_{idx}", f"image not found: {image_ref}"))
                continue
            for cap_idx, caption in enumerate(captions):
                pos = clean_text(caption, max_words=max_words)
                negs, neg_types = _word_order_negatives(pos, seed + idx * 100 + cap_idx, max_words=max_words)
                if not negs:
                    skipped.append(_skip("aro", file_path, f"{subcategory}_{idx}_{cap_idx}", "could not generate order negative"))
                    continue
                samples.append(
                    PairwiseSample(
                        benchmark="aro",
                        sample_id=f"{subcategory}_{idx}_{cap_idx}",
                        image_path=image_path,
                        positive_caption=pos,
                        negative_captions=negs,
                        category="order",
                        subcategory=subcategory,
                        source_file=file_path,
                        image_id=str(row.get("image_id") or image_ref),
                        negative_types=neg_types,
                        metadata={"order_negative_generation": "deterministic_token_perturbations"},
                    )
                )

    limited = balanced_limit(samples, limit, key_fn=lambda item: item.category)
    log.info("ARO candidates: %s", [str(p) for p in candidates])
    log.info("ARO selected data files: %s", [str(p) for p in data_files])
    log.info("ARO loaded %d samples (%d after limit)", len(samples), len(limited))
    status = "ok" if limited else "empty"
    return BenchmarkLoadResult(
        "aro",
        limited,
        status,
        data_files=data_files,
        candidates=candidates,
        warnings=warnings,
        skipped_samples=skipped,
        metadata={"loaded_sample_count": len(samples), "order_negative_generation": "deterministic_token_perturbations"},
    )


def load_sugarcrepe(dataset_root: Path, limit: int | None, config: dict[str, Any], logger: logging.Logger | None = None) -> BenchmarkLoadResult:
    log = _logger(logger)
    root = Path(config.get("root") or dataset_root / "sugarcrepe")
    candidates = list_candidate_files(root)
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []
    samples: list[PairwiseSample] = []
    if not root.exists():
        warning = f"SugarCrepe root not found: {root}"
        log.warning(warning)
        return BenchmarkLoadResult("sugarcrepe", [], "missing", candidates, candidates, [warning], skipped)
    data_files = [p for p in candidates if p.name in {
        "add_att.json",
        "add_obj.json",
        "replace_att.json",
        "replace_obj.json",
        "replace_rel.json",
        "swap_att.json",
        "swap_obj.json",
    }]
    if not data_files:
        data_files = [p for p in candidates if p.suffix == ".json"]
    image_dirs = [
        root / "coco_val2017_images",
        dataset_root / "coco" / "images" / "val2017",
        root / "ready" / "coco2017" / "val2017",
        root / "coco2017" / "val2017",
        Path("/vepfs/dataset/sugarcrepe/ready/coco2017/val2017"),
        root,
    ]
    for file_path in sorted(data_files):
        category = file_path.stem
        match = re.match(r"(?P<kind>[^_]+)_(?P<sub>.+)", category)
        subcategory = match.group("sub") if match else None
        rows = load_records(file_path)
        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or row.get("id") or idx)
            image_ref = row.get("filename") or row.get("image") or row.get("image_path")
            pos = row.get("caption") or row.get("positive_caption") or row.get("true_caption")
            neg = row.get("negative_caption") or row.get("false_caption")
            if not image_ref or not pos or not neg:
                skipped.append(_skip("sugarcrepe", file_path, sample_id, "missing image or caption fields"))
                continue
            image_path = resolve_existing_path(str(image_ref), image_dirs)
            if image_path is None:
                skipped.append(_skip("sugarcrepe", file_path, sample_id, f"image not found: {image_ref}"))
                continue
            samples.append(
                PairwiseSample(
                    benchmark="sugarcrepe",
                    sample_id=f"{category}_{sample_id}",
                    image_path=image_path,
                    positive_caption=clean_text(pos),
                    negative_captions=[clean_text(neg)],
                    category=category,
                    subcategory=subcategory,
                    source_file=file_path,
                    image_id=str(image_ref),
                    negative_types=["hard_negative"],
                    metadata={k: v for k, v in row.items() if k not in {"caption", "negative_caption"}},
                )
            )
    limited = balanced_limit(samples, limit, key_fn=lambda item: item.category)
    log.info("SugarCrepe candidates: %s", [str(p) for p in candidates])
    log.info("SugarCrepe selected data files: %s", [str(p) for p in data_files])
    log.info("SugarCrepe loaded %d samples (%d after limit)", len(samples), len(limited))
    return BenchmarkLoadResult(
        "sugarcrepe",
        limited,
        "ok" if limited else "empty",
        data_files=data_files,
        candidates=candidates,
        warnings=warnings,
        skipped_samples=skipped,
        metadata={"loaded_sample_count": len(samples)},
    )


def load_sugarcrepe_pp(dataset_root: Path, limit: int | None, config: dict[str, Any], logger: logging.Logger | None = None) -> BenchmarkLoadResult:
    log = _logger(logger)
    root = Path(config.get("root") or dataset_root / "sugarcrepe_pp")
    candidates = list_candidate_files(root)
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []
    samples: list[PairwiseSample] = []
    if not root.exists():
        warning = f"SugarCrepe++ root not found: {root}"
        log.warning(warning)
        return BenchmarkLoadResult("sugarcrepe_pp", [], "missing", candidates, candidates, [warning], skipped)
    data_files = [p for p in candidates if p.name in {
        "replace_att.json",
        "replace_obj.json",
        "replace_rel.json",
        "swap_att.json",
        "swap_obj.json",
    }]
    if not data_files:
        data_files = [p for p in candidates if p.suffix == ".json"]
    image_dirs = [
        root / "coco_val2017_images",
        dataset_root / "coco" / "images" / "val2017",
        root / "ready" / "coco2017" / "val2017",
        root / "coco2017" / "val2017",
        Path("/vepfs/dataset/sugarcrepe/ready/coco2017/val2017"),
        root,
    ]
    for file_path in sorted(data_files):
        category = file_path.stem
        match = re.match(r"(?P<kind>[^_]+)_(?P<sub>.+)", category)
        subcategory = match.group("sub") if match else None
        rows = load_records(file_path)
        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or row.get("id") or idx)
            image_ref = row.get("filename") or row.get("image") or row.get("image_path")
            pos_1 = row.get("caption") or row.get("positive_caption") or row.get("true_caption")
            pos_2 = row.get("caption2") or row.get("positive_caption_2")
            neg = row.get("negative_caption") or row.get("false_caption")
            if not image_ref or not pos_1 or not neg:
                skipped.append(_skip("sugarcrepe_pp", file_path, sample_id, "missing image or caption fields"))
                continue
            image_path = resolve_existing_path(str(image_ref), image_dirs)
            if image_path is None:
                skipped.append(_skip("sugarcrepe_pp", file_path, sample_id, f"image not found: {image_ref}"))
                continue
            positives = [("caption", pos_1)]
            if pos_2:
                positives.append(("caption2", pos_2))
            for variant, pos in positives:
                samples.append(
                    PairwiseSample(
                        benchmark="sugarcrepe_pp",
                        sample_id=f"{category}_{sample_id}_{variant}",
                        image_path=image_path,
                        positive_caption=clean_text(pos),
                        negative_captions=[clean_text(neg)],
                        category=category,
                        subcategory=subcategory,
                        source_file=file_path,
                        image_id=str(image_ref),
                        negative_types=["hard_negative"],
                        metadata={
                            **{k: v for k, v in row.items() if k not in {"caption", "caption2", "negative_caption"}},
                            "positive_variant": variant,
                        },
                    )
                )
    limited = balanced_limit(samples, limit, key_fn=lambda item: item.category)
    log.info("SugarCrepe++ candidates: %s", [str(p) for p in candidates])
    log.info("SugarCrepe++ selected data files: %s", [str(p) for p in data_files])
    log.info("SugarCrepe++ loaded %d pairwise samples (%d after limit)", len(samples), len(limited))
    return BenchmarkLoadResult(
        "sugarcrepe_pp",
        limited,
        "ok" if limited else "empty",
        data_files=data_files,
        candidates=candidates,
        warnings=warnings,
        skipped_samples=skipped,
        metadata={"loaded_sample_count": len(samples), "positive_variants": ["caption", "caption2"]},
    )


def _image_bytes_from_cell(cell: Any) -> bytes | None:
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        value = cell.get("bytes") or cell.get("data")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    return None


def _materialize_bivlc_image(cell: Any, path: Path) -> Path | None:
    image_bytes = _image_bytes_from_cell(cell)
    if image_bytes is None:
        return None
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.convert("RGB").save(path, format="JPEG", quality=95)
    return path


def load_bivlc(dataset_root: Path, limit: int | None, config: dict[str, Any], logger: logging.Logger | None = None) -> BenchmarkLoadResult:
    log = _logger(logger)
    root = Path(config.get("root") or dataset_root / "bivlc")
    candidates = list_candidate_files(root, exts=(".json", ".jsonl", ".csv", ".tsv", ".parquet"))
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []
    samples: list[WinogroundSample] = []
    if not root.exists():
        warning = f"BiVLC root not found: {root}"
        log.warning(warning)
        return BenchmarkLoadResult("bivlc", [], "missing", candidates, candidates, [warning], skipped)
    data_files = [p for p in candidates if p.suffix == ".parquet"]
    if not data_files:
        warning = f"BiVLC parquet files not found under {root}"
        log.warning(warning)
        return BenchmarkLoadResult("bivlc", [], "missing", [], candidates, [warning], skipped)
    image_root = Path(config.get("image_root") or root / "images_or_links" / "extracted")
    for file_path in sorted(data_files):
        frame = pd.read_parquet(file_path)
        for idx, row in frame.iterrows():
            sample_id = f"{file_path.stem}_{idx}"
            caption = row.get("caption")
            negative_caption = row.get("negative_caption")
            if not caption or not negative_caption:
                skipped.append(_skip("bivlc", file_path, sample_id, "missing caption fields"))
                continue
            pos_path = image_root / file_path.stem / f"{idx:06d}_pos.jpg"
            neg_path = image_root / file_path.stem / f"{idx:06d}_neg.jpg"
            try:
                pos_image = _materialize_bivlc_image(row.get("image"), pos_path)
                neg_image = _materialize_bivlc_image(row.get("negative_image"), neg_path)
            except Exception as exc:
                skipped.append(_skip("bivlc", file_path, sample_id, f"image materialization failed: {exc!r}"))
                continue
            if pos_image is None or neg_image is None:
                skipped.append(_skip("bivlc", file_path, sample_id, "missing image bytes"))
                continue
            category = str(row.get("type") or "")
            subcategory = str(row.get("subtype") or "")
            samples.append(
                WinogroundSample(
                    benchmark="bivlc",
                    sample_id=sample_id,
                    image_0_path=pos_image,
                    image_1_path=neg_image,
                    caption_0=clean_text(caption),
                    caption_1=clean_text(negative_caption),
                    category=category,
                    source_file=file_path,
                    metadata={"type": category, "subtype": subcategory},
                )
            )
    limited = balanced_limit(samples, limit, key_fn=lambda item: str(item.category))
    log.info("BiVLC candidates: %s", [str(p) for p in candidates])
    log.info("BiVLC selected data files: %s", [str(p) for p in data_files])
    log.info("BiVLC loaded %d 2x2 samples (%d after limit)", len(samples), len(limited))
    return BenchmarkLoadResult(
        "bivlc",
        limited,
        "ok" if limited else "empty",
        data_files=data_files,
        candidates=candidates,
        warnings=warnings,
        skipped_samples=skipped,
        metadata={"loaded_sample_count": len(samples), "retrieval_instances_per_sample": 4, "image_root": str(image_root)},
    )


def load_winoground(dataset_root: Path, limit: int | None, config: dict[str, Any], logger: logging.Logger | None = None) -> BenchmarkLoadResult:
    log = _logger(logger)
    root = Path(config.get("root") or dataset_root / "winoground")
    candidates = list_candidate_files(root)
    warnings: list[str] = []
    skipped: list[dict[str, Any]] = []
    samples: list[WinogroundSample] = []
    if not root.exists():
        warning = f"Winoground root not found: {root}"
        log.warning(warning)
        return BenchmarkLoadResult("winoground", [], "missing", candidates, candidates, [warning], skipped)
    preferred = None
    for candidate in candidates:
        if candidate.name in {"annotations.jsonl", "examples.jsonl"} and "facebook-winoground" in str(candidate):
            preferred = candidate
            break
    if preferred is None:
        for candidate in candidates:
            if candidate.name in {"annotations.jsonl", "examples.jsonl"}:
                preferred = candidate
                break
    if preferred is None:
        warning = f"Winoground annotations.jsonl/examples.jsonl not found under {root}"
        log.warning(warning)
        warnings.append(warning)
        return BenchmarkLoadResult("winoground", [], "missing", [], candidates, warnings, skipped)
    rows = load_records(preferred)
    if rows and {"caption_0", "caption_1", "image_0_path", "image_1_path"}.issubset(rows[0].keys()):
        for idx, row in enumerate(rows):
            sample_id = str(row.get("id") or row.get("sample_id") or idx)
            image0 = resolve_existing_path(str(row.get("image_0_path")), [preferred.parent])
            image1 = resolve_existing_path(str(row.get("image_1_path")), [preferred.parent])
            if image0 is None or image1 is None:
                skipped.append(_skip("winoground", preferred, sample_id, "image_0 or image_1 not found"))
                continue
            samples.append(
                WinogroundSample(
                    benchmark="winoground",
                    sample_id=sample_id,
                    image_0_path=image0,
                    image_1_path=image1,
                    caption_0=clean_text(row["caption_0"]),
                    caption_1=clean_text(row["caption_1"]),
                    category=str(row.get("collapsed_tag") or row.get("tag") or ""),
                    source_file=preferred,
                    metadata={k: v for k, v in row.items() if k not in {"caption_0", "caption_1", "image_0_path", "image_1_path"}},
                )
            )
    else:
        if len(rows) % 2 != 0:
            warnings.append("Flattened Winoground annotations have an odd row count; final row will be skipped.")
        for idx in range(0, len(rows) - 1, 2):
            row0, row1 = rows[idx], rows[idx + 1]
            sample_id = str(idx // 2)
            image0 = resolve_existing_path(str(row0.get("image")), [preferred.parent])
            image1 = resolve_existing_path(str(row1.get("image")), [preferred.parent])
            if image0 is None or image1 is None:
                skipped.append(_skip("winoground", preferred, sample_id, "flattened image pair not found"))
                continue
            samples.append(
                WinogroundSample(
                    benchmark="winoground",
                    sample_id=sample_id,
                    image_0_path=image0,
                    image_1_path=image1,
                    caption_0=clean_text(row0.get("true_caption")),
                    caption_1=clean_text(row0.get("false_caption")),
                    source_file=preferred,
                    metadata={"flattened_source": True},
                )
            )
    limited = samples[:limit] if limit and limit > 0 else samples
    log.info("Winoground candidates: %s", [str(p) for p in candidates])
    log.info("Winoground selected data file: %s", preferred)
    log.info("Winoground loaded %d samples (%d after limit)", len(samples), len(limited))
    return BenchmarkLoadResult(
        "winoground",
        limited,
        "ok" if limited else "empty",
        data_files=[preferred],
        candidates=candidates,
        warnings=warnings,
        skipped_samples=skipped,
        metadata={"loaded_sample_count": len(samples)},
    )


def load_benchmarks(dataset_root: Path, limit: int | None, config: dict[str, Any], seed: int, logger: logging.Logger | None = None) -> dict[str, BenchmarkLoadResult]:
    benchmark_cfg = config.get("benchmarks") or {}
    enabled = benchmark_cfg.get("enabled") or ["aro", "sugarcrepe", "winoground"]
    results: dict[str, BenchmarkLoadResult] = {}
    if "aro" in enabled:
        results["aro"] = load_aro(dataset_root, limit, seed, benchmark_cfg.get("aro") or {}, logger)
    if "sugarcrepe" in enabled:
        results["sugarcrepe"] = load_sugarcrepe(dataset_root, limit, benchmark_cfg.get("sugarcrepe") or {}, logger)
    if "sugarcrepe_pp" in enabled:
        results["sugarcrepe_pp"] = load_sugarcrepe_pp(dataset_root, limit, benchmark_cfg.get("sugarcrepe_pp") or {}, logger)
    if "bivlc" in enabled:
        results["bivlc"] = load_bivlc(dataset_root, limit, benchmark_cfg.get("bivlc") or {}, logger)
    if "winoground" in enabled:
        results["winoground"] = load_winoground(dataset_root, limit, benchmark_cfg.get("winoground") or {}, logger)
    return results
