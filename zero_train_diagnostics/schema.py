from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PairwiseSample:
    benchmark: str
    sample_id: str
    image_path: Path
    positive_caption: str
    negative_captions: list[str]
    category: str
    subcategory: str | None = None
    source_file: Path | None = None
    image_id: str | None = None
    crop_box: tuple[int, int, int, int] | None = None
    negative_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WinogroundSample:
    benchmark: str
    sample_id: str
    image_0_path: Path
    image_1_path: Path
    caption_0: str
    caption_1: str
    category: str | None = None
    source_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkLoadResult:
    name: str
    samples: list[PairwiseSample] | list[WinogroundSample]
    status: str
    data_files: list[Path] = field(default_factory=list)
    candidates: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_samples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelLoadResult:
    name: str
    model_type: str
    status: str
    wrapper: Any | None = None
    checkpoint_path: Path | None = None
    local_dir: Path | None = None
    parameter_count: int | None = None
    device: str | None = None
    dtype: str | None = None
    dummy_forward_success: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
