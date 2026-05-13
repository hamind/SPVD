from __future__ import annotations

import csv
import json
import tarfile
from argparse import Namespace
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from data import CaptionRelabeler, ManifestImageTextDataset, _wds_caption_source_from_sample, build_dataloader, sample_subcaptions, split_caption


def _write_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(100, 80, 60))
    image.save(path)


def _webdataset_args(shard: Path, train_num_samples: int = 1, filter_relabel_success: bool = False) -> Namespace:
    return Namespace(
        train_data=str(shard),
        val_data=None,
        dataset_type="webdataset",
        train_num_samples=train_num_samples,
        val_num_samples=None,
        batch_size=1,
        workers=0,
        world_size=1,
        seed=0,
        dataset_resampled=False,
        train_data_upsampling_factors=None,
        filter_relabel_success=filter_relabel_success,
        relabel_success_key="longSV",
        strict_caption_match=True,
        num_sampled_captions=4,
        max_merged_num=3,
    )


def test_toy_manifest_dataset_loads(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "sample.jpg")
    manifest = tmp_path / "toy.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerow({"image": "sample.jpg", "caption": "a small image"})

    dataset = ManifestImageTextDataset(
        manifest_file=manifest,
        image_root=image_dir,
        image_key="image",
        caption_key="caption",
        transform=transforms.ToTensor(),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["caption"] == "a small image"


def test_mock_webdataset_shard_loads(tmp_path: Path) -> None:
    image_path = tmp_path / "000000001.jpg"
    text_path = tmp_path / "000000001.txt"
    _write_image(image_path)
    text_path.write_text("a shard caption", encoding="utf-8")
    shard = tmp_path / "toy-0000.tar"
    with tarfile.open(shard, "w") as tar:
        tar.add(image_path, arcname=image_path.name)
        tar.add(text_path, arcname=text_path.name)

    tokenizer = lambda texts: torch.ones((len(texts), 8), dtype=torch.long)
    loader = build_dataloader(_webdataset_args(shard), transforms.ToTensor(), tokenizer, is_train=True)
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 3, 32, 32)
    assert batch["text"].shape == (1, 4, 8)
    assert len(batch["caption"][0]) == 4


def test_caption_relabeler_uses_sqlite_index(tmp_path: Path) -> None:
    relabel_csv = tmp_path / "long_caption.csv"
    with relabel_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Image Path", "longSV_captions", "raw_caption"])
        writer.writeheader()
        writer.writerow(
            {
                "Image Path": "https://example.test/image.jpg",
                "longSV_captions": "a long relabeled caption",
                "raw_caption": "a raw shard caption",
            }
        )
    relabeler = CaptionRelabeler(
        caption_file=relabel_csv,
        caption_key="longSV_captions",
        index_path=tmp_path / "long_caption.sqlite",
    )
    assert relabeler.lookup("https://example.test/image.jpg") == "a long relabeled caption"


def test_webdataset_filters_failed_long_caption_samples(tmp_path: Path) -> None:
    good_image = tmp_path / "000000001.jpg"
    good_text = tmp_path / "000000001.txt"
    good_json = tmp_path / "000000001.json"
    bad_image = tmp_path / "000000002.jpg"
    bad_text = tmp_path / "000000002.txt"
    bad_json = tmp_path / "000000002.json"
    _write_image(good_image)
    _write_image(bad_image)
    good_text.write_text("a long successful caption", encoding="utf-8")
    bad_text.write_text("a raw failed caption", encoding="utf-8")
    good_json.write_text(
        json.dumps(
            {
                "key": "000000001",
                "default_caption_key": "longSV",
                "caption": "a long successful caption",
                "captions": {"raw": "a raw caption", "longSV": "a long successful caption"},
            }
        ),
        encoding="utf-8",
    )
    bad_json.write_text(json.dumps({"key": "000000002", "caption": "a raw failed caption"}), encoding="utf-8")
    shard = tmp_path / "toy-0000.tar"
    with tarfile.open(shard, "w") as tar:
        for path in (good_image, good_json, good_text, bad_image, bad_json, bad_text):
            tar.add(path, arcname=path.name)

    tokenizer = lambda texts: torch.ones((len(texts), 8), dtype=torch.long)
    loader = build_dataloader(
        _webdataset_args(shard, train_num_samples=1, filter_relabel_success=True),
        transforms.ToTensor(),
        tokenizer,
        is_train=True,
    )
    batch = next(iter(loader))
    assert batch["sample_id"] == ["000000001"]
    assert batch["text"].shape == (1, 4, 8)
    assert len(batch["caption"][0]) == 4
    allowed_parts = {"a raw caption", "a long successful caption"}
    assert all(set(caption.split(".\n")) <= allowed_parts for caption in batch["caption"][0])


def test_wds_caption_source_uses_flair_caption_pool() -> None:
    source = _wds_caption_source_from_sample(
        {
            "text": {
                "captions": {
                    "raw": "a compact raw caption",
                    "shortIB": "a short image based caption",
                    "longIB": "The first long image based sentence. The second long image based sentence.",
                    "shortSV": "a short scene caption",
                    "longSV": "The first long scene sentence. The second long scene sentence.",
                    "shortLLA": "a short language aligned caption",
                    "longLLA": "The first long language aligned sentence. The second long language aligned sentence.",
                }
            }
        },
        relabel_success_key="longSV",
    )

    assert isinstance(source, list)
    assert "a compact raw caption" in source
    assert "a short image based caption" in source
    assert "The first long scene sentence" in source
    assert "The second long language aligned sentence" in source


def test_dynamic_subcaption_sampler_splits_filters_and_merges() -> None:
    torch.manual_seed(0)
    import random

    random.seed(0)
    caption = "Short. A calm lake reflects the orange sky! The trees line the shore?\nwww.example.com !!! A small boat waits near the dock."
    sentences = split_caption(caption)
    assert "Short" not in sentences
    assert len(sentences) == 3

    sampled = sample_subcaptions(sentences, k=8, max_merged_num=3)
    assert len(sampled) == 8
    assert all(item for item in sampled)
    assert any(".\n" in item for item in sampled)


def test_build_dataloader_tokenizes_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "sample.jpg")
    manifest = tmp_path / "toy.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerow({"image": "sample.jpg", "caption": "a small image"})

    args = Namespace(
        train_data=str(manifest),
        val_data=None,
        dataset_type="csv",
        image_root=str(image_dir),
        csv_img_key="image",
        csv_caption_key="caption",
        batch_size=1,
        workers=0,
        image_size=32,
    )
    tokenizer = lambda texts: torch.zeros((len(texts), 8), dtype=torch.long)
    loader = build_dataloader(args, transforms.ToTensor(), tokenizer, is_train=True)
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 3, 32, 32)
    assert batch["text"].shape == (1, 8)
