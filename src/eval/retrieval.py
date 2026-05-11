"""Retrieval evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from data import build_dataloader
from distributed import cleanup_distributed, init_distributed_device, is_master
from eval.metrics import retrieval_metrics
from factory import create_model_and_transforms, create_tokenizer
from logger import setup_logging
from params import parse_args
from training import extract_features


def main() -> None:
    """Run retrieval evaluation."""
    front = argparse.ArgumentParser(add_help=False)
    front.add_argument("--checkpoint", default=None)
    known, rest = front.parse_known_args()
    args = parse_args(rest)
    dist_info = init_distributed_device(args)
    args.device = dist_info.device
    args.rank = dist_info.rank
    args.local_rank = dist_info.local_rank
    args.world_size = dist_info.world_size
    logger = setup_logging(Path(args.logs_dir) / (args.name or "retrieval") / "retrieval.log" if is_master(args) else None)
    try:
        model, _, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=args.device,
            force_image_size=args.image_size,
            output_dict=True,
            config_dict=args.config_dict,
            siglip=args.siglip,
        )
        model = model.to(args.device)
        if known.checkpoint:
            load_checkpoint(known.checkpoint, model, map_location=args.device)
            logger.info("loaded checkpoint: %s", known.checkpoint)
        tokenizer = create_tokenizer(args.model)
        loader = build_dataloader(args, preprocess_val, tokenizer, is_train=False)
        image_features, text_features = extract_features(model, loader, args.device)
        metrics = retrieval_metrics(image_features @ text_features.t())
        if is_master(args):
            out_dir = Path(args.logs_dir) / (args.name or "retrieval")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            for key, value in metrics.items():
                logger.info("%s=%.4f", key, value)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
