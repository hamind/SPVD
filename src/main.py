"""Main training entrypoint."""

from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from checkpoint import load_checkpoint, save_checkpoint
from data import get_data
from diagnostics import get_git_info
from distributed import cleanup_distributed, init_distributed_device, is_master, unwrap_model
from factory import create_loss, create_model_and_transforms, create_optimizer, get_tokenizer
from logger import setup_logging
from params import parse_args
from scheduler import cosine_lr
from training import train_one_epoch


def _seed(seed: int, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _write_params(path: Path, args: object) -> None:
    """Write sorted run parameters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name in sorted(vars(args)):
            handle.write(f"{name}: {getattr(args, name)}\n")


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer tensors to the active training device."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _resolved_config(args: object) -> dict[str, object]:
    """Return YAML-safe resolved arguments."""
    resolved: dict[str, object] = {}
    for key, value in vars(args).items():
        if key == "config_dict":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            resolved[key] = value
        elif isinstance(value, (list, tuple)):
            resolved[key] = list(value)
        elif isinstance(value, dict):
            resolved[key] = value
        else:
            resolved[key] = str(value)
    return resolved


def main(argv: Sequence[str] | None = None) -> None:
    """Train a CLIP or SigLIP baseline."""
    args = parse_args(argv)
    dist_info = init_distributed_device(args)
    args.device = dist_info.device
    args.rank = dist_info.rank
    args.local_rank = dist_info.local_rank
    args.world_size = dist_info.world_size
    args.distributed = dist_info.distributed
    _seed(int(args.seed), dist_info.rank)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    name = args.name or f"{args.model}_cc3m"
    log_dir = Path(args.logs_dir) / name
    args.log_dir = str(log_dir)
    logger = setup_logging(log_dir / "out.log" if is_master(args) else None)
    logger.info("running %s on rank %s/%s", name, args.rank, args.world_size)
    writer = None
    if is_master(args):
        _write_params(log_dir / "params.txt", args)
        with (log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_resolved_config(args), handle, sort_keys=True)
        with (log_dir / "git_info.json").open("w", encoding="utf-8") as handle:
            json.dump(get_git_info(Path(__file__).resolve().parents[1]), handle, ensure_ascii=True, indent=2, sort_keys=True)
        if "tensorboard" in str(args.report_to).lower() or "all" in str(args.report_to).lower():
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError as exc:
                if exc.name != "tensorboard":
                    raise
                logger.warning("tensorboard is not installed; continuing without TensorBoard logging")
            else:
                writer = SummaryWriter(log_dir / "tensorboard")

    try:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=args.device,
            force_image_size=args.image_size,
            output_dict=True,
            config_dict=args.config_dict,
            siglip=args.siglip,
            **getattr(args, "model_kwargs", {}),
        )
        tokenizer = get_tokenizer(args.model)
        loss_fn = create_loss(args).to(args.device)
        model = model.to(args.device)

        resume_payload = None
        resume_epoch = 0
        resume_step = 0
        if args.resume:
            resume_payload = load_checkpoint(args.resume, model, map_location=args.device)
            resume_epoch = int(resume_payload.get("epoch", 0) or 0)
            resume_step = int(resume_payload.get("step", 0) or 0)
            logger.info("loaded checkpoint model state: %s (epoch=%d, step=%d)", args.resume, resume_epoch, resume_step)

        if args.distributed:
            model = DistributedDataParallel(model, device_ids=[args.local_rank] if args.device.type == "cuda" else None)

        data = get_data(args, (preprocess_train, preprocess_val), tokenizer=tokenizer)
        if "train" not in data:
            raise ValueError("No training data configured.")
        train_info = data["train"]
        optimizer = create_optimizer(model, args)
        steps_per_epoch = max(int(args.train_num_samples or 0) // max(args.batch_size * args.world_size, 1), 1)
        total_steps = max(steps_per_epoch * int(args.epochs), 1)
        scheduler = cosine_lr(optimizer, float(args.lr), int(args.warmup), total_steps)
        scaler = GradScaler(enabled=str(args.precision).startswith("amp") and args.device.type == "cuda")
        if resume_payload is not None:
            if "optimizer" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer"])
                _optimizer_to_device(optimizer, args.device)
                logger.info("resumed optimizer state from: %s", args.resume)
            else:
                logger.warning("checkpoint has no optimizer state; resume will behave like weight initialization")
            if "scaler" in resume_payload and scaler.is_enabled():
                scaler.load_state_dict(resume_payload["scaler"])
                logger.info("resumed AMP scaler state from: %s", args.resume)

        if args.dry_run:
            logger.info("dry run complete")
            return

        global_step = resume_step
        start_epoch = resume_epoch + 1 if args.resume else 1
        save_frequency = max(int(getattr(args, "save_frequency", 0) or 0), 0)
        if start_epoch > int(args.epochs):
            logger.info("resume checkpoint epoch %d already reached configured epochs=%d", resume_epoch, int(args.epochs))
            return
        logger.info("starting training at epoch=%d, global_step=%d", start_epoch, global_step)
        for epoch in range(start_epoch, int(args.epochs) + 1):
            train_info.set_epoch(epoch)
            steps = train_one_epoch(model, train_info.dataloader, loss_fn, optimizer, scaler, scheduler, epoch, args, logger, writer, global_step)
            global_step += steps
            reached_step_limit = args.max_steps is not None and global_step >= int(args.max_steps)
            is_final_epoch = epoch == int(args.epochs)
            if is_master(args) and save_frequency > 0 and epoch % save_frequency == 0:
                ckpt = log_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
                save_checkpoint(ckpt, unwrap_model(model), optimizer, epoch, global_step, args, scaler=scaler)
                logger.info("saved checkpoint: %s", ckpt)
            if is_master(args) and (is_final_epoch or reached_step_limit):
                ckpt = log_dir / "checkpoints" / "epoch_final.pt"
                save_checkpoint(ckpt, unwrap_model(model), optimizer, epoch, global_step, args, scaler=scaler)
                logger.info("saved final checkpoint: %s", ckpt)
            if reached_step_limit:
                break
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
