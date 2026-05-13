"""Training and feature-extraction loops."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from distributed import is_master


class AverageMeter:
    """Running average meter."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when wrapped by DDP-like containers."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def backward(total_loss: torch.Tensor, scaler: GradScaler | None) -> None:
    """Backpropagate with optional gradient scaling."""
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def _autocast(args: object):
    """Return the appropriate autocast context."""
    precision = str(getattr(args, "precision", "amp"))
    enabled = precision in {"amp", "amp_bf16", "bf16"} and torch.cuda.is_available()
    dtype = torch.bfloat16 if precision in {"amp_bf16", "bf16"} else torch.float16
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)


def _input_dtype(args: object) -> torch.dtype | None:
    """Return image input dtype matching the precision mode."""
    precision = str(getattr(args, "precision", "amp"))
    if precision in {"fp16", "pure_fp16"}:
        return torch.float16
    if precision in {"bf16", "pure_bf16"}:
        return torch.bfloat16
    return None


def _get_batch_tensors(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image and text tensors from the project dataloader batch."""
    if isinstance(batch, dict):
        return batch["image"], batch["text"]
    return batch[:2]


def _loader_count(dataloader: Any, attr: str, fallback: int) -> int:
    """Read dataloader metadata without forcing ``len`` on iterable loaders."""
    value = getattr(dataloader, attr, None)
    if value is not None:
        return int(value)
    try:
        return int(len(dataloader))
    except TypeError:
        return fallback



def _normalize_retrieval_features(features: torch.Tensor) -> torch.Tensor:
    """Return 2D L2-normalized features before retrieval dot products."""
    if features.ndim == 3:
        features = features.mean(dim=1)
    if features.ndim != 2:
        raise ValueError(f"Retrieval features must be 2D or 3D, got shape={tuple(features.shape)}.")
    return torch.nn.functional.normalize(features.float(), dim=-1)

def _unpack_outputs(outputs: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Normalize OpenCLIP output formats."""
    if isinstance(outputs, dict):
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        logit_scale = outputs["logit_scale"]
        logit_bias = outputs.get("logit_bias")
    else:
        image_features, text_features, logit_scale = outputs[:3]
        logit_bias = outputs[3] if len(outputs) > 3 else None
    if logit_bias is None:
        logit_bias = torch.tensor(-10.0, device=device)
    return image_features, text_features, logit_scale, logit_bias


def _scalar_loss_dict(loss: torch.Tensor, logit_scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a minimal OpenCLIP-style loss dict for logging."""
    detached = loss.detach()
    return {
        "loss": detached,
        "loss_infonce": detached,
        "logit_scale": logit_scale.detach(),
    }


def _loss_dict_to_floats(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert scalar tensor metrics into Python floats."""
    values: dict[str, float] = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value) and value.numel() == 1:
            values[key] = float(value.detach().cpu())
    return values


def _prefixed_loss_values(loss_dict: dict[str, torch.Tensor], total_loss: torch.Tensor) -> dict[str, float]:
    """Map available scalar loss names to stable train/* diagnostics keys."""
    values = _loss_dict_to_floats(loss_dict)
    metrics: dict[str, float] = {"train/loss_total": float(total_loss.detach().cpu())}
    aliases = {
        "loss": "loss_total",
        "loss_align": "loss_align",
        "loss_infonce": "loss_align",
        "loss_sigmoid": "loss_align",
        "loss_align_global": "loss_align_global",
        "loss_align_caption": "loss_align_caption",
        "loss_branch": "loss_branch",
        "loss_branch_s_text": "loss_branch_s_text",
        "loss_branch_r_text": "loss_branch_r_text",
        "branch_sim_s_text": "branch_sim_s_text",
        "branch_sim_r_text": "branch_sim_r_text",
        "branch_gap_s_minus_r": "branch_gap_s_minus_r",
        "loss_residual_variance": "loss_residual_variance",
        "gate_mean": "gate_mean",
        "gate_std": "gate_std",
        "gate_min": "gate_min",
        "gate_max": "gate_max",
        "loss_pri": "loss_private",
        "loss_private": "loss_private",
        "loss_dis": "loss_dis",
        "loss_sem": "loss_semantic",
        "loss_uniform": "loss_uniform",
        "loss_bce": "loss_bce",
        "loss_routing": "loss_routing",
        "loss_lang_sem": "loss_lang_sem",
    }
    for key, value in values.items():
        if key == "logit_scale":
            continue
        metrics[f"train/{aliases.get(key, key)}"] = value
    return metrics


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = lambda: _autocast(args)
    input_dtype = _input_dtype(args)

    model.train()

    if isinstance(data, dict):
        data["train"].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = data["train"].dataloader
    else:
        dataloader = getattr(data, "dataloader", data)

    raw_num_batches = _loader_count(dataloader, "num_batches", 1)
    raw_num_samples = _loader_count(dataloader, "num_samples", raw_num_batches * int(args.batch_size) * int(args.world_size))
    num_batches_per_epoch = max(raw_num_batches, 1)
    sample_digits = math.ceil(math.log(raw_num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    logger = logging.getLogger("spvd")
    end = time.time()
    optimizer_steps = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * (epoch - 1) + i

        if not bool(getattr(args, "skip_scheduler", False)):
            scheduler(step)

        images, texts = _get_batch_tensors(batch)
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            model_out = model(images, texts)
            logit_scale = model_out["logit_scale"]
            losses = loss(**model_out, output_dict=True)

            total_loss = sum(losses.values())
            losses["loss"] = total_loss

        backward(total_loss, scaler)

        if scaler is not None:
            if bool(getattr(args, "horovod", False)):
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                grad_clip_norm = getattr(args, "grad_clip_norm", None)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                grad_clip_norm = getattr(args, "grad_clip_norm", None)
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            grad_clip_norm = getattr(args, "grad_clip_norm", None)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
            optimizer.step()
        optimizer_steps += 1

        reached_step_limit = args.max_steps is not None and optimizer_steps >= int(args.max_steps)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch or reached_step_limit):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = raw_num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {math.log(logit_scale_scalar):.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": math.log(logit_scale_scalar),
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data = {"train/" + name: val for name, val in log_data.items()}

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if reached_step_limit:
            break

    return optimizer_steps


@torch.no_grad()
def extract_features(model: nn.Module, data_loader: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized image and text features for retrieval."""
    model.eval()
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    for batch in data_loader:
        outputs = model(batch["image"].to(device), batch["text"].to(device))
        image, text, _, _ = _unpack_outputs(outputs, device)
        image_features.append(_normalize_retrieval_features(image).cpu())
        text_features.append(_normalize_retrieval_features(text).cpu())
    if not image_features:
        raise ValueError("Dataloader produced no samples.")
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)
