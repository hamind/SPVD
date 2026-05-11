"""Training and feature-extraction loops."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from diagnostics import (
    collect_feature_tensors,
    compute_feature_stats,
    compute_global_param_stats,
    compute_logits_stats,
    compute_redundancy_stats,
    compute_update_stats,
    snapshot_params_for_update,
    write_jsonl,
)
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
    return model.module if hasattr(model, "module") else model


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
        "loss_dec": "loss_compress",
        "loss_decomp": "loss_compress",
        "loss_residual": "loss_private",
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


def _write_tb_scalars(tb_writer: Any | None, metrics: dict[str, float], step: int) -> None:
    if tb_writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tb_writer.add_scalar(key, value, step)


def _compute_loss(loss_fn: nn.Module, outputs: Any, args: object, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute CLIP, SigLIP, or SPVD loss."""
    if bool(getattr(loss_fn, "expects_output_dict", False)):
        total_loss, loss_dict = loss_fn(outputs)
        return total_loss, loss_dict

    image_features, text_features, logit_scale, logit_bias = _unpack_outputs(outputs, device)
    if bool(getattr(args, "siglip", False)):
        loss = loss_fn(image_features, text_features, logit_scale, logit_bias)
    else:
        loss = loss_fn(image_features, text_features, logit_scale)
    return loss, _scalar_loss_dict(loss, logit_scale)


def train_one_epoch(
    model: nn.Module,
    data_loader: Any,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    epoch: int,
    args: object,
    logger: Any,
    tb_writer: Any | None = None,
    global_step_offset: int = 0,
) -> int:
    """Train one epoch and return optimizer steps."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = getattr(args, "device")
    accum_freq = max(int(getattr(args, "accum_freq", 1)), 1)
    log_interval = int(getattr(args, "log_every_n_steps", 50))
    diagnostics_enabled = bool(getattr(args, "diagnostics_enabled", True))
    diag_log_interval = max(int(getattr(args, "diagnostics_log_interval", log_interval)), 1)
    heavy_log_interval = max(int(getattr(args, "diagnostics_heavy_log_interval", 500)), 1)
    diag_eps = float(getattr(args, "diagnostics_eps", 1.0e-12))
    log_update_rms = bool(getattr(args, "diagnostics_log_update_rms", True))
    log_effective_rank = bool(getattr(args, "diagnostics_log_effective_rank", True))
    log_redundancy = bool(getattr(args, "diagnostics_log_redundancy", True))
    per_layer = bool(getattr(args, "diagnostics_per_layer", True))
    max_logged_layers = int(getattr(args, "diagnostics_max_logged_layers", 80))
    diag_dir = Path(getattr(args, "log_dir", getattr(args, "logs_dir", ".")))
    train_metrics_path = diag_dir / "train_metrics.jsonl"
    per_layer_path = diag_dir / "diagnostics_per_layer.jsonl"
    summary_path = diag_dir / "diagnostics_summary.json"
    world_size = int(getattr(args, "world_size", 1))
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.perf_counter()
    start = end
    samples_seen = 0
    optimizer_steps = 0
    recent_clip_flags: deque[bool] = deque(maxlen=100)
    last_diagnostics: dict[str, float] = {}

    for step, batch in enumerate(data_loader, start=1):
        data_time_m.update(time.perf_counter() - end)
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"].to(device, non_blocking=True)
        if scheduler is not None:
            scheduler(optimizer_steps)
        with _autocast(args):
            outputs = model(images, texts)
            raw_loss, loss_dict = _compute_loss(loss_fn, outputs, args, device)
            loss = raw_loss / accum_freq
        backward(loss, scaler)
        samples_seen += images.shape[0] * world_size

        if step % accum_freq == 0:
            next_optimizer_step = global_step_offset + optimizer_steps + 1
            should_log_diag = diagnostics_enabled and next_optimizer_step % diag_log_interval == 0
            should_log_heavy = diagnostics_enabled and next_optimizer_step % heavy_log_interval == 0
            if scaler is not None and scaler.is_enabled() and (should_log_diag or should_log_heavy):
                scaler.unscale_(optimizer)
            grad_stats = compute_global_param_stats(model, distributed=bool(getattr(args, "distributed", False)), eps=diag_eps) if should_log_diag else {}
            grad_norm_before_clip = grad_stats.get("grad_norm")
            grad_norm_after_clip = grad_norm_before_clip
            clip_applied = False
            recent_clip_flags.append(clip_applied)
            update_snapshot = None
            if should_log_diag and log_update_rms:
                update_snapshot = snapshot_params_for_update(model)
            old_scale = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale() if scaler is not None and scaler.is_enabled() else old_scale
            optimizer_step_skipped = bool(old_scale is not None and new_scale is not None and new_scale < old_scale)
            update_stats: dict[str, float] = {}
            layer_records: list[dict[str, float | str]] = []
            if should_log_diag and log_update_rms:
                if optimizer_step_skipped:
                    update_stats = {"update_rms": 0.0, "update_weight_ratio": 0.0}
                else:
                    update_stats, layer_records = compute_update_stats(
                        model,
                        update_snapshot,
                        distributed=bool(getattr(args, "distributed", False)),
                        eps=diag_eps,
                        max_logged_layers=max_logged_layers if per_layer and should_log_heavy else 0,
                    )
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if should_log_diag and is_master(args):
                elapsed = max(time.perf_counter() - start, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                raw_loss_value = float((loss.detach() * accum_freq).cpu())
                metrics: dict[str, float] = {
                    "train/lr": float(lr),
                    "train/data_time": float(data_time_m.val),
                    "train/batch_time": float(batch_time_m.val),
                    "train/samples_per_second": samples_seen / elapsed,
                    "train/grad_norm_before_clip": float(grad_norm_before_clip or 0.0),
                    "train/grad_norm_after_clip": float(grad_norm_after_clip or 0.0),
                    "train/clip_applied": float(clip_applied),
                    "train/clip_fraction": sum(recent_clip_flags) / max(len(recent_clip_flags), 1),
                    "train/optimizer_step_skipped": float(optimizer_step_skipped),
                }
                metrics.update(_prefixed_loss_values(loss_dict, loss.detach() * accum_freq))
                metrics.update({f"train/{key}": value for key, value in grad_stats.items()})
                metrics.update({f"train/{key}": value for key, value in update_stats.items()})
                metrics.update(compute_logits_stats(outputs, eps=diag_eps))
                if should_log_heavy:
                    features = collect_feature_tensors(outputs)
                    if log_effective_rank:
                        for name, tensor in features.items():
                            metrics.update(compute_feature_stats(tensor, f"train/{name}", eps=diag_eps))
                    if log_redundancy:
                        metrics.update(compute_redundancy_stats(features.get("z_vs"), features.get("z_vp"), eps=diag_eps))
                record = {
                    "step": step,
                    "epoch": epoch,
                    "global_step": next_optimizer_step,
                    "lr": lr,
                    "loss_total": metrics.get("train/loss_total", raw_loss_value),
                    "grad_rms": metrics.get("train/grad_rms"),
                    "weight_rms": metrics.get("train/weight_rms"),
                    "update_rms": metrics.get("train/update_rms"),
                    **metrics,
                }
                write_jsonl(train_metrics_path, record)
                if per_layer and should_log_heavy:
                    for layer_record in layer_records:
                        write_jsonl(per_layer_path, {"step": step, "epoch": epoch, "global_step": next_optimizer_step, **layer_record})
                        layer_name = str(layer_record["layer_name"])
                        for key in ("weight_rms", "grad_rms", "update_rms", "update_weight_ratio"):
                            metrics[f"train/layer/{layer_name}/{key}"] = float(layer_record[key])
                _write_tb_scalars(tb_writer, metrics, next_optimizer_step)
                last_diagnostics = metrics

        if is_master(args) and step % log_interval == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            lr = optimizer.param_groups[0]["lr"]
            _, _, logit_scale, _ = _unpack_outputs(outputs, device)
            raw_loss_value = float((loss.detach() * accum_freq).cpu())
            logit_scale_value = float(logit_scale.detach().cpu())
            samples_per_second = samples_seen / elapsed
            batch_time_m.update(time.perf_counter() - end)
            loss_values = _loss_dict_to_floats(loss_dict)
            aux_keys = (
                "loss_align",
                "loss_sigmoid",
                "loss_align_global",
                "loss_align_caption",
                "loss_dec",
                "loss_pri",
                "loss_dis",
                "loss_sem",
                "shared_coverage",
                "private_coverage",
            )
            aux_text = " ".join(f"{key}={loss_values[key]:.4f}" for key in aux_keys if key in loss_values)
            logger.info(
                "epoch=%s step=%s loss=%.4f logit_scale=%.3f lr=%.6g data_time=%.3f batch_time=%.3f samples/sec=%.2f%s",
                epoch,
                step,
                raw_loss_value,
                logit_scale_value,
                lr,
                data_time_m.avg,
                batch_time_m.avg,
                samples_per_second,
                f" {aux_text}" if aux_text else "",
            )
            if tb_writer is not None and not (diagnostics_enabled and (global_step_offset + optimizer_steps) % diag_log_interval == 0):
                tb_step = global_step_offset + optimizer_steps
                tb_writer.add_scalar("train/loss", raw_loss_value, tb_step)
                tb_writer.add_scalar("train/logit_scale", logit_scale_value, tb_step)
                tb_writer.add_scalar("train/lr", lr, tb_step)
                tb_writer.add_scalar("train/data_time", data_time_m.val, tb_step)
                tb_writer.add_scalar("train/batch_time", batch_time_m.val, tb_step)
                tb_writer.add_scalar("train/samples_per_second", samples_per_second, tb_step)
                for key, value in loss_values.items():
                    if key not in {"loss", "logit_scale"}:
                        tb_writer.add_scalar(f"train/{key}", value, tb_step)

        max_steps = getattr(args, "max_steps", None)
        if max_steps is not None and optimizer_steps >= int(max_steps):
            break
        end = time.perf_counter()
    if diagnostics_enabled and is_master(args) and last_diagnostics:
        import json

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(last_diagnostics, handle, ensure_ascii=True, allow_nan=True, indent=2, sort_keys=True)
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
        image_features.append(image.cpu())
        text_features.append(text.cpu())
    if not image_features:
        raise ValueError("Dataloader produced no samples.")
    return torch.cat(image_features), torch.cat(text_features)
