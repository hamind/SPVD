"""Training and feature-extraction loops."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from diagnostics import write_jsonl
from distributed import is_master, unwrap_model


def backward(total_loss: torch.Tensor, scaler: GradScaler | None) -> None:
    """Backpropagate with optional gradient scaling."""
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def clamp_logit_scale_(model: nn.Module, min_scale: float, max_scale: float) -> bool:
    """Clamp a model logit_scale parameter whose stored value is log-space."""
    raw_model = unwrap_model(model)
    if not hasattr(raw_model, "logit_scale"):
        return False
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if min_scale <= 0 or max_scale <= 0 or max_scale < min_scale:
        raise ValueError(f"Invalid logit scale clamp range: min_scale={min_scale}, max_scale={max_scale}")
    with torch.no_grad():
        raw_model.logit_scale.clamp_(math.log(min_scale), math.log(max_scale))
    return True


def clip_gradients_(model: nn.Module, grad_clip_norm: float | None) -> bool:
    """Optionally clip gradients and report whether clipping was requested."""
    if grad_clip_norm is None or float(grad_clip_norm) <= 0:
        return False
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), norm_type=2.0)
    return True


def _model_logit_stats(model: nn.Module, eps: float = 1.0e-12) -> dict[str, float]:
    """Read post-step logit scale/bias values from the unwrapped model."""
    raw_model = unwrap_model(model)
    stats: dict[str, float] = {}
    logit_scale = getattr(raw_model, "logit_scale", None)
    if torch.is_tensor(logit_scale) and logit_scale.numel() == 1:
        log_scale = float(logit_scale.detach().float().cpu())
        scale = math.exp(log_scale)
        stats["train/logit_scale_exp"] = scale
        stats["train/logit_scale_log"] = log_scale
        stats["train/logit_scale"] = scale
        stats["train/temperature"] = 1.0 / max(scale, eps)
    logit_bias = getattr(raw_model, "logit_bias", None)
    if torch.is_tensor(logit_bias) and logit_bias.numel() == 1:
        stats["train/logit_bias"] = float(logit_bias.detach().float().cpu())
    return stats


def _autocast(args: object):
    """Return the appropriate autocast context."""
    precision = str(getattr(args, "precision", "amp"))
    enabled = precision in {"amp", "amp_bf16", "bf16"} and torch.cuda.is_available()
    dtype = torch.bfloat16 if precision in {"amp_bf16", "bf16"} else torch.float16
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)



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


def _extract_logit_scale(outputs: Any, device: torch.device) -> torch.Tensor:
    """Read logit_scale without assuming OpenCLIP feature-key names."""
    if isinstance(outputs, dict):
        logit_scale = outputs.get("logit_scale")
    else:
        logit_scale = outputs[2] if isinstance(outputs, (tuple, list)) and len(outputs) >= 3 else None
    if torch.is_tensor(logit_scale):
        return logit_scale
    if logit_scale is not None:
        return torch.tensor(float(logit_scale), device=device)
    return torch.tensor(1.0, device=device)


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


def _is_loss_metric_key(key: str) -> bool:
    """Keep logs focused on actual loss terms."""
    if key in {"loss_branch_weight_effective", "loss_residual_variance_weight_effective"}:
        return False
    return key == "loss" or key.startswith("loss_") or key.endswith("_loss")


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
        "caption_primary_loss": "caption_primary_loss",
        "caption_pos_loss": "caption_pos_loss",
        "caption_neg_loss": "caption_neg_loss",
        "caption_region_positive_loss": "caption_region_positive_loss",
        "caption_region_negative_loss": "caption_region_negative_loss",
        "caption_region_same_image_loss": "caption_region_same_image_loss",
        "loss_branch": "loss_branch",
        "loss_branch_s_text": "loss_branch_s_text",
        "loss_branch_r_text": "loss_branch_r_text",
        "loss_residual_variance": "loss_residual_variance",
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
        if key == "logit_scale" or not _is_loss_metric_key(key):
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
    diag_dir = Path(getattr(args, "log_dir", getattr(args, "logs_dir", ".")))
    train_metrics_path = diag_dir / "train_metrics.jsonl"
    summary_path = diag_dir / "diagnostics_summary.json"
    optimizer_steps = 0
    last_diagnostics: dict[str, float] = {}
    grad_clip_norm = getattr(args, "grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    for step, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"].to(device, non_blocking=True)
        current_global_step = global_step_offset + optimizer_steps
        if scheduler is not None:
            scheduler(current_global_step)
        if hasattr(loss_fn, "set_global_step"):
            loss_fn.set_global_step(current_global_step)
        with _autocast(args):
            outputs = model(images, texts)
            raw_loss, loss_dict = _compute_loss(loss_fn, outputs, args, device)
            loss = raw_loss / accum_freq
        backward(loss, scaler)

        if step % accum_freq == 0:
            next_optimizer_step = global_step_offset + optimizer_steps + 1
            should_log_diag = diagnostics_enabled and next_optimizer_step % diag_log_interval == 0
            needs_unscale = (
                scaler is not None
                and scaler.is_enabled()
                and (grad_clip_norm is not None and grad_clip_norm > 0)
            )
            if needs_unscale:
                scaler.unscale_(optimizer)
            clip_gradients_(model, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if bool(getattr(args, "clamp_logit_scale", False)):
                clamp_logit_scale_(
                    model,
                    float(getattr(args, "min_logit_scale", 1.0)),
                    float(getattr(args, "max_logit_scale", 100.0)),
                )
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if should_log_diag and is_master(args):
                raw_loss_value = float((loss.detach() * accum_freq).cpu())
                metrics = _prefixed_loss_values(loss_dict, loss.detach() * accum_freq)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "global_step": next_optimizer_step,
                    "loss_total": metrics.get("train/loss_total", raw_loss_value),
                    **metrics,
                }
                write_jsonl(train_metrics_path, record)
                _write_tb_scalars(tb_writer, metrics, next_optimizer_step)
                last_diagnostics = metrics

        if is_master(args) and step % log_interval == 0:
            raw_loss_value = float((loss.detach() * accum_freq).cpu())
            loss_values = _loss_dict_to_floats(loss_dict)
            logged_losses: dict[str, float] = {"loss": raw_loss_value}
            for key, value in loss_values.items():
                if key != "loss" and _is_loss_metric_key(key):
                    logged_losses[key] = value
            loss_text = " ".join(f"{key}={value:.4f}" for key, value in logged_losses.items())
            logger.info("epoch=%s step=%s %s", epoch, step, loss_text)
            if tb_writer is not None and not (diagnostics_enabled and (global_step_offset + optimizer_steps) % diag_log_interval == 0):
                tb_step = global_step_offset + optimizer_steps
                for key, value in logged_losses.items():
                    tb_key = "train/loss_total" if key == "loss" else f"train/{key}"
                    tb_writer.add_scalar(tb_key, value, tb_step)

        max_steps = getattr(args, "max_steps", None)
        if max_steps is not None and (global_step_offset + optimizer_steps) >= int(max_steps):
            break
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
        image_features.append(_normalize_retrieval_features(image).cpu())
        text_features.append(_normalize_retrieval_features(text).cpu())
    if not image_features:
        raise ValueError("Dataloader produced no samples.")
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)
