"""Low-overhead training diagnostics for SPVD runs."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


def _dist_ready(distributed: bool = False) -> bool:
    return bool(distributed and dist is not None and dist.is_available() and dist.is_initialized())


def _all_reduce_sums(values: Tensor, distributed: bool = False) -> Tensor:
    if _dist_ready(distributed):
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def _finite_float(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() != 1:
            return float("nan")
        value = value.float().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rms(sum_sq: float, count: float, eps: float = 1.0e-12) -> float:
    if count <= 0:
        return 0.0
    return math.sqrt(max(sum_sq, 0.0) / max(count, eps))


@torch.no_grad()
def compute_global_param_stats(model: nn.Module, distributed: bool = False, eps: float = 1.0e-12) -> dict[str, float]:
    """Compute global gradient and parameter RMS without moving tensors to CPU first."""
    device = next((p.device for p in model.parameters() if p.requires_grad), torch.device("cpu"))
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    grad_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    for param in model.parameters():
        if not param.requires_grad:
            continue
        data = param.detach().float()
        totals[2] += data.pow(2).sum(dtype=torch.float64)
        totals[3] += data.numel()
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_sq = grad.pow(2).sum(dtype=torch.float64)
        totals[0] += grad_sq
        totals[1] += grad.numel()
        grad_norm_sq += grad_sq
    totals = _all_reduce_sums(totals, distributed)
    grad_norm_sq = _all_reduce_sums(grad_norm_sq, distributed)
    grad_rms = _rms(float(totals[0].item()), float(totals[1].item()), eps)
    weight_rms = _rms(float(totals[2].item()), float(totals[3].item()), eps)
    return {
        "grad_norm": math.sqrt(max(float(grad_norm_sq.item()), 0.0)),
        "grad_rms": grad_rms,
        "weight_rms": weight_rms,
        "grad_weight_ratio": grad_rms / max(weight_rms, eps),
    }


def should_log_layer(name: str, param: Tensor) -> bool:
    """Select major trainable matrix-like layers for per-layer diagnostics."""
    if param.ndim < 2:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias") or "norm" in lowered or "embedding" in lowered:
        return False
    return True


@torch.no_grad()
def snapshot_params_for_update(model: nn.Module, filter_fn: Any | None = None) -> dict[str, Tensor]:
    """Clone selected trainable parameters immediately before optimizer.step."""
    snapshot: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filter_fn is not None and not filter_fn(name, param):
            continue
        snapshot[name] = param.detach().clone()
    return snapshot


@torch.no_grad()
def compute_update_stats(
    model: nn.Module,
    snapshot: dict[str, Tensor] | None,
    distributed: bool = False,
    eps: float = 1.0e-12,
    max_logged_layers: int = 80,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Compute RMS of actual parameter updates after optimizer.step."""
    if not snapshot:
        return {"update_rms": 0.0, "update_weight_ratio": 0.0}, []
    device = next((p.device for p in model.parameters() if p.requires_grad), torch.device("cpu"))
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    layer_records: list[dict[str, float | str]] = []
    for name, param in model.named_parameters():
        before = snapshot.get(name)
        if before is None:
            continue
        after = param.detach()
        update = after.float() - before.to(device=after.device, dtype=after.dtype).float()
        update_sq = update.pow(2).sum(dtype=torch.float64)
        weight_sq = after.float().pow(2).sum(dtype=torch.float64)
        count = update.numel()
        totals[0] += update_sq
        totals[1] += weight_sq
        totals[2] += count
        if len(layer_records) < max_logged_layers and should_log_layer(name, param):
            grad_sq = param.grad.detach().float().pow(2).sum(dtype=torch.float64) if param.grad is not None else update_sq.new_zeros(())
            grad_count = param.grad.numel() if param.grad is not None else 0
            weight_rms = _rms(float(weight_sq.item()), float(count), eps)
            update_rms = _rms(float(update_sq.item()), float(count), eps)
            layer_records.append(
                {
                    "layer_name": name,
                    "weight_rms": weight_rms,
                    "grad_rms": _rms(float(grad_sq.item()), float(grad_count), eps),
                    "update_rms": update_rms,
                    "update_weight_ratio": update_rms / max(weight_rms, eps),
                }
            )
    totals = _all_reduce_sums(totals, distributed)
    update_rms = _rms(float(totals[0].item()), float(totals[2].item()), eps)
    weight_rms = _rms(float(totals[1].item()), float(totals[2].item()), eps)
    return {"update_rms": update_rms, "update_weight_ratio": update_rms / max(weight_rms, eps)}, layer_records


@torch.no_grad()
def compute_feature_stats(features: Tensor | None, prefix: str, eps: float = 1.0e-12) -> dict[str, float]:
    """Compute local-batch representation diagnostics for a [B, D] feature tensor."""
    if features is None or not torch.is_tensor(features) or features.ndim != 2:
        return {}
    x = features.detach().float()
    if x.numel() == 0:
        return {}
    stats: dict[str, float] = {}
    norms = x.norm(dim=-1)
    stats[f"{prefix}/norm_mean"] = _finite_float(norms.mean())
    stats[f"{prefix}/norm_std"] = _finite_float(norms.std(unbiased=False))
    stats[f"{prefix}/feature_mean_abs"] = _finite_float(x.mean(dim=0).abs().mean())
    stats[f"{prefix}/feature_variance_mean"] = _finite_float(x.var(dim=0, unbiased=False).mean())
    if x.shape[0] < 2:
        stats[f"{prefix}/effective_rank"] = float("nan")
        stats[f"{prefix}/singular_value_top1"] = float("nan")
        stats[f"{prefix}/singular_value_top5_mean"] = float("nan")
        return stats
    centered = x - x.mean(dim=0, keepdim=True)
    try:
        singular_values = torch.linalg.svdvals(centered)
    except RuntimeError:
        return stats
    if singular_values.numel() == 0:
        return stats
    probs = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    top_k = min(5, int(singular_values.numel()))
    stats[f"{prefix}/effective_rank"] = _finite_float(torch.exp(entropy))
    stats[f"{prefix}/singular_value_top1"] = _finite_float(singular_values[0])
    stats[f"{prefix}/singular_value_top5_mean"] = _finite_float(singular_values[:top_k].mean())
    return stats


@torch.no_grad()
def compute_linear_cka(x: Tensor, y: Tensor, eps: float = 1.0e-12) -> float:
    """Compute linear CKA on local detached features."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    x = x.detach().float() - x.detach().float().mean(dim=0, keepdim=True)
    y = y.detach().float() - y.detach().float().mean(dim=0, keepdim=True)
    hsic = (x.transpose(0, 1) @ y).pow(2).sum()
    x_norm = (x.transpose(0, 1) @ x).pow(2).sum().sqrt()
    y_norm = (y.transpose(0, 1) @ y).pow(2).sum().sqrt()
    return _finite_float(hsic / (x_norm * y_norm).clamp_min(eps))


@torch.no_grad()
def compute_redundancy_stats(zs: Tensor | None, zp: Tensor | None, eps: float = 1.0e-12) -> dict[str, float]:
    """Compute local-batch shared/private redundancy diagnostics."""
    if zs is None or zp is None or not torch.is_tensor(zs) or not torch.is_tensor(zp):
        return {}
    if zs.ndim != 2 or zp.ndim != 2 or zs.shape[0] != zp.shape[0]:
        return {}
    if zs.shape[0] < 2:
        return {
            "train/redundancy/zvs_zvp_cosine_abs_mean": float("nan"),
            "train/redundancy/zvs_zvp_cross_cov_fro": float("nan"),
            "train/redundancy/zvs_zvp_linear_cka": float("nan"),
        }
    x = zs.detach().float()
    y = zp.detach().float()
    stats: dict[str, float] = {}
    if x.shape[-1] == y.shape[-1]:
        stats["train/redundancy/zvs_zvp_cosine_abs_mean"] = _finite_float(torch.nn.functional.cosine_similarity(x, y, dim=-1).abs().mean())
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    cross_cov = x_centered.transpose(0, 1) @ y_centered / float(max(int(x.shape[0]) - 1, 1))
    stats["train/redundancy/zvs_zvp_cross_cov_fro"] = _finite_float(torch.linalg.norm(cross_cov, ord="fro"))
    stats["train/redundancy/zvs_zvp_linear_cka"] = compute_linear_cka(x, y, eps)
    return stats


@torch.no_grad()
def collect_feature_tensors(outputs: Any) -> dict[str, Tensor]:
    """Best-effort feature discovery from model outputs without assuming one schema."""
    if not isinstance(outputs, dict):
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            return {"image_features": outputs[0], "text_features": outputs[1]}
        return {}
    aliases = {
        "z_vs": ("z_v_s", "image_features_shared", "shared_features", "zs", "shared_global", "image_features"),
        "z_vp": ("z_v_p", "image_features_private", "private_features", "zp", "private_global", "residual_visual_features"),
        "z_t": ("z_t", "text_features", "text_global"),
    }
    found: dict[str, Tensor] = {}
    for canonical, names in aliases.items():
        for name in names:
            value = outputs.get(name)
            if torch.is_tensor(value) and value.ndim == 2:
                found[canonical] = value
                break
    return found


@torch.no_grad()
def compute_logits_stats(outputs: Any, eps: float = 1.0e-12) -> dict[str, float]:
    """Compute cheap logit-scale and local positive/negative logit diagnostics."""
    stats: dict[str, float] = {}
    def first_tensor(mapping: dict[str, Any], names: tuple[str, ...]) -> Tensor | None:
        for name in names:
            value = mapping.get(name)
            if torch.is_tensor(value):
                return value
        return None

    if isinstance(outputs, dict):
        image = first_tensor(outputs, ("image_features", "z_v_s", "shared_global"))
        text = first_tensor(outputs, ("text_features", "z_t", "text_global"))
        logit_scale = outputs.get("logit_scale")
        temperature = outputs.get("temperature")
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        image, text, logit_scale = outputs[:3]
        temperature = None
    else:
        return stats
    if torch.is_tensor(logit_scale) and logit_scale.numel() == 1:
        scale = logit_scale.detach().float()
        stats["train/logit_scale"] = _finite_float(scale)
        stats["train/logit_scale_exp"] = _finite_float(scale.exp())
        stats["train/temperature"] = _finite_float(1.0 / scale.exp().clamp_min(eps))
    if torch.is_tensor(temperature) and temperature.numel() == 1:
        stats["train/temperature"] = _finite_float(temperature)
    if torch.is_tensor(image) and torch.is_tensor(text) and image.ndim == 2 and text.ndim == 2 and image.shape[0] == text.shape[0]:
        if image.shape[0] > 1 and image.shape[1] == text.shape[1]:
            scale = logit_scale.detach().float() if torch.is_tensor(logit_scale) else image.new_tensor(1.0)
            logits_i2t = scale * torch.nn.functional.normalize(image.detach().float(), dim=-1) @ torch.nn.functional.normalize(text.detach().float(), dim=-1).T
            logits_t2i = logits_i2t.T
            eye = torch.eye(logits_i2t.shape[0], device=logits_i2t.device, dtype=torch.bool)
            stats["train/logits_pos_mean"] = _finite_float(logits_i2t.diag().mean())
            stats["train/logits_neg_mean"] = _finite_float(logits_i2t.masked_select(~eye).mean())
            stats["train/logits_i2t_pos_mean"] = stats["train/logits_pos_mean"]
            stats["train/logits_i2t_neg_mean"] = stats["train/logits_neg_mean"]
            stats["train/logits_t2i_pos_mean"] = _finite_float(logits_t2i.diag().mean())
            stats["train/logits_t2i_neg_mean"] = _finite_float(logits_t2i.masked_select(~eye).mean())
    return stats


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=True, sort_keys=True) + "\n")


def get_git_info(repo_dir: str | Path) -> dict[str, Any]:
    """Return git commit metadata, or unknown values outside a git repo."""
    repo_dir = Path(repo_dir)

    def run_git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo_dir), *args], stderr=subprocess.DEVNULL, text=True).strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run_git("status", "--porcelain"))
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
