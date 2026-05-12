"""SPVD loss functions with OpenCLIP-style contrastive plumbing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from open_clip.loss import SigLipLoss as OpenCLIPSigLipLoss
from torch import Tensor, nn


def _zero(reference: Tensor | None = None) -> Tensor:
    if reference is None:
        return torch.tensor(0.0)
    return reference.new_zeros(())


def _first_tensor(outputs: Mapping[str, Any], names: tuple[str, ...]) -> Tensor | None:
    for name in names:
        value = outputs.get(name)
        if torch.is_tensor(value):
            return value
    return None


def assert_finite_tensor(name: str, tensor: Tensor | None, enabled: bool = False) -> None:
    if not enabled or tensor is None or not torch.is_tensor(tensor):
        return
    if not torch.isfinite(tensor).all():
        finite = torch.isfinite(tensor)
        finite_ratio = finite.float().mean().item()
        nonfinite_count = int((~finite).sum().item())
        raise FloatingPointError(
            f"{name} contains non-finite values: "
            f"finite_ratio={finite_ratio:.6f}, nonfinite_count={nonfinite_count}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )


class ResidualVarianceLoss(nn.Module):
    """Variance floor for the residual branch."""

    def __init__(self, gamma: float = 1.0, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, residual_features: Tensor) -> Tensor:
        residual = residual_features.float().reshape(-1, residual_features.shape[-1])
        std = torch.sqrt(residual.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.gamma - std).mean()


class BranchBCELoss(nn.Module):
    """Semantic-text positive BCE plus residual-text negative BCE."""

    def __init__(
        self,
        logit_scale: float = 5.0,
        residual_negative_weight: float = 0.25,
        detach_text_for_residual: bool = True,
    ) -> None:
        super().__init__()
        self.logit_scale = float(logit_scale)
        self.residual_negative_weight = float(residual_negative_weight)
        self.detach_text_for_residual = bool(detach_text_for_residual)

    def forward(self, shared_features: Tensor, residual_features: Tensor, text_features: Tensor) -> dict[str, Tensor]:
        if shared_features.shape[:-1] != residual_features.shape[:-1]:
            raise ValueError(
                "shared_features and residual_features must have matching leading dimensions, "
                f"got {tuple(shared_features.shape)} and {tuple(residual_features.shape)}."
            )
        if shared_features.shape[:-1] != text_features.shape[:-1]:
            raise ValueError(
                "shared_features and text_features must have matching leading dimensions for branch BCE, "
                f"got {tuple(shared_features.shape)} and {tuple(text_features.shape)}."
            )
        shared = F.normalize(shared_features.float().reshape(-1, shared_features.shape[-1]), dim=-1)
        residual = F.normalize(residual_features.float().reshape(-1, residual_features.shape[-1]), dim=-1)
        text = F.normalize(text_features.float().reshape(-1, text_features.shape[-1]), dim=-1)
        residual_text = text.detach() if self.detach_text_for_residual else text

        sim_s_text = (shared * text).sum(dim=-1)
        sim_r_text = (residual * residual_text).sum(dim=-1)
        logits_s = self.logit_scale * sim_s_text
        logits_r = self.logit_scale * sim_r_text
        loss_s_text = F.binary_cross_entropy_with_logits(logits_s, torch.ones_like(logits_s))
        loss_r_text = F.binary_cross_entropy_with_logits(logits_r, torch.zeros_like(logits_r))
        loss_branch = loss_s_text + self.residual_negative_weight * loss_r_text
        return {
            "loss_branch": loss_branch,
            "loss_branch_s_text": loss_s_text,
            "loss_branch_r_text": loss_r_text,
            "branch_sim_s_text": sim_s_text.detach().mean(),
            "branch_sim_r_text": sim_r_text.detach().mean(),
            "branch_gap_s_minus_r": (sim_s_text.detach() - sim_r_text.detach()).mean(),
        }


class GateMapStats(nn.Module):
    """Scalar diagnostics for the sigmoid gate map."""

    def forward(self, sigmoid_map: Tensor) -> dict[str, Tensor]:
        gate = sigmoid_map.detach().float()
        return {
            "gate_mean": gate.mean(),
            "gate_std": gate.std(unbiased=False),
            "gate_min": gate.min(),
            "gate_max": gate.max(),
        }


class SPVDLoss(nn.Module):
    """OpenCLIP SigLIP alignment plus optional soft-cue losses."""

    expects_output_dict = True

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        branch_bce_weight: float = 0.0,
        branch_logit_scale: float = 5.0,
        residual_negative_weight: float = 0.25,
        detach_text_for_residual: bool = True,
        residual_variance_weight: float = 0.0,
        residual_variance_gamma: float = 1.0,
        align_weight: float = 1.0,
        global_align_weight: float = 1.0,
        caption_align_weight: float = 1.0,
        loss_dist_impl: str | None = None,
        debug_finite_checks: bool = False,
    ) -> None:
        super().__init__()
        self.align_loss = OpenCLIPSigLipLoss(
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            dist_impl=loss_dist_impl,
        )
        self.align_weight = float(align_weight)
        self.global_align_weight = float(global_align_weight)
        self.caption_align_weight = float(caption_align_weight)
        self.branch_bce_weight = float(branch_bce_weight)
        self.residual_variance_weight = float(residual_variance_weight)
        self.branch_bce = BranchBCELoss(
            logit_scale=branch_logit_scale,
            residual_negative_weight=residual_negative_weight,
            detach_text_for_residual=detach_text_for_residual,
        )
        self.residual_variance = ResidualVarianceLoss(gamma=residual_variance_gamma)
        self.gate_stats = GateMapStats()
        self.debug_finite_checks = bool(debug_finite_checks)

    def forward(self, outputs: Mapping[str, Any] | tuple[Tensor, ...]) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(outputs, Mapping):
            outputs = {
                "image_features": outputs[0],
                "text_features": outputs[1],
                "logit_scale": outputs[2],
            }

        image_features = _first_tensor(outputs, ("shared_visual_features", "image_features"))
        text_features = _first_tensor(outputs, ("text_features",))
        caption_visual_features = _first_tensor(outputs, ("caption_shared_visual_features",))
        caption_text_features = _first_tensor(outputs, ("caption_text_features",))
        logit_scale = _first_tensor(outputs, ("logit_scale",))
        logit_bias = _first_tensor(outputs, ("logit_bias",))
        if logit_scale is None:
            raise KeyError("SPVDLoss requires logit_scale.")

        if caption_visual_features is None and image_features is not None and image_features.ndim == 3:
            caption_visual_features = image_features

        normalized_image_features = F.normalize(image_features, dim=-1) if image_features is not None else None
        align_terms: list[tuple[float, Tensor]] = []
        zero_reference = normalized_image_features
        if zero_reference is None:
            zero_reference = caption_visual_features if caption_visual_features is not None else caption_text_features

        if (
            image_features is not None
            and text_features is not None
            and image_features.ndim == 2
            and text_features.ndim == 2
            and self.global_align_weight != 0.0
        ):
            align_global_image = F.normalize(image_features.float(), dim=-1)
            align_global_text = F.normalize(text_features.float(), dim=-1)
            loss_align_global = self.align_loss(align_global_image, align_global_text, logit_scale, logit_bias=logit_bias)
            align_terms.append((self.global_align_weight, loss_align_global))
        else:
            loss_align_global = _zero(zero_reference)

        if caption_visual_features is not None and caption_text_features is not None:
            if caption_visual_features.shape != caption_text_features.shape:
                raise ValueError(
                    "caption_shared_visual_features and caption_text_features must have the same shape "
                    f"for caption-level sigmoid loss, got {tuple(caption_visual_features.shape)} and "
                    f"{tuple(caption_text_features.shape)}."
                )
            align_image_features = F.normalize(
                caption_visual_features.reshape(-1, caption_visual_features.shape[-1]).float(),
                dim=-1,
            )
            align_text_features = F.normalize(
                caption_text_features.reshape(-1, caption_text_features.shape[-1]).float(),
                dim=-1,
            )
            if self.caption_align_weight != 0.0:
                loss_align_caption = self.align_loss(align_image_features, align_text_features, logit_scale, logit_bias=logit_bias)
                align_terms.append((self.caption_align_weight, loss_align_caption))
            else:
                loss_align_caption = _zero(align_image_features)
        else:
            loss_align_caption = _zero(zero_reference)
            align_image_features = image_features if image_features is not None else zero_reference

        if not align_terms:
            raise KeyError(
                "SPVDLoss requires shared_visual_features/text_features or "
                "caption_shared_visual_features/caption_text_features for sigmoid alignment."
            )
        align_weight_sum = sum(weight for weight, _ in align_terms)
        loss_align = sum(weight * term for weight, term in align_terms) / align_weight_sum

        zero = _zero(align_image_features)
        gate_logits = _first_tensor(outputs, ("gate_logits",))
        sigmoid_map = _first_tensor(outputs, ("sigmoid_map",))
        residual_map = _first_tensor(outputs, ("residual_map",))
        residual_features = _first_tensor(outputs, ("residual_visual_features",))
        has_decomp = sigmoid_map is not None or gate_logits is not None or residual_features is not None
        if has_decomp and (sigmoid_map is None or residual_map is None or gate_logits is None or residual_features is None):
            raise KeyError("Sigmoid decomposition loss requires sigmoid_map, residual_map, gate_logits, and residual_visual_features.")

        if self.debug_finite_checks:
            assert_finite_tensor("gate_logits", gate_logits, enabled=True)
            assert_finite_tensor("sigmoid_map", sigmoid_map, enabled=True)
            assert_finite_tensor("residual_map", residual_map, enabled=True)
            assert_finite_tensor("caption_shared_visual_features", caption_visual_features, enabled=True)
            assert_finite_tensor("shared_visual_features", image_features, enabled=True)
            assert_finite_tensor("residual_visual_features", residual_features, enabled=True)

        branch_text_features = text_features
        if (
            image_features is not None
            and image_features.ndim == 3
            and caption_text_features is not None
            and caption_text_features.shape[:-1] == image_features.shape[:-1]
        ):
            branch_text_features = caption_text_features

        if has_decomp and image_features is not None and residual_features is not None and branch_text_features is not None:
            branch_terms = {
                key: value.to(align_image_features.device)
                for key, value in self.branch_bce(image_features, residual_features, branch_text_features).items()
            }
        else:
            branch_terms = {
                "loss_branch": zero,
                "loss_branch_s_text": zero,
                "loss_branch_r_text": zero,
                "branch_sim_s_text": zero,
                "branch_sim_r_text": zero,
                "branch_gap_s_minus_r": zero,
            }

        if has_decomp and residual_features is not None:
            loss_residual_variance = self.residual_variance(residual_features).to(align_image_features.device)
        else:
            loss_residual_variance = zero

        if sigmoid_map is not None:
            gate_terms = {
                key: value.to(align_image_features.device)
                for key, value in self.gate_stats(sigmoid_map).items()
            }
        else:
            gate_terms = {
                "gate_mean": zero,
                "gate_std": zero,
                "gate_min": zero,
                "gate_max": zero,
            }

        total_loss = (
            self.align_weight * loss_align
            + self.branch_bce_weight * branch_terms["loss_branch"]
            + self.residual_variance_weight * loss_residual_variance
        )
        if self.debug_finite_checks:
            assert_finite_tensor("loss_align", loss_align, enabled=True)
            assert_finite_tensor("loss_branch", branch_terms["loss_branch"], enabled=True)
            assert_finite_tensor("loss_residual_variance", loss_residual_variance, enabled=True)
            assert_finite_tensor("total_loss", total_loss, enabled=True)
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_align": loss_align.detach(),
            "loss_sigmoid": loss_align.detach(),
            "loss_align_global": loss_align_global.detach(),
            "loss_align_caption": loss_align_caption.detach(),
            "loss_branch": branch_terms["loss_branch"].detach(),
            "loss_branch_s_text": branch_terms["loss_branch_s_text"].detach(),
            "loss_branch_r_text": branch_terms["loss_branch_r_text"].detach(),
            "branch_sim_s_text": branch_terms["branch_sim_s_text"].detach(),
            "branch_sim_r_text": branch_terms["branch_sim_r_text"].detach(),
            "branch_gap_s_minus_r": branch_terms["branch_gap_s_minus_r"].detach(),
            "loss_residual_variance": loss_residual_variance.detach(),
            "gate_mean": gate_terms["gate_mean"].detach(),
            "gate_std": gate_terms["gate_std"].detach(),
            "gate_min": gate_terms["gate_min"].detach(),
            "gate_max": gate_terms["gate_max"].detach(),
            "logit_scale": logit_scale.detach(),
        }
