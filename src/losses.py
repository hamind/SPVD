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


def _linear_warmup_weight(start: float, target: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(target)
    alpha = min(float(step) / float(warmup_steps), 1.0)
    return float(start) + (float(target) - float(start)) * alpha


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


def _caption_zero_stats(reference: Tensor, valid_negative_fraction: float = 0.0) -> dict[str, Tensor]:
    zero = reference.detach().new_zeros((), dtype=torch.float32)
    return {
        "caption_primary_loss": zero,
        "caption_pos_loss": zero,
        "caption_neg_loss": zero,
        "caption_pos_sim_mean": zero,
        "caption_neg_sim_mean": zero,
        "caption_margin_sim": zero,
        "caption_pos_logit_mean": zero,
        "caption_neg_logit_mean": zero,
        "caption_margin_logit": zero,
        "caption_num_positive_pairs": zero,
        "caption_num_negative_pairs": zero,
        "caption_num_masked_same_image_pairs": zero,
        "caption_num_valid_pairs": zero,
        "caption_num_total_pairs": zero,
        "caption_valid_negative_fraction": zero.new_tensor(float(valid_negative_fraction), dtype=torch.float32),
        "caption_masked_same_image_pairs": zero,
        "caption_num_pairs": zero,
        "caption_region_positive_loss": zero,
        "caption_region_negative_loss": zero,
        "caption_region_same_image_loss": zero,
        "caption_region_positive_weight_sum": zero,
        "caption_region_negative_weight_sum": zero,
        "caption_region_positive_active_fraction": zero,
        "caption_region_negative_active_fraction": zero,
        "caption_region_active_fraction": zero,
        "caption_region_overlap_mean": zero,
        "caption_region_overlap_min": zero,
        "caption_region_overlap_max": zero,
        "caption_region_overlap_std": zero,
        "caption_region_pos_weight_effective": zero,
        "caption_region_neg_weight_effective": zero,
        "caption_region_weight_effective": zero,
        "caption_region_warmup_alpha": zero,
        "caption_region_weight_mean": zero,
        "caption_region_weight_sum": zero,
        "caption_same_image_mode_code": zero,
    }


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


class MaskedCaptionSigLipLoss(nn.Module):
    """Sigmoid/SigLIP-style caption-level alignment loss for [B, K, D] features."""

    def __init__(
        self,
        mask_same_image: bool = True,
        caption_same_image_mode: str = "ignore",
        caption_region_pos_weight: float = 0.05,
        caption_region_neg_weight: float = 0.02,
        caption_region_start_weight: float = 0.0,
        caption_region_warmup_steps: int = 0,
        caption_region_pos_min_overlap: float = 0.3,
        caption_region_neg_max_overlap: float = 0.1,
        caption_region_overlap_gamma: float = 1.0,
        caption_region_detach_overlap: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.mask_same_image = bool(mask_same_image)
        self.caption_same_image_mode = str(caption_same_image_mode)
        if self.caption_same_image_mode not in {"ignore", "region_soft_positive", "region_soft_signed"}:
            raise ValueError(
                "caption_same_image_mode must be one of "
                "'ignore', 'region_soft_positive', or 'region_soft_signed', "
                f"got {self.caption_same_image_mode!r}."
            )
        self.caption_region_pos_weight = float(caption_region_pos_weight)
        self.caption_region_neg_weight = float(caption_region_neg_weight)
        self.caption_region_start_weight = float(caption_region_start_weight)
        self.caption_region_warmup_steps = int(caption_region_warmup_steps)
        self.caption_region_pos_min_overlap = float(caption_region_pos_min_overlap)
        self.caption_region_neg_max_overlap = float(caption_region_neg_max_overlap)
        if self.caption_region_neg_max_overlap >= self.caption_region_pos_min_overlap:
            raise ValueError(
                "caption_region_neg_max_overlap must be smaller than "
                "caption_region_pos_min_overlap, got "
                f"{self.caption_region_neg_max_overlap} >= {self.caption_region_pos_min_overlap}."
            )
        self.caption_region_overlap_gamma = float(caption_region_overlap_gamma)
        self.caption_region_detach_overlap = bool(caption_region_detach_overlap)
        self.eps = float(eps)
        self.base_loss = OpenCLIPSigLipLoss(rank=0, world_size=1)

    def forward(
        self,
        caption_visual_features: Tensor,
        caption_text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        sigmoid_map: Tensor | None = None,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if caption_visual_features.shape != caption_text_features.shape:
            raise ValueError(
                "caption_visual_features and caption_text_features must have the same shape, "
                f"got {tuple(caption_visual_features.shape)} and {tuple(caption_text_features.shape)}."
            )
        if caption_visual_features.ndim == 2:
            image = F.normalize(caption_visual_features.float(), dim=-1)
            text = F.normalize(caption_text_features.float(), dim=-1)
            loss = self.base_loss(image, text, logit_scale, logit_bias=logit_bias)
            stats = _caption_zero_stats(image, valid_negative_fraction=1.0)
            stats["caption_primary_loss"] = loss.detach().float()
            stats["caption_same_image_mode_code"] = image.detach().new_tensor(
                self._mode_code(), dtype=torch.float32
            )
            return loss, stats
        if caption_visual_features.ndim != 3:
            raise ValueError(
                "MaskedCaptionSigLipLoss expects [B, K, D] or [B, D] features, "
                f"got shape={tuple(caption_visual_features.shape)}."
            )

        bsz, num_captions, dim = caption_visual_features.shape
        image = F.normalize(caption_visual_features.float().reshape(bsz * num_captions, dim), dim=-1)
        text = F.normalize(caption_text_features.float().reshape(bsz * num_captions, dim), dim=-1)
        similarities = image @ text.t()
        logits = logit_scale * similarities
        if logit_bias is not None:
            logits = logits + logit_bias

        num_items = bsz * num_captions
        flat_ids = torch.arange(num_items, device=image.device)
        image_ids = torch.arange(bsz, device=image.device).repeat_interleave(num_captions)
        same_pair = flat_ids[:, None].eq(flat_ids[None, :])
        same_image = image_ids[:, None].eq(image_ids[None, :])
        same_image_different_caption = same_image & (~same_pair)
        valid_mask = ~same_image_different_caption

        labels = torch.full_like(logits, -1.0)
        labels.masked_fill_(same_pair, 1.0)
        loss_matrix = -F.logsigmoid(labels * logits)
        loss_matrix = loss_matrix.masked_fill(~valid_mask, 0.0)
        valid_per_row = valid_mask.float().sum(dim=1).clamp_min(1.0)
        loss_primary = (loss_matrix.sum(dim=1) / valid_per_row).mean()

        caption_num_pairs = num_items * num_items
        caption_masked_same_image_pairs = bsz * num_captions * (num_captions - 1)
        caption_num_valid_pairs = caption_num_pairs - caption_masked_same_image_pairs
        total_negative_count = max(caption_num_pairs - num_items, 1)
        valid_negative_count = caption_num_valid_pairs - num_items
        positive_mask = same_pair
        negative_mask = valid_mask & (~same_pair)
        caption_stats = self._primary_stats(
            image,
            similarities,
            logits,
            loss_matrix,
            positive_mask,
            negative_mask,
            caption_num_pairs,
            caption_masked_same_image_pairs,
            caption_num_valid_pairs,
            valid_negative_count,
            total_negative_count,
            loss_primary,
        )

        loss_region_same_image, region_stats = self._region_same_image_loss(
            logits=logits,
            sigmoid_map=sigmoid_map,
            bsz=bsz,
            num_captions=num_captions,
            global_step=global_step,
        )
        caption_stats.update(region_stats)
        loss = loss_primary + loss_region_same_image
        return loss, caption_stats

    def _mode_code(self) -> float:
        if self.caption_same_image_mode == "region_soft_positive":
            return 1.0
        if self.caption_same_image_mode == "region_soft_signed":
            return 2.0
        return 0.0

    def _scalar(self, reference: Tensor, value: float) -> Tensor:
        return reference.detach().new_tensor(float(value), dtype=torch.float32)

    def _masked_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any().item()):
            return values.detach().new_zeros((), dtype=torch.float32)
        return values.masked_select(mask).detach().float().mean()

    def _primary_stats(
        self,
        reference: Tensor,
        similarities: Tensor,
        logits: Tensor,
        loss_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        caption_num_pairs: int,
        caption_masked_same_image_pairs: int,
        caption_num_valid_pairs: int,
        valid_negative_count: int,
        total_negative_count: int,
        loss_primary: Tensor,
    ) -> dict[str, Tensor]:
        pos_sim = self._masked_mean(similarities, positive_mask)
        neg_sim = self._masked_mean(similarities, negative_mask)
        pos_logit = self._masked_mean(logits, positive_mask)
        neg_logit = self._masked_mean(logits, negative_mask)
        stats = {
            "caption_primary_loss": loss_primary.detach().float(),
            "caption_pos_loss": self._masked_mean(loss_matrix, positive_mask),
            "caption_neg_loss": self._masked_mean(loss_matrix, negative_mask),
            "caption_pos_sim_mean": pos_sim,
            "caption_neg_sim_mean": neg_sim,
            "caption_margin_sim": (pos_sim - neg_sim).detach().float(),
            "caption_pos_logit_mean": pos_logit,
            "caption_neg_logit_mean": neg_logit,
            "caption_margin_logit": (pos_logit - neg_logit).detach().float(),
            "caption_num_positive_pairs": self._scalar(reference, int(positive_mask.sum().item())),
            "caption_num_negative_pairs": self._scalar(reference, int(negative_mask.sum().item())),
            "caption_num_masked_same_image_pairs": self._scalar(reference, caption_masked_same_image_pairs),
            "caption_num_valid_pairs": self._scalar(reference, caption_num_valid_pairs),
            "caption_num_total_pairs": self._scalar(reference, caption_num_pairs),
            "caption_valid_negative_fraction": self._scalar(
                reference, float(valid_negative_count) / float(total_negative_count)
            ),
            "caption_masked_same_image_pairs": self._scalar(reference, caption_masked_same_image_pairs),
            "caption_num_pairs": self._scalar(reference, caption_num_pairs),
        }
        return stats

    def _region_zero_stats(self, reference: Tensor) -> dict[str, Tensor]:
        zero = reference.detach().new_zeros((), dtype=torch.float32)
        return {
            "caption_region_positive_loss": zero,
            "caption_region_negative_loss": zero,
            "caption_region_same_image_loss": zero,
            "caption_region_positive_weight_sum": zero,
            "caption_region_negative_weight_sum": zero,
            "caption_region_positive_active_fraction": zero,
            "caption_region_negative_active_fraction": zero,
            "caption_region_active_fraction": zero,
            "caption_region_overlap_mean": zero,
            "caption_region_overlap_min": zero,
            "caption_region_overlap_max": zero,
            "caption_region_overlap_std": zero,
            "caption_region_pos_weight_effective": zero,
            "caption_region_neg_weight_effective": zero,
            "caption_region_weight_effective": zero,
            "caption_region_warmup_alpha": zero,
            "caption_region_weight_mean": zero,
            "caption_region_weight_sum": zero,
            "caption_same_image_mode_code": self._scalar(reference, self._mode_code()),
        }

    def _prepare_sigmoid_map(self, sigmoid_map: Tensor, bsz: int, num_captions: int) -> Tensor:
        if sigmoid_map.ndim == 3:
            sigmoid_map = sigmoid_map.unsqueeze(1)
        if sigmoid_map.ndim != 4:
            raise ValueError(
                "region_soft_positive requires sigmoid_map with shape [B, K, S, N] or [B, S, N], "
                f"got shape={tuple(sigmoid_map.shape)}."
            )
        if sigmoid_map.shape[0] != bsz:
            raise ValueError(f"sigmoid_map batch size must be {bsz}, got {sigmoid_map.shape[0]}.")
        if sigmoid_map.shape[1] != num_captions:
            raise ValueError(
                "sigmoid_map caption dimension must match caption features, "
                f"got {sigmoid_map.shape[1]} and {num_captions}."
            )
        return sigmoid_map

    def _region_same_image_loss(
        self,
        logits: Tensor,
        sigmoid_map: Tensor | None,
        bsz: int,
        num_captions: int,
        global_step: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.caption_same_image_mode == "ignore":
            return logits.new_zeros(()), self._region_zero_stats(logits)
        if sigmoid_map is None:
            raise KeyError("region_soft_positive requires sigmoid_map for coverage overlap.")

        sigmoid_map = self._prepare_sigmoid_map(sigmoid_map, bsz, num_captions)
        coverage = sigmoid_map.float().mean(dim=2)
        coverage = coverage.clamp_min(0.0)
        coverage = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        overlap = torch.minimum(coverage[:, :, None, :], coverage[:, None, :, :]).sum(dim=-1)
        if self.caption_region_detach_overlap:
            overlap = overlap.detach()

        eye = torch.eye(num_captions, device=overlap.device, dtype=torch.bool).unsqueeze(0)
        cross_caption_mask = ~eye
        positive_region_mask = (overlap >= self.caption_region_pos_min_overlap) & cross_caption_mask
        pos_region_weights = overlap.clamp_min(0.0).pow(self.caption_region_overlap_gamma)
        pos_region_weights = pos_region_weights.masked_fill(~positive_region_mask, 0.0)

        batch_ids = torch.arange(bsz, device=logits.device)
        same_image_logits = logits.reshape(bsz, num_captions, bsz, num_captions)[batch_ids, :, batch_ids, :]
        pos_region_loss_matrix = -F.logsigmoid(same_image_logits)
        pos_weight_sum = pos_region_weights.sum()
        loss_region_positive = torch.where(
            pos_weight_sum > 0,
            (pos_region_loss_matrix * pos_region_weights).sum() / pos_weight_sum.clamp_min(self.eps),
            logits.new_zeros(()),
        )

        if self.caption_same_image_mode == "region_soft_signed":
            negative_region_mask = (overlap <= self.caption_region_neg_max_overlap) & cross_caption_mask
            neg_region_weights = (1.0 - overlap).clamp_min(0.0).pow(self.caption_region_overlap_gamma)
            neg_region_weights = neg_region_weights.masked_fill(~negative_region_mask, 0.0)
            neg_region_loss_matrix = -F.logsigmoid(-same_image_logits)
            neg_weight_sum = neg_region_weights.sum()
            loss_region_negative = torch.where(
                neg_weight_sum > 0,
                (neg_region_loss_matrix * neg_region_weights).sum() / neg_weight_sum.clamp_min(self.eps),
                logits.new_zeros(()),
            )
        else:
            negative_region_mask = torch.zeros_like(cross_caption_mask)
            neg_region_weights = overlap.detach().new_zeros(overlap.shape)
            neg_weight_sum = logits.new_zeros(())
            loss_region_negative = logits.new_zeros(())

        warmup_alpha = _linear_warmup_weight(
            self.caption_region_start_weight,
            1.0,
            global_step,
            self.caption_region_warmup_steps,
        )
        pos_weight_effective = warmup_alpha * self.caption_region_pos_weight
        neg_weight_effective = (
            warmup_alpha * self.caption_region_neg_weight
            if self.caption_same_image_mode == "region_soft_signed"
            else 0.0
        )
        loss_region = pos_weight_effective * loss_region_positive + neg_weight_effective * loss_region_negative

        num_cross_caption_pairs = max(bsz * num_captions * (num_captions - 1), 1)
        active_fraction = float((positive_region_mask | negative_region_mask).sum().item()) / float(num_cross_caption_pairs)
        positive_active_fraction = float(positive_region_mask.sum().item()) / float(num_cross_caption_pairs)
        negative_active_fraction = float(negative_region_mask.sum().item()) / float(num_cross_caption_pairs)
        weight_sum = pos_weight_sum.detach().float() + neg_weight_sum.detach().float()
        weight_mean = weight_sum / float(num_cross_caption_pairs)
        cross_overlap = overlap.masked_select(cross_caption_mask).detach().float()
        if cross_overlap.numel() == 0:
            cross_overlap = logits.detach().new_zeros((1,), dtype=torch.float32)
        stats = {
            "caption_region_positive_loss": loss_region_positive.detach().float(),
            "caption_region_negative_loss": loss_region_negative.detach().float(),
            "caption_region_same_image_loss": loss_region.detach().float(),
            "caption_region_positive_weight_sum": pos_weight_sum.detach().float(),
            "caption_region_negative_weight_sum": neg_weight_sum.detach().float(),
            "caption_region_positive_active_fraction": self._scalar(logits, positive_active_fraction),
            "caption_region_negative_active_fraction": self._scalar(logits, negative_active_fraction),
            "caption_region_active_fraction": self._scalar(logits, active_fraction),
            "caption_region_overlap_mean": cross_overlap.mean(),
            "caption_region_overlap_min": cross_overlap.min(),
            "caption_region_overlap_max": cross_overlap.max(),
            "caption_region_overlap_std": cross_overlap.std(unbiased=False),
            "caption_region_pos_weight_effective": self._scalar(logits, pos_weight_effective),
            "caption_region_neg_weight_effective": self._scalar(logits, neg_weight_effective),
            "caption_region_weight_effective": self._scalar(logits, pos_weight_effective + neg_weight_effective),
            "caption_region_warmup_alpha": self._scalar(logits, warmup_alpha),
            "caption_region_weight_mean": weight_mean.detach().float(),
            "caption_region_weight_sum": weight_sum.detach().float(),
            "caption_same_image_mode_code": self._scalar(logits, self._mode_code()),
        }
        return loss_region, stats


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
        caption_loss_impl: str = "masked_sigmoid",
        caption_mask_same_image: bool = True,
        caption_same_image_mode: str = "ignore",
        caption_region_pos_weight: float = 0.05,
        caption_region_neg_weight: float = 0.02,
        caption_region_start_weight: float = 0.0,
        caption_region_warmup_steps: int = 0,
        caption_region_pos_min_overlap: float = 0.3,
        caption_region_neg_max_overlap: float = 0.1,
        caption_region_overlap_gamma: float = 1.0,
        caption_region_detach_overlap: bool = True,
        branch_bce_start_weight: float = 0.0,
        residual_variance_start_weight: float = 0.0,
        branch_bce_warmup_steps: int = 0,
        residual_variance_warmup_steps: int = 0,
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
        self.caption_loss_impl = str(caption_loss_impl)
        self.caption_mask_same_image = bool(caption_mask_same_image)
        self.masked_caption_loss = MaskedCaptionSigLipLoss(
            mask_same_image=self.caption_mask_same_image,
            caption_same_image_mode=caption_same_image_mode,
            caption_region_pos_weight=caption_region_pos_weight,
            caption_region_neg_weight=caption_region_neg_weight,
            caption_region_start_weight=caption_region_start_weight,
            caption_region_warmup_steps=caption_region_warmup_steps,
            caption_region_pos_min_overlap=caption_region_pos_min_overlap,
            caption_region_neg_max_overlap=caption_region_neg_max_overlap,
            caption_region_overlap_gamma=caption_region_overlap_gamma,
            caption_region_detach_overlap=caption_region_detach_overlap,
        )
        self.branch_bce_weight = float(branch_bce_weight)
        self.residual_variance_weight = float(residual_variance_weight)
        self.branch_bce_start_weight = float(branch_bce_start_weight)
        self.residual_variance_start_weight = float(residual_variance_start_weight)
        self.branch_bce_warmup_steps = int(branch_bce_warmup_steps)
        self.residual_variance_warmup_steps = int(residual_variance_warmup_steps)
        self.global_step = 0
        self.branch_bce = BranchBCELoss(
            logit_scale=branch_logit_scale,
            residual_negative_weight=residual_negative_weight,
            detach_text_for_residual=detach_text_for_residual,
        )
        self.residual_variance = ResidualVarianceLoss(gamma=residual_variance_gamma)
        self.gate_stats = GateMapStats()
        self.debug_finite_checks = bool(debug_finite_checks)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

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
        sigmoid_map = _first_tensor(outputs, ("sigmoid_map",))
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

        caption_stats: dict[str, Tensor] = {}
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
            if self.caption_align_weight != 0.0:
                if self.caption_loss_impl == "masked_sigmoid":
                    loss_align_caption, caption_stats = self.masked_caption_loss(
                        caption_visual_features,
                        caption_text_features,
                        logit_scale,
                        logit_bias=logit_bias,
                        sigmoid_map=sigmoid_map,
                        global_step=self.global_step,
                    )
                elif self.caption_loss_impl == "openclip_siglip":
                    align_text_features = F.normalize(
                        caption_text_features.reshape(-1, caption_text_features.shape[-1]).float(),
                        dim=-1,
                    )
                    loss_align_caption = self.align_loss(align_image_features, align_text_features, logit_scale, logit_bias=logit_bias)
                    caption_stats = _caption_zero_stats(align_image_features, valid_negative_fraction=1.0)
                else:
                    raise ValueError(
                        "Unsupported caption_loss_impl: "
                        f"{self.caption_loss_impl!r}. Expected 'masked_sigmoid' or 'openclip_siglip'."
                    )
                align_terms.append((self.caption_align_weight, loss_align_caption))
            else:
                loss_align_caption = _zero(align_image_features)
                caption_stats = _caption_zero_stats(align_image_features)
        else:
            loss_align_caption = _zero(zero_reference)
            align_image_features = image_features if image_features is not None else zero_reference
            if align_image_features is not None:
                caption_stats = _caption_zero_stats(align_image_features)

        if not align_terms:
            raise KeyError(
                "SPVDLoss requires shared_visual_features/text_features or "
                "caption_shared_visual_features/caption_text_features for sigmoid alignment."
            )
        align_weight_sum = sum(weight for weight, _ in align_terms)
        loss_align = sum(weight * term for weight, term in align_terms) / align_weight_sum

        zero = _zero(align_image_features)
        if not caption_stats:
            caption_stats = _caption_zero_stats(zero)

        gate_logits = _first_tensor(outputs, ("gate_logits",))
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

        branch_weight = _linear_warmup_weight(
            self.branch_bce_start_weight,
            self.branch_bce_weight,
            self.global_step,
            self.branch_bce_warmup_steps,
        )
        residual_variance_weight = _linear_warmup_weight(
            self.residual_variance_start_weight,
            self.residual_variance_weight,
            self.global_step,
            self.residual_variance_warmup_steps,
        )
        total_loss = (
            self.align_weight * loss_align
            + branch_weight * branch_terms["loss_branch"]
            + residual_variance_weight * loss_residual_variance
        )
        if self.debug_finite_checks:
            assert_finite_tensor("loss_align", loss_align, enabled=True)
            assert_finite_tensor("loss_branch", branch_terms["loss_branch"], enabled=True)
            assert_finite_tensor("loss_residual_variance", loss_residual_variance, enabled=True)
            assert_finite_tensor("total_loss", total_loss, enabled=True)
        logit_scale_exp = logit_scale.detach().float()
        logit_scale_log = logit_scale_exp.clamp_min(1.0e-12).log()
        logit_bias_value = logit_bias.detach().float() if logit_bias is not None else zero
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_align": loss_align.detach(),
            "loss_sigmoid": loss_align.detach(),
            "loss_align_global": loss_align_global.detach(),
            "loss_align_caption": loss_align_caption.detach(),
            "global_align_enabled": zero.new_tensor(float(self.global_align_weight != 0.0)),
            "caption_align_enabled": zero.new_tensor(float(self.caption_align_weight != 0.0)),
            **{key: value.detach() for key, value in caption_stats.items()},
            "loss_branch": branch_terms["loss_branch"].detach(),
            "loss_branch_s_text": branch_terms["loss_branch_s_text"].detach(),
            "loss_branch_r_text": branch_terms["loss_branch_r_text"].detach(),
            "branch_sim_s_text": branch_terms["branch_sim_s_text"].detach(),
            "branch_sim_r_text": branch_terms["branch_sim_r_text"].detach(),
            "branch_gap_s_minus_r": branch_terms["branch_gap_s_minus_r"].detach(),
            "loss_residual_variance": loss_residual_variance.detach(),
            "loss_branch_weight_effective": zero.new_tensor(branch_weight),
            "loss_residual_variance_weight_effective": zero.new_tensor(residual_variance_weight),
            "gate_mean": gate_terms["gate_mean"].detach(),
            "gate_std": gate_terms["gate_std"].detach(),
            "gate_min": gate_terms["gate_min"].detach(),
            "gate_max": gate_terms["gate_max"].detach(),
            "logit_scale": logit_scale.detach(),
            "logit_scale_exp": logit_scale_exp.detach(),
            "logit_scale_log": logit_scale_log.detach(),
            "logit_bias": logit_bias_value.detach(),
        }
