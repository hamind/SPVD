"""SPVD loss functions with OpenCLIP-style contrastive plumbing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from open_clip.loss import (
    SigLipLoss as OpenCLIPSigLipLoss,
    neighbour_exchange_bidir_with_grad,
    neighbour_exchange_with_grad,
)
from torch import Tensor, nn

try:
    import torch.distributed.nn
    from torch import distributed as dist

    _HAS_DISTRIBUTED = True
except ImportError:
    _HAS_DISTRIBUTED = False

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
        "caption_ddp_negative_loss": zero,
        "caption_num_ignored_pairs": zero,
        "caption_same_image_pair_count": zero,
        "caption_same_image_iou_mean": zero,
        "caption_same_image_iou_min": zero,
        "caption_same_image_iou_max": zero,
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
    """DDP-compatible simplified sigmoid loss for SPVD multi-sub-caption training.

    Inputs:
        caption_visual_features: [B, K, D]
            Caption-conditioned semantic visual features.
        caption_text_features: [B, K, D]
            Corresponding sub-caption text features.
        sigmoid_map: [B, K, S, N]
            Caption-conditioned visual gate maps. Required only when
            caption_same_image_mode is "positive" or "signed".

    Label rule on local rank:
        same image, same sub-caption:
            label = +1
        different image:
            label = -1
        same image, different sub-caption:
            mode = "ignore":
                label = 0
            mode = "positive":
                label = binary_IoU if binary_IoU > 0 else 0
            mode = "signed":
                label = binary_IoU if binary_IoU > 0 else -1

    DDP rule:
        Text features from other ranks are used as negative-only samples,
        following OpenCLIP SigLipLoss.
    """

    def __init__(
        self,
        caption_same_image_mode: str = "ignore",
        gate_binarize_threshold: float = 0.5,
        eps: float = 1.0e-6,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: str | None = None,
    ) -> None:
        super().__init__()

        mode = str(caption_same_image_mode).lower()
        aliases = {
            "ignore": "ignore",
            "none": "ignore",
            "positive": "positive",
            "pos": "positive",
            "region_soft_positive": "positive",
            "soft_positive": "positive",
            "signed": "signed",
            "positive_negative": "signed",
            "pos_neg": "signed",
            "region_soft_signed": "signed",
            "soft_signed": "signed",
        }
        if mode not in aliases:
            raise ValueError(
                "caption_same_image_mode must be one of "
                "'ignore', 'positive', 'signed', "
                "'region_soft_positive', or 'region_soft_signed', "
                f"got {caption_same_image_mode!r}."
            )

        self.caption_same_image_mode = aliases[mode]
        self.gate_binarize_threshold = float(gate_binarize_threshold)
        self.eps = float(eps)

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dist_impl = dist_impl or "bidir"
        if self.dist_impl not in {"bidir", "shift", "reduce", "gather"}:
            raise ValueError(
                "dist_impl must be one of 'bidir', 'shift', 'reduce', or 'gather', "
                f"got {self.dist_impl!r}."
            )

    def forward(
        self,
        caption_visual_features: Tensor,
        caption_text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        sigmoid_map: Tensor | None = None,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if caption_visual_features.ndim != 3:
            raise ValueError(
                "MaskedCaptionSigLipLoss expects caption_visual_features with "
                f"shape [B, K, D], got {tuple(caption_visual_features.shape)}."
            )
        if caption_text_features.ndim != 3:
            raise ValueError(
                "MaskedCaptionSigLipLoss expects caption_text_features with "
                f"shape [B, K, D], got {tuple(caption_text_features.shape)}."
            )
        if caption_visual_features.shape != caption_text_features.shape:
            raise ValueError(
                "caption_visual_features and caption_text_features must have "
                "the same shape, got "
                f"{tuple(caption_visual_features.shape)} and "
                f"{tuple(caption_text_features.shape)}."
            )

        bsz, num_captions, dim = caption_visual_features.shape
        num_items = bsz * num_captions

        image = F.normalize(
            caption_visual_features.float().reshape(num_items, dim),
            dim=-1,
        )
        text = F.normalize(
            caption_text_features.float().reshape(num_items, dim),
            dim=-1,
        )

        local_loss, stats = self._local_loss(
            image_features=image,
            text_features=text,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            bsz=bsz,
            num_captions=num_captions,
            sigmoid_map=sigmoid_map,
        )

        loss = local_loss
        remote_negative_loss = image.detach().new_zeros((), dtype=torch.float32)
        remote_negative_pairs = 0

        if self.world_size > 1:
            if not _HAS_DISTRIBUTED:
                raise RuntimeError("torch.distributed is required when world_size > 1.")

            remote_negative_loss, remote_negative_pairs = self._ddp_negative_loss(
                image_features=image,
                text_features=text,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )
            loss = loss + remote_negative_loss

        stats["caption_ddp_negative_loss"] = remote_negative_loss.detach().float()
        stats["caption_ddp_negative_pairs"] = self._scalar(image, remote_negative_pairs)

        if remote_negative_pairs > 0:
            stats["caption_num_negative_pairs"] = (
                stats["caption_num_negative_pairs"] + self._scalar(image, remote_negative_pairs)
            )
            stats["caption_num_valid_pairs"] = (
                stats["caption_num_valid_pairs"] + self._scalar(image, remote_negative_pairs)
            )
            stats["caption_num_total_pairs"] = (
                stats["caption_num_total_pairs"] + self._scalar(image, remote_negative_pairs)
            )
            stats["caption_num_pairs"] = (
                stats["caption_num_pairs"] + self._scalar(image, remote_negative_pairs)
            )

            total_negative_count = max(
                int(stats["caption_num_total_pairs"].item())
                - int(stats["caption_num_positive_pairs"].item()),
                1,
            )
            stats["caption_valid_negative_fraction"] = self._scalar(
                image,
                float(stats["caption_num_negative_pairs"].item()) / float(total_negative_count),
            )

        return loss, stats

    def _mode_code(self) -> float:
        if self.caption_same_image_mode == "positive":
            return 1.0
        if self.caption_same_image_mode == "signed":
            return 2.0
        return 0.0

    def _scalar(self, reference: Tensor, value: float) -> Tensor:
        return reference.detach().new_tensor(float(value), dtype=torch.float32)

    def _masked_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any().item()):
            return values.detach().new_zeros((), dtype=torch.float32)
        return values.masked_select(mask).detach().float().mean()

    def _get_logits(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None,
    ) -> Tensor:
        logits = logit_scale * image_features @ text_features.t()
        if logit_bias is not None:
            logits = logits + logit_bias
        return logits

    def _local_loss(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None,
        bsz: int,
        num_captions: int,
        sigmoid_map: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self._get_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        labels, masks, iou_values = self._build_labels(
            logits=logits,
            bsz=bsz,
            num_captions=num_captions,
            sigmoid_map=sigmoid_map,
        )

        valid_mask = labels.ne(0.0)
        if not bool(valid_mask.any().item()):
            raise RuntimeError("Caption sigmoid loss has no valid local pairs.")

        loss_matrix = torch.zeros_like(logits)
        loss_matrix[valid_mask] = -F.logsigmoid(labels[valid_mask] * logits[valid_mask])

        # Same reduction style as OpenCLIP SigLipLoss:
        # sum over pairwise losses, divide by local image-side rows.
        loss = loss_matrix.sum() / image_features.shape[0]

        stats = self._stats(
            reference=image_features,
            similarities=image_features @ text_features.t(),
            logits=logits,
            loss_matrix=loss_matrix,
            labels=labels,
            masks=masks,
            iou_values=iou_values,
            loss=loss,
            bsz=bsz,
            num_captions=num_captions,
        )
        return loss, stats

    def _negative_only_loss(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None,
    ) -> Tensor:
        logits = self._get_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        # label = -1 for every remote pair:
        # -logsigmoid(-logit) == softplus(logit)
        return F.softplus(logits).sum() / image_features.shape[0]

    def _ddp_negative_loss(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None,
    ) -> tuple[Tensor, int]:
        loss = image_features.detach().new_zeros((), dtype=torch.float32)
        remote_pairs = 0

        if self.dist_impl == "bidir":
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size

            text_to_left = text_features
            text_to_right = text_features

            num_bidir, remainder = divmod(self.world_size - 1, 2)
            for _ in range(num_bidir):
                text_from_right, text_from_left = neighbour_exchange_bidir_with_grad(
                    left_rank,
                    right_rank,
                    text_to_left,
                    text_to_right,
                )

                loss = loss + self._negative_only_loss(
                    image_features=image_features,
                    text_features=text_from_right,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                )
                loss = loss + self._negative_only_loss(
                    image_features=image_features,
                    text_features=text_from_left,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                )

                remote_pairs += image_features.shape[0] * text_from_right.shape[0]
                remote_pairs += image_features.shape[0] * text_from_left.shape[0]

                text_to_right = text_from_right
                text_to_left = text_from_left

            if remainder:
                text_from_left = neighbour_exchange_with_grad(
                    left_rank,
                    right_rank,
                    text_to_right,
                )
                loss = loss + self._negative_only_loss(
                    image_features=image_features,
                    text_features=text_from_left,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                )
                remote_pairs += image_features.shape[0] * text_from_left.shape[0]

        elif self.dist_impl == "shift":
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size

            text_to_right = text_features
            for _ in range(self.world_size - 1):
                text_from_left = neighbour_exchange_with_grad(
                    left_rank,
                    right_rank,
                    text_to_right,
                )
                loss = loss + self._negative_only_loss(
                    image_features=image_features,
                    text_features=text_from_left,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                )
                remote_pairs += image_features.shape[0] * text_from_left.shape[0]
                text_to_right = text_from_left

        elif self.dist_impl == "reduce":
            for rank_idx in range(self.world_size):
                text_from_other = torch.distributed.nn.all_reduce(
                    text_features * float(self.rank == rank_idx),
                    dist.ReduceOp.SUM,
                )
                if rank_idx != self.rank:
                    loss = loss + self._negative_only_loss(
                        image_features=image_features,
                        text_features=text_from_other,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                    )
                    remote_pairs += image_features.shape[0] * text_from_other.shape[0]

        elif self.dist_impl == "gather":
            all_text = torch.distributed.nn.all_gather(text_features)
            for rank_idx, text_from_other in enumerate(all_text):
                if rank_idx != self.rank:
                    loss = loss + self._negative_only_loss(
                        image_features=image_features,
                        text_features=text_from_other,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                    )
                    remote_pairs += image_features.shape[0] * text_from_other.shape[0]

        else:
            raise RuntimeError(f"Unexpected dist_impl: {self.dist_impl}")

        return loss, int(remote_pairs)

    def _prepare_sigmoid_map(
        self,
        sigmoid_map: Tensor,
        bsz: int,
        num_captions: int,
    ) -> Tensor:
        if sigmoid_map.ndim != 4:
            raise ValueError(
                "sigmoid_map must have shape [B, K, S, N] for same-image "
                f"sub-caption supervision, got {tuple(sigmoid_map.shape)}."
            )
        if sigmoid_map.shape[0] != bsz:
            raise ValueError(
                f"sigmoid_map batch size must be {bsz}, got {sigmoid_map.shape[0]}."
            )
        if sigmoid_map.shape[1] != num_captions:
            raise ValueError(
                "sigmoid_map caption dimension must match caption features, "
                f"got {sigmoid_map.shape[1]} and {num_captions}."
            )
        return sigmoid_map

    def _compute_binary_iou(
        self,
        sigmoid_map: Tensor,
        bsz: int,
        num_captions: int,
    ) -> Tensor:
        """Compute token-level binary IoU between same-image sub-caption maps.

        Args:
            sigmoid_map: [B, K, S, N]

        Returns:
            binary_iou: [B, K, K]
        """
        sigmoid_map = self._prepare_sigmoid_map(sigmoid_map, bsz, num_captions)

        # Detach because the IoU is used only to construct labels.
        binary_map = (
            sigmoid_map.detach().float() > self.gate_binarize_threshold
        ).float()

        # [B, K, S, N] -> [B, K, N]
        # A visual token is covered by a sub-caption if any soft cue selects it.
        coverage = binary_map.max(dim=2).values

        intersection = torch.minimum(
            coverage[:, :, None, :],
            coverage[:, None, :, :],
        ).sum(dim=-1)

        union = torch.maximum(
            coverage[:, :, None, :],
            coverage[:, None, :, :],
        ).sum(dim=-1).clamp_min(self.eps)

        return (intersection / union).clamp(0.0, 1.0)

    def _expand_iou_to_flat(
        self,
        iou: Tensor,
        bsz: int,
        num_captions: int,
        device: torch.device,
    ) -> Tensor:
        """Expand [B, K, K] same-image IoU to [B*K, B*K]."""
        num_items = bsz * num_captions
        flat_iou = torch.zeros(
            (num_items, num_items),
            device=device,
            dtype=iou.dtype,
        )

        idx = torch.arange(num_items, device=device).reshape(bsz, num_captions)
        rows = idx[:, :, None].expand(-1, -1, num_captions).reshape(-1)
        cols = idx[:, None, :].expand(-1, num_captions, -1).reshape(-1)

        flat_iou[rows, cols] = iou.reshape(-1)
        return flat_iou

    def _build_labels(
        self,
        logits: Tensor,
        bsz: int,
        num_captions: int,
        sigmoid_map: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor | None]:
        device = logits.device
        dtype = logits.dtype
        num_items = bsz * num_captions

        flat_ids = torch.arange(num_items, device=device)
        image_ids = torch.arange(bsz, device=device).repeat_interleave(num_captions)

        same_pair = flat_ids[:, None].eq(flat_ids[None, :])
        same_image = image_ids[:, None].eq(image_ids[None, :])
        diff_image = ~same_image
        same_image_diff_caption = same_image & (~same_pair)

        labels = torch.zeros((num_items, num_items), device=device, dtype=dtype)

        # 1. Matched sub-caption and its conditioned visual feature.
        labels.masked_fill_(same_pair, 1.0)

        # 2. Different-image pairs are negatives.
        labels.masked_fill_(diff_image, -1.0)

        same_image_positive = torch.zeros_like(same_pair)
        same_image_negative = torch.zeros_like(same_pair)
        iou_values = None

        # 3. Same-image different-sub-caption supervision.
        if self.caption_same_image_mode == "ignore":
            pass

        else:
            if sigmoid_map is None:
                raise KeyError(
                    "caption_same_image_mode='positive' or 'signed' requires "
                    "sigmoid_map to compute binary IoU."
                )

            binary_iou = self._compute_binary_iou(
                sigmoid_map=sigmoid_map,
                bsz=bsz,
                num_captions=num_captions,
            )
            flat_iou = self._expand_iou_to_flat(
                iou=binary_iou,
                bsz=bsz,
                num_captions=num_captions,
                device=device,
            ).to(dtype=dtype)

            has_overlap = flat_iou.gt(0.0)
            same_image_positive = same_image_diff_caption & has_overlap
            no_overlap = same_image_diff_caption & (~has_overlap)

            # Soft positive: label is binary IoU, not hard +1.
            labels[same_image_positive] = flat_iou[same_image_positive]

            if self.caption_same_image_mode == "positive":
                # No-overlap same-image pairs remain ignored.
                labels[no_overlap] = 0.0
            elif self.caption_same_image_mode == "signed":
                # No-overlap same-image pairs become negatives.
                labels[no_overlap] = -1.0
                same_image_negative = no_overlap
            else:
                raise RuntimeError(
                    f"Unexpected caption_same_image_mode: {self.caption_same_image_mode}"
                )

            iou_values = flat_iou.masked_select(same_image_diff_caption).detach().float()

        masks = {
            "same_pair": same_pair,
            "same_image": same_image,
            "diff_image": diff_image,
            "same_image_diff_caption": same_image_diff_caption,
            "same_image_positive": same_image_positive,
            "same_image_negative": same_image_negative,
            "positive": labels.gt(0.0),
            "negative": labels.lt(0.0),
            "valid": labels.ne(0.0),
            "ignored": labels.eq(0.0),
        }
        return labels, masks, iou_values

    def _stats(
        self,
        reference: Tensor,
        similarities: Tensor,
        logits: Tensor,
        loss_matrix: Tensor,
        labels: Tensor,
        masks: dict[str, Tensor],
        iou_values: Tensor | None,
        loss: Tensor,
        bsz: int,
        num_captions: int,
    ) -> dict[str, Tensor]:
        positive_mask = masks["positive"]
        negative_mask = masks["negative"]
        valid_mask = masks["valid"]
        ignored_mask = masks["ignored"]
        same_image_diff_caption = masks["same_image_diff_caption"]
        same_image_positive = masks["same_image_positive"]
        same_image_negative = masks["same_image_negative"]

        num_items = bsz * num_captions
        num_total_pairs = num_items * num_items
        num_valid_pairs = int(valid_mask.sum().item())
        num_ignored_pairs = int(ignored_mask.sum().item())
        num_positive_pairs = int(positive_mask.sum().item())
        num_negative_pairs = int(negative_mask.sum().item())
        num_same_image_diff_caption = int(same_image_diff_caption.sum().item())

        total_negative_count = max(num_total_pairs - num_items, 1)
        valid_negative_count = num_negative_pairs

        pos_sim = self._masked_mean(similarities, positive_mask)
        neg_sim = self._masked_mean(similarities, negative_mask)
        pos_logit = self._masked_mean(logits, positive_mask)
        neg_logit = self._masked_mean(logits, negative_mask)

        zero = reference.detach().new_zeros((), dtype=torch.float32)

        if iou_values is not None and iou_values.numel() > 0:
            iou_mean = iou_values.mean()
            iou_min = iou_values.min()
            iou_max = iou_values.max()
            iou_std = iou_values.std(unbiased=False)
        else:
            iou_mean = iou_min = iou_max = iou_std = zero

        same_image_active = same_image_positive | same_image_negative
        same_image_weight = labels.masked_select(
            same_image_active
        ).detach().abs().float()

        if same_image_weight.numel() > 0:
            same_image_weight_mean = same_image_weight.mean()
            same_image_weight_sum = same_image_weight.sum()
        else:
            same_image_weight_mean = zero
            same_image_weight_sum = zero

        return {
            "caption_primary_loss": loss.detach().float(),
            "caption_pos_loss": self._masked_mean(loss_matrix, positive_mask),
            "caption_neg_loss": self._masked_mean(loss_matrix, negative_mask),
            "caption_pos_sim_mean": pos_sim,
            "caption_neg_sim_mean": neg_sim,
            "caption_margin_sim": (pos_sim - neg_sim).detach().float(),
            "caption_pos_logit_mean": pos_logit,
            "caption_neg_logit_mean": neg_logit,
            "caption_margin_logit": (pos_logit - neg_logit).detach().float(),
            "caption_num_positive_pairs": self._scalar(reference, num_positive_pairs),
            "caption_num_negative_pairs": self._scalar(reference, num_negative_pairs),
            "caption_num_masked_same_image_pairs": self._scalar(reference, num_ignored_pairs),
            "caption_num_valid_pairs": self._scalar(reference, num_valid_pairs),
            "caption_num_total_pairs": self._scalar(reference, num_total_pairs),
            "caption_valid_negative_fraction": self._scalar(
                reference,
                float(valid_negative_count) / float(total_negative_count),
            ),
            "caption_masked_same_image_pairs": self._scalar(reference, num_ignored_pairs),
            "caption_num_pairs": self._scalar(reference, num_total_pairs),
            "caption_same_image_mode_code": self._scalar(reference, self._mode_code()),
            "caption_num_ignored_pairs": self._scalar(reference, num_ignored_pairs),
            "caption_num_same_image_diff_caption_pairs": self._scalar(
                reference,
                num_same_image_diff_caption,
            ),
            "caption_same_image_positive_pairs": self._scalar(
                reference,
                int(same_image_positive.sum().item()),
            ),
            "caption_same_image_negative_pairs": self._scalar(
                reference,
                int(same_image_negative.sum().item()),
            ),
            "caption_label_abs_mean": labels.detach().abs().float().mean(),
            # Compatibility keys for current training logs.
            "caption_region_positive_loss": self._masked_mean(
                loss_matrix,
                same_image_positive,
            ),
            "caption_region_negative_loss": self._masked_mean(
                loss_matrix,
                same_image_negative,
            ),
            "caption_region_same_image_loss": self._masked_mean(
                loss_matrix,
                same_image_active,
            ),
            "caption_region_positive_weight_sum": labels.masked_select(
                same_image_positive,
            ).detach().abs().float().sum()
            if bool(same_image_positive.any().item())
            else zero,
            "caption_region_negative_weight_sum": labels.masked_select(
                same_image_negative,
            ).detach().abs().float().sum()
            if bool(same_image_negative.any().item())
            else zero,
            "caption_region_positive_active_fraction": self._scalar(
                reference,
                float(same_image_positive.sum().item())
                / float(max(num_same_image_diff_caption, 1)),
            ),
            "caption_region_negative_active_fraction": self._scalar(
                reference,
                float(same_image_negative.sum().item())
                / float(max(num_same_image_diff_caption, 1)),
            ),
            "caption_region_active_fraction": self._scalar(
                reference,
                float(same_image_active.sum().item())
                / float(max(num_same_image_diff_caption, 1)),
            ),
            "caption_region_overlap_mean": iou_mean,
            "caption_region_overlap_min": iou_min,
            "caption_region_overlap_max": iou_max,
            "caption_region_overlap_std": iou_std,
            "caption_region_pos_weight_effective": self._scalar(reference, 1.0),
            "caption_region_neg_weight_effective": self._scalar(reference, 1.0),
            "caption_region_weight_effective": self._scalar(reference, 1.0),
            "caption_region_warmup_alpha": self._scalar(reference, 1.0),
            "caption_region_weight_mean": same_image_weight_mean,
            "caption_region_weight_sum": same_image_weight_sum,
        }

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
    """SPVD loss for multi-sub-caption conditional visual decomposition.

    This loss assumes the SPVD training path always uses multi-sub-caption input.

    Required outputs:
        caption_semantic_visual_features or caption_shared_visual_features: [B, K, D]
        caption_text_features: [B, K, D]
        logit_scale: scalar

    Optional outputs:
        caption_residual_visual_features or residual_visual_features: [B, K, D]
        sigmoid_map: [B, K, S, N]
        residual_map: [B, K, S, N]
        gate_logits: [B, K, S, N]
        logit_bias: scalar

    No single-caption [B, D] fallback.
    No global image-text alignment.
    No OpenCLIP-style image_features/text_features fallback.
    """

    expects_output_dict = True

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        caption_align_weight: float = 1.0,
        branch_bce_weight: float = 0.0,
        branch_logit_scale: float = 5.0,
        residual_negative_weight: float = 0.25,
        detach_text_for_residual: bool = True,
        residual_variance_weight: float = 0.0,
        residual_variance_gamma: float = 1.0,
        caption_same_image_mode: str = "ignore",
        gate_binarize_threshold: float = 0.5,
        loss_dist_impl: str | None = None,
        debug_finite_checks: bool = False,
    ) -> None:
        super().__init__()

        self.caption_align_weight = float(caption_align_weight)
        self.branch_bce_weight = float(branch_bce_weight)
        self.residual_variance_weight = float(residual_variance_weight)
        self.debug_finite_checks = bool(debug_finite_checks)

        if self.caption_align_weight <= 0.0:
            raise ValueError(
                "SPVDLoss requires caption_align_weight > 0 because SPVD "
                "does not use global image-text alignment."
            )

        self.caption_loss = MaskedCaptionSigLipLoss(
            caption_same_image_mode=caption_same_image_mode,
            gate_binarize_threshold=gate_binarize_threshold,
            rank=rank,
            world_size=world_size,
            dist_impl=loss_dist_impl,
        )

        self.branch_bce = BranchBCELoss(
            logit_scale=branch_logit_scale,
            residual_negative_weight=residual_negative_weight,
            detach_text_for_residual=detach_text_for_residual,
        )

        self.residual_variance = ResidualVarianceLoss(
            gamma=residual_variance_gamma,
        )

        self.gate_stats = GateMapStats()
        self.global_step = 0

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def forward(
        self,
        outputs: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(outputs, Mapping):
            raise TypeError(
                "SPVDLoss expects an output dict. Tuple outputs are not supported "
                "in the multi-sub-caption SPVD training path."
            )

        caption_visual_features = _first_tensor(
            outputs,
            ("caption_semantic_visual_features", "caption_shared_visual_features"),
        )
        caption_text_features = _first_tensor(outputs, ("caption_text_features",))
        residual_features = _first_tensor(
            outputs,
            ("caption_residual_visual_features", "residual_visual_features"),
        )

        sigmoid_map = _first_tensor(outputs, ("sigmoid_map",))
        residual_map = _first_tensor(outputs, ("residual_map",))
        gate_logits = _first_tensor(outputs, ("gate_logits",))
        logit_scale = _first_tensor(outputs, ("logit_scale",))
        logit_bias = _first_tensor(outputs, ("logit_bias",))

        if logit_scale is None:
            raise KeyError("SPVDLoss requires logit_scale.")

        if caption_visual_features is None:
            raise KeyError(
                "SPVDLoss requires caption_semantic_visual_features or "
                "caption_shared_visual_features."
            )

        if caption_text_features is None:
            raise KeyError("SPVDLoss requires caption_text_features.")

        self._assert_3d_same_shape(
            caption_visual_features,
            caption_text_features,
            name_a="caption_visual_features",
            name_b="caption_text_features",
        )

        if residual_features is not None:
            self._assert_3d_same_shape(
                caption_visual_features,
                residual_features,
                name_a="caption_visual_features",
                name_b="residual_features",
            )

        if sigmoid_map is not None:
            self._assert_map_shape(
                sigmoid_map,
                caption_visual_features,
                name="sigmoid_map",
            )

        if residual_map is not None:
            self._assert_map_shape(
                residual_map,
                caption_visual_features,
                name="residual_map",
            )

        if gate_logits is not None:
            self._assert_map_shape(
                gate_logits,
                caption_visual_features,
                name="gate_logits",
            )

        if self.debug_finite_checks:
            assert_finite_tensor("caption_visual_features", caption_visual_features, enabled=True)
            assert_finite_tensor("caption_text_features", caption_text_features, enabled=True)
            assert_finite_tensor("residual_features", residual_features, enabled=True)
            assert_finite_tensor("sigmoid_map", sigmoid_map, enabled=True)
            assert_finite_tensor("residual_map", residual_map, enabled=True)
            assert_finite_tensor("gate_logits", gate_logits, enabled=True)

        loss_caption, caption_stats = self.caption_loss(
            caption_visual_features,
            caption_text_features,
            logit_scale,
            logit_bias=logit_bias,
            sigmoid_map=sigmoid_map,
            global_step=self.global_step,
        )

        reference = caption_visual_features
        zero = _zero(reference)

        if self.branch_bce_weight > 0.0:
            if residual_features is None:
                raise KeyError(
                    "branch_bce_weight > 0 requires caption_residual_visual_features "
                    "or residual_visual_features."
                )
            branch_terms = {
                key: value.to(reference.device)
                for key, value in self.branch_bce(
                    caption_visual_features,
                    residual_features,
                    caption_text_features,
                ).items()
            }
            loss_branch = branch_terms["loss_branch"]
        else:
            branch_terms = self._zero_branch_terms(zero)
            loss_branch = zero

        if self.residual_variance_weight > 0.0:
            if residual_features is None:
                raise KeyError(
                    "residual_variance_weight > 0 requires caption_residual_visual_features "
                    "or residual_visual_features."
                )
            loss_residual_variance = self.residual_variance(residual_features).to(reference.device)
        else:
            loss_residual_variance = zero

        if sigmoid_map is not None:
            gate_terms = {
                key: value.to(reference.device)
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
            self.caption_align_weight * loss_caption
            + self.branch_bce_weight * loss_branch
            + self.residual_variance_weight * loss_residual_variance
        )

        if self.debug_finite_checks:
            assert_finite_tensor("loss_caption", loss_caption, enabled=True)
            assert_finite_tensor("loss_branch", loss_branch, enabled=True)
            assert_finite_tensor("loss_residual_variance", loss_residual_variance, enabled=True)
            assert_finite_tensor("total_loss", total_loss, enabled=True)

        logit_scale_exp = logit_scale.detach().float()
        logit_scale_log = logit_scale_exp.clamp_min(1.0e-12).log()
        logit_bias_value = (
            logit_bias.detach().float()
            if logit_bias is not None
            else zero.detach().float()
        )

        loss_dict = {
            "loss": total_loss.detach(),
            "loss_align": loss_caption.detach(),
            "loss_sigmoid": loss_caption.detach(),
            "loss_align_caption": loss_caption.detach(),

            # Kept for logging compatibility, but global alignment is removed.
            "loss_align_global": zero.detach(),
            "global_align_enabled": zero.new_tensor(0.0),
            "caption_align_enabled": zero.new_tensor(1.0),

            **{key: value.detach() for key, value in caption_stats.items()},

            "loss_branch": branch_terms["loss_branch"].detach(),
            "loss_branch_s_text": branch_terms["loss_branch_s_text"].detach(),
            "loss_branch_r_text": branch_terms["loss_branch_r_text"].detach(),
            "branch_sim_s_text": branch_terms["branch_sim_s_text"].detach(),
            "branch_sim_r_text": branch_terms["branch_sim_r_text"].detach(),
            "branch_gap_s_minus_r": branch_terms["branch_gap_s_minus_r"].detach(),

            "loss_residual_variance": loss_residual_variance.detach(),
            "loss_branch_weight_effective": zero.new_tensor(self.branch_bce_weight),
            "loss_residual_variance_weight_effective": zero.new_tensor(
                self.residual_variance_weight
            ),

            "gate_mean": gate_terms["gate_mean"].detach(),
            "gate_std": gate_terms["gate_std"].detach(),
            "gate_min": gate_terms["gate_min"].detach(),
            "gate_max": gate_terms["gate_max"].detach(),

            "logit_scale": logit_scale.detach(),
            "logit_scale_exp": logit_scale_exp.detach(),
            "logit_scale_log": logit_scale_log.detach(),
            "logit_bias": logit_bias_value.detach(),
        }

        return total_loss, loss_dict

    def _assert_3d_same_shape(
        self,
        a: Tensor,
        b: Tensor,
        name_a: str,
        name_b: str,
    ) -> None:
        if a.ndim != 3:
            raise ValueError(
                f"{name_a} must have shape [B, K, D]. "
                f"Single-caption [B, D] input is not supported. "
                f"Got {tuple(a.shape)}."
            )
        if b.ndim != 3:
            raise ValueError(
                f"{name_b} must have shape [B, K, D]. "
                f"Single-caption [B, D] input is not supported. "
                f"Got {tuple(b.shape)}."
            )
        if a.shape != b.shape:
            raise ValueError(
                f"{name_a} and {name_b} must have the same shape, "
                f"got {tuple(a.shape)} and {tuple(b.shape)}."
            )

    def _assert_map_shape(
        self,
        tensor: Tensor,
        caption_features: Tensor,
        name: str,
    ) -> None:
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B, K, S, N], got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != caption_features.shape[0]:
            raise ValueError(
                f"{name} batch dimension must match caption features: "
                f"{tensor.shape[0]} vs {caption_features.shape[0]}."
            )
        if tensor.shape[1] != caption_features.shape[1]:
            raise ValueError(
                f"{name} caption dimension must match caption features: "
                f"{tensor.shape[1]} vs {caption_features.shape[1]}."
            )

    def _zero_branch_terms(self, zero: Tensor) -> dict[str, Tensor]:
        return {
            "loss_branch": zero,
            "loss_branch_s_text": zero,
            "loss_branch_r_text": zero,
            "branch_sim_s_text": zero,
            "branch_sim_r_text": zero,
            "branch_gap_s_minus_r": zero,
        }