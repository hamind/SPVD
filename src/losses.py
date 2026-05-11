"""SPVD loss functions with OpenCLIP-style contrastive plumbing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from open_clip.loss import SigLipLoss as OpenCLIPSigLipLoss
from torch import Tensor, nn

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - distributed is available in normal torch wheels
    dist = None

try:
    import torch.distributed.nn as dist_nn
except ImportError:  # pragma: no cover
    dist_nn = None


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


def _distributed_ready(world_size: int) -> bool:
    return bool(world_size > 1 and dist is not None and dist.is_available() and dist.is_initialized())


def _gather_one(
    features: Tensor,
    *,
    local_loss: bool,
    gather_with_grad: bool,
    rank: int,
    world_size: int,
) -> Tensor:
    if not _distributed_ready(world_size):
        return features
    if gather_with_grad:
        if dist_nn is None:
            raise RuntimeError("torch.distributed.nn is required when gather_with_grad=True.")
        return torch.cat(dist_nn.all_gather(features), dim=0)

    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features)
    if not local_loss:
        gathered[rank] = features
    return torch.cat(gathered, dim=0)


def gather_features(
    image_features: Tensor,
    text_features: Tensor,
    *,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Tensor, Tensor]:
    """Gather image/text features across ranks for contrastive losses."""
    all_image_features = _gather_one(
        image_features,
        local_loss=local_loss,
        gather_with_grad=gather_with_grad,
        rank=rank,
        world_size=world_size,
    )
    all_text_features = _gather_one(
        text_features,
        local_loss=local_loss,
        gather_with_grad=gather_with_grad,
        rank=rank,
        world_size=world_size,
    )
    return all_image_features, all_text_features


class InfoNCELoss(nn.Module):
    """Symmetric CLIP InfoNCE loss implemented locally for project models."""

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.local_loss = bool(local_loss)
        self.gather_with_grad = bool(gather_with_grad)
        self.cache_labels = bool(cache_labels)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._labels: dict[tuple[torch.device, int, int, bool], Tensor] = {}

    def _ground_truth(self, device: torch.device, batch_size: int, distributed: bool) -> Tensor:
        offset = self.rank * batch_size if distributed and self.local_loss else 0
        key = (device, batch_size, offset, self.local_loss)
        if not self.cache_labels or key not in self._labels:
            self._labels[key] = torch.arange(batch_size, device=device, dtype=torch.long) + offset
        return self._labels[key]

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
    ) -> Tensor:
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        distributed = _distributed_ready(self.world_size)
        all_image_features, all_text_features = gather_features(
            image_features,
            text_features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
        )

        if distributed and self.local_loss:
            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
            labels = self._ground_truth(image_features.device, image_features.shape[0], distributed=True)
        else:
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T
            labels = self._ground_truth(logits_per_image.device, logits_per_image.shape[0], distributed=False)

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text = logits_per_text + logit_bias
        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


def bidirectional_routing_bce_loss(
    shared_routing: Tensor,
    residual_routing: Tensor,
    relevance_scores: Tensor,
    cue_weights: Tensor | None = None,
    detach_relevance: bool = True,
    positive_constraint: bool = True,
    negative_constraint: bool = True,
    eps: float = 1.0e-6,
) -> Tensor:
    """BCE routing loss: high relevance to shared, low relevance to residual."""
    rho = relevance_scores.detach() if detach_relevance else relevance_scores
    terms: list[Tensor] = []
    if positive_constraint:
        terms.append(rho * torch.log(shared_routing + eps))
    if negative_constraint:
        terms.append((1.0 - rho) * torch.log(residual_routing + eps))
    if not terms:
        return shared_routing.new_zeros(())
    loss = -sum(terms)
    if cue_weights is not None:
        loss = loss * cue_weights.to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
    return loss.mean()


def residual_preservation_loss(
    residual_features: Tensor,
    gamma: float = 1.0,
    eps: float = 1.0e-4,
) -> Tensor:
    """Variance loss that discourages residual features from collapsing to constants."""
    std = torch.sqrt(residual_features.float().var(dim=0, unbiased=False) + eps)
    return F.relu(float(gamma) - std).mean()


def shared_residual_decorrelation_loss(
    shared_features: Tensor,
    residual_features: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    """Reduce linear redundancy between shared and residual representations."""
    if shared_features.shape[0] < 2:
        return _zero(shared_features)
    z_s = shared_features.float() - shared_features.float().mean(dim=0, keepdim=True)
    z_r = residual_features.float() - residual_features.float().mean(dim=0, keepdim=True)
    z_s = z_s / (z_s.std(dim=0, unbiased=False, keepdim=True) + eps)
    z_r = z_r / (z_r.std(dim=0, unbiased=False, keepdim=True) + eps)
    cross_cov = z_s.transpose(0, 1) @ z_r / float(max(int(z_s.shape[0]) - 1, 1))
    return cross_cov.pow(2).sum() / float(z_s.shape[-1])


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
        decomp_loss_weight: float = 0.0,
        route_positive_constraint: bool = True,
        route_negative_constraint: bool = True,
        residual_loss_weight: float = 0.0,
        orth_loss_weight: float = 0.0,
        detach_relevance: bool = True,
        residual_variance_gamma: float = 1.0,
        align_weight: float = 1.0,
        global_align_weight: float = 1.0,
        caption_align_weight: float = 1.0,
        loss_dist_impl: str | None = None,
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
        self.decomp_loss_weight = float(decomp_loss_weight)
        self.route_positive_constraint = bool(route_positive_constraint)
        self.route_negative_constraint = bool(route_negative_constraint)
        self.residual_loss_weight = float(residual_loss_weight)
        self.orth_loss_weight = float(orth_loss_weight)
        self.detach_relevance = bool(detach_relevance)
        self.residual_variance_gamma = float(residual_variance_gamma)

    def forward(self, outputs: Mapping[str, Any] | tuple[Tensor, ...]) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(outputs, Mapping):
            outputs = {
                "image_features": outputs[0],
                "text_features": outputs[1],
                "logit_scale": outputs[2],
            }

        image_features = _first_tensor(outputs, ("image_features", "shared_visual_features", "z_v_s"))
        text_features = _first_tensor(outputs, ("text_features", "z_t", "text_global"))
        caption_visual_features = _first_tensor(outputs, ("caption_shared_visual_features",))
        caption_text_features = _first_tensor(outputs, ("caption_text_features",))
        logit_scale = _first_tensor(outputs, ("logit_scale",))
        logit_bias = _first_tensor(outputs, ("logit_bias",))
        if logit_scale is None:
            raise KeyError("SPVDLoss requires logit_scale.")

        normalized_image_features = F.normalize(image_features, dim=-1) if image_features is not None else None
        align_terms: list[tuple[float, Tensor]] = []
        zero_reference = normalized_image_features
        if zero_reference is None:
            zero_reference = caption_visual_features if caption_visual_features is not None else caption_text_features

        if image_features is not None and text_features is not None and self.global_align_weight != 0.0:
            loss_align_global = self.align_loss(image_features, text_features, logit_scale, logit_bias=logit_bias)
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
            align_image_features = caption_visual_features.reshape(-1, caption_visual_features.shape[-1])
            align_text_features = caption_text_features.reshape(-1, caption_text_features.shape[-1])
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
        shared_routing = _first_tensor(outputs, ("shared_routing",))
        residual_routing = _first_tensor(outputs, ("residual_routing",))
        relevance_scores = _first_tensor(outputs, ("relevance_scores",))
        cue_weights = _first_tensor(outputs, ("cue_weights",))
        residual_features = _first_tensor(outputs, ("residual_visual_features", "z_v_p"))

        has_decomp = shared_routing is not None or residual_routing is not None or relevance_scores is not None
        if has_decomp and (shared_routing is None or residual_routing is None or relevance_scores is None or residual_features is None):
            raise KeyError("Soft-cue loss requires shared_routing, residual_routing, relevance_scores, and residual_visual_features.")

        if has_decomp and self.decomp_loss_weight != 0.0:
            loss_decomp = bidirectional_routing_bce_loss(
                shared_routing,
                residual_routing,
                relevance_scores,
                cue_weights,
                detach_relevance=self.detach_relevance,
                positive_constraint=self.route_positive_constraint,
                negative_constraint=self.route_negative_constraint,
            ).to(align_image_features.device)
        else:
            loss_decomp = zero

        if has_decomp and self.residual_loss_weight != 0.0:
            loss_residual = residual_preservation_loss(residual_features, gamma=self.residual_variance_gamma).to(align_image_features.device)
        else:
            loss_residual = zero

        if has_decomp and self.orth_loss_weight != 0.0:
            orth_anchor = normalized_image_features if normalized_image_features is not None else align_image_features
            loss_orth = shared_residual_decorrelation_loss(orth_anchor, residual_features).to(align_image_features.device)
        else:
            loss_orth = zero

        total_loss = (
            self.align_weight * loss_align
            + self.decomp_loss_weight * loss_decomp
            + self.residual_loss_weight * loss_residual
            + self.orth_loss_weight * loss_orth
        )
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_align": loss_align.detach(),
            "loss_sigmoid": loss_align.detach(),
            "loss_align_global": loss_align_global.detach(),
            "loss_align_caption": loss_align_caption.detach(),
            "loss_decomp": loss_decomp.detach(),
            "loss_residual": loss_residual.detach(),
            "loss_orth": loss_orth.detach(),
            "logit_scale": logit_scale.detach(),
        }
