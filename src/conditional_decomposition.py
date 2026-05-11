"""Soft-cue-conditioned bidirectional visual decomposition."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SoftCueVisualRouter(nn.Module):
    """Predict shared routing with normalized cue-image similarity."""

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        router_dim = int(hidden_dim or embed_dim)
        self.q_proj = nn.Linear(embed_dim, router_dim)
        self.v_proj = nn.Linear(embed_dim, router_dim)

    def forward(self, q: Tensor, v: Tensor) -> Tensor:
        """Return shared routing probabilities [B, S, M] or [B, K, S, M]."""
        q = F.normalize(self.q_proj(q), dim=-1)
        v = F.normalize(self.v_proj(v), dim=-1)
        if q.ndim == 3:
            similarity = torch.einsum("bsd,bmd->bsm", q, v)
        elif q.ndim == 4:
            similarity = torch.einsum("bksd,bmd->bksm", q, v)
        else:
            raise ValueError("q must have shape [B, S, D] or [B, K, S, D].")
        return (similarity + 1.0) * 0.5


class SoftCueBidirectionalDecomposition(nn.Module):
    """Condition visual shared/residual decomposition on language soft cues."""

    def __init__(
        self,
        visual_dim: int,
        embed_dim: int,
        relevance_temperature: float = 1.0,
        routing_temperature: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if relevance_temperature <= 0:
            raise ValueError("relevance_temperature must be positive.")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive.")
        self.visual_dim = int(visual_dim)
        self.embed_dim = int(embed_dim)
        self.relevance_temperature = float(relevance_temperature)
        self.routing_temperature = float(routing_temperature)
        self.eps = float(eps)
        self.visual_proj_for_relevance = nn.Linear(self.visual_dim, self.embed_dim)
        self.cue_proj_for_relevance = nn.Linear(self.embed_dim, self.embed_dim)
        self.router = SoftCueVisualRouter(self.embed_dim)
        self.shared_out_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.residual_out_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.cue_weight_head = nn.Linear(self.embed_dim, 1)

    def _aggregate_caption_features(self, caption_features: Tensor) -> Tensor:
        if caption_features.ndim == 2:
            return caption_features
        return F.normalize(caption_features.mean(dim=1), dim=-1)

    def forward(self, visual_tokens: Tensor, soft_cues: Tensor) -> dict[str, Tensor]:
        """Decompose visual tokens [B, M, D_v] using cues [B, S, D] or [B, K, S, D]."""
        if soft_cues.ndim not in {3, 4}:
            raise ValueError("soft_cues must have shape [B, S, D] or [B, K, S, D].")
        visual_tokens = visual_tokens.to(dtype=self.visual_proj_for_relevance.weight.dtype)
        soft_cues = soft_cues.to(device=visual_tokens.device, dtype=self.cue_proj_for_relevance.weight.dtype)

        q_proj = self.cue_proj_for_relevance(soft_cues)
        v_proj = self.visual_proj_for_relevance(visual_tokens)
        if soft_cues.ndim == 3:
            relevance_logits = torch.einsum("bsd,bmd->bsm", q_proj, v_proj) / math.sqrt(float(self.embed_dim))
            image_attention = torch.softmax(relevance_logits / self.relevance_temperature, dim=-1)
            cue_attended = torch.einsum("bsm,bmd->bsd", image_attention, visual_tokens)
            cue_weights = torch.softmax(self.cue_weight_head(soft_cues).squeeze(-1), dim=1)
            cue_feature_expr = "bs,bsd->bd"
        else:
            relevance_logits = torch.einsum("bksd,bmd->bksm", q_proj, v_proj) / math.sqrt(float(self.embed_dim))
            image_attention = torch.softmax(relevance_logits / self.relevance_temperature, dim=-1)
            cue_attended = torch.einsum("bksm,bmd->bksd", image_attention, visual_tokens)
            cue_weights = torch.softmax(self.cue_weight_head(soft_cues).squeeze(-1), dim=2)
            cue_feature_expr = "bks,bksd->bkd"

        relevance_scores = torch.sigmoid(relevance_logits / self.relevance_temperature)
        shared_routing = self.router(q_proj, v_proj)
        residual_routing = 1.0 - shared_routing
        routing_logits = torch.stack((shared_routing, residual_routing), dim=-1)

        den_r = residual_routing.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if soft_cues.ndim == 3:
            cue_residual = torch.einsum("bsm,bmd->bsd", residual_routing, visual_tokens) / den_r
        else:
            cue_residual = torch.einsum("bksm,bmd->bksd", residual_routing, visual_tokens) / den_r

        cue_visual_features = F.normalize(self.shared_out_proj(cue_attended), dim=-1)
        cue_residual_features = F.normalize(self.residual_out_proj(cue_residual), dim=-1)
        caption_shared_features = F.normalize(torch.einsum(cue_feature_expr, cue_weights, cue_visual_features), dim=-1)
        caption_residual_features = F.normalize(torch.einsum(cue_feature_expr, cue_weights, cue_residual_features), dim=-1)
        shared_visual_features = self._aggregate_caption_features(caption_shared_features)
        residual_visual_features = self._aggregate_caption_features(caption_residual_features)

        outputs = {
            "shared_visual_features": shared_visual_features,
            "residual_visual_features": residual_visual_features,
            "cue_visual_features": cue_visual_features,
            "cue_residual_features": cue_residual_features,
            "relevance_scores": relevance_scores,
            "image_attention": image_attention,
            "shared_routing": shared_routing,
            "residual_routing": residual_routing,
            "routing_logits": routing_logits,
            "cue_weights": cue_weights,
        }
        if soft_cues.ndim == 4:
            outputs["caption_shared_visual_features"] = caption_shared_features
            outputs["caption_residual_visual_features"] = caption_residual_features
        return outputs
