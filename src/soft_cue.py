"""Soft language cue extraction modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SlotAttentionBlock(nn.Module):
    """Slot Attention update block over text token features."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0, mlp_ratio: float = 4.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = float(eps)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.slot_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(embed_dim, embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, value: Tensor, sequence_dim: int) -> Tensor:
        batch_size = value.shape[0]
        return value.reshape(batch_size, sequence_dim, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, slots: Tensor, text_tokens: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Update slots [B, S, D] from text_tokens [B, L, D] using slot-normalized attention."""
        batch_size, num_slots, _ = slots.shape
        num_tokens = text_tokens.shape[1]
        norm_inputs = self.input_norm(text_tokens)
        slots_prev = slots
        norm_slots = self.slot_norm(slots)

        q = self._split_heads(self.q_proj(norm_slots), num_slots)
        k = self._split_heads(self.k_proj(norm_inputs), num_tokens)
        v = self._split_heads(self.v_proj(norm_inputs), num_tokens)

        logits = torch.einsum("bhld,bhsd->bhls", k, q) * self.scale
        attn = torch.softmax(logits, dim=-1)
        attn = attn + self.eps
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, num_tokens):
                raise ValueError(f"attention_mask must have shape {(batch_size, num_tokens)}, got {tuple(attention_mask.shape)}.")
            valid = attention_mask.to(device=attn.device, dtype=attn.dtype).unsqueeze(1).unsqueeze(-1)
            attn = attn * valid
        attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(self.eps)
        attn = self.dropout(attn)

        updates = torch.einsum("bhls,bhld->bhsd", attn, v)
        updates = updates.transpose(1, 2).reshape(batch_size, num_slots, self.embed_dim)
        updates = self.out_proj(updates)

        slots = self.gru(updates.reshape(-1, self.embed_dim), slots_prev.reshape(-1, self.embed_dim))
        slots = slots.reshape(batch_size, num_slots, self.embed_dim)
        slots = slots + self.mlp(self.mlp_norm(slots))
        return slots


class SoftCueExtractor(nn.Module):
    """Extract fine-grained language soft cues from token features with Slot Attention."""

    def __init__(
        self,
        text_dim: int,
        embed_dim: int,
        num_soft_cues: int = 4,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_soft_cues <= 0:
            raise ValueError("num_soft_cues must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.num_soft_cues = int(num_soft_cues)
        self.text_proj = nn.Identity() if self.text_dim == self.embed_dim else nn.Linear(self.text_dim, self.embed_dim)
        self.soft_cue_slots = nn.Parameter(torch.empty(self.num_soft_cues, self.embed_dim))
        self.blocks = nn.ModuleList(
            SlotAttentionBlock(self.embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.soft_cue_slots, std=0.02)

    def forward(self, text_tokens: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return soft cues [B, S, D] from text tokens [B, L, D_t]."""
        if text_tokens.ndim != 3:
            raise ValueError(f"text_tokens must have shape [B, L, D], got {tuple(text_tokens.shape)}.")
        if attention_mask is not None and attention_mask.shape != text_tokens.shape[:2]:
            raise ValueError(f"attention_mask must have shape {tuple(text_tokens.shape[:2])}, got {tuple(attention_mask.shape)}.")
        if isinstance(self.text_proj, nn.Linear):
            text_tokens = text_tokens.to(dtype=self.text_proj.weight.dtype)
        text_tokens = self.text_proj(text_tokens)
        batch_size = text_tokens.shape[0]
        slots = self.soft_cue_slots.unsqueeze(0).expand(batch_size, -1, -1)
        slots = slots.to(device=text_tokens.device, dtype=text_tokens.dtype)
        mask = attention_mask.to(device=text_tokens.device, dtype=torch.bool) if attention_mask is not None else None
        for block in self.blocks:
            slots = block(slots, text_tokens, attention_mask=mask)
        return self.out_norm(slots)
