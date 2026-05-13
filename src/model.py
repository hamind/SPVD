"""SPVD model definition."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from open_clip.hf_model import HFTextEncoder
from open_clip.modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import (
    Attention,
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    TextTransformer,
    VisionTransformer,
    text_global_pool,
)
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CLIPVisionCfg:
    layers: tuple[int, int, int, int] | int = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: tuple[int, int] | int = 224

    ls_init_value: float | None = None
    patch_dropout: float = 0.0
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: dict[str, Any] | None = None
    norm_kwargs: dict[str, Any] | None = None

    timm_model_name: str | None = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: float | None = None


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: str | None = None
    tokenizer_kwargs: dict[str, Any] | None = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: float | None = None
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict[str, Any] | None = None
    norm_kwargs: dict[str, Any] | None = None

    hf_model_name: str | None = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"


def get_cast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg | dict[str, Any],
    quick_gelu: bool = False,
    cast_dtype: torch.dtype | None = None,
) -> nn.Module:
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        return TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )

    if isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        return ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    if vision_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
    if vision_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **vision_cfg.act_kwargs)

    return VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        pos_embed_type=vision_cfg.pos_embed_type,
        no_ln_pre=vision_cfg.no_ln_pre,
        final_ln_after_pool=vision_cfg.final_ln_after_pool,
        pool_type=vision_cfg.pool_type,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg | dict[str, Any],
    quick_gelu: bool = False,
    cast_dtype: torch.dtype | None = None,
) -> nn.Module:
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        return HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    if text_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
    if text_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **text_cfg.act_kwargs)

    return TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        no_causal_mask=text_cfg.no_causal_mask,
        pad_id=text_cfg.pad_id,
        pool_type=text_cfg.pool_type,
        proj_bias=text_cfg.proj_bias,
        output_tokens=text_cfg.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )


def convert_weights_to_lp(model: nn.Module, dtype: torch.dtype = torch.float16) -> None:
    """Convert applicable model parameters to a lower precision dtype."""

    def _convert_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.to(dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dtype)

        if isinstance(module, (nn.MultiheadAttention, Attention)):
            for attr_name in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(module, attr_name, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        attr = getattr(module, "text_projection", None)
        if attr is not None and hasattr(attr, "data"):
            attr.data = attr.data.to(dtype)

        if isinstance(module, VisionTransformer):
            attr = getattr(module, "proj", None)
            if attr is not None and hasattr(attr, "data"):
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def set_model_preprocess_cfg(model: nn.Module, preprocess_cfg: dict[str, Any]) -> None:
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg["mean"]
    module.image_std = preprocess_cfg["std"]
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


def apply_projection(x: Tensor, projection: Any) -> Tensor:
    """Apply an OpenCLIP-style projection module or parameter matrix."""
    if projection is None:
        return x
    if isinstance(projection, nn.Linear):
        return projection(x)
    return x @ projection


class TextCueEncoder(nn.Module):
    def __init__(
        self,
        cue_num: int,
        embed_dim: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.cue_num = cue_num
        self.embed_dim = embed_dim

        self.cue_embed = nn.Embedding(cue_num, embed_dim)

        self.token_norm = nn.LayerNorm(embed_dim)
        self.cue_norm = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        text_token: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        B, S, L, D = text_token.shape
        assert D == self.embed_dim

        x = self.token_norm(text_token)

        cue = self.cue_embed.weight[None, None, :, :].expand(B, S, -1, -1)
        cue = self.cue_norm(cue)

        q = self.q_proj(cue)          # (B, S, C, D)
        k = self.k_proj(x)            # (B, S, L, D)
        v = self.v_proj(x)            # (B, S, L, D)

        attn_logits = torch.einsum("b s c d, b s l d -> b s c l", q, k)
        attn_logits = attn_logits / math.sqrt(D)

        if text_mask is not None:
            mask = text_mask[:, :, None, :].bool()  # (B, S, 1, L)
            attn_logits = attn_logits.masked_fill(
                ~mask,
                torch.finfo(attn_logits.dtype).min,
            )

        attn = F.softmax(attn_logits, dim=-1)       # (B, S, C, L)
        attn = self.attn_drop(attn)

        cue_update = torch.einsum("b s c l, b s l d -> b s c d", attn, v)
        cue_update = self.out_proj(cue_update)
        cue_update = self.proj_drop(cue_update)

        cue_out = self.cue_embed.weight[None, None, :, :].expand(B, S, -1, -1)
        cue_out = cue_out + cue_update

        cue_out = cue_out + self.ffn(self.ffn_norm(cue_out))
        cue_out = self.out_norm(cue_out)

        if return_attn:
            return cue_out, attn

        return cue_out


class CueDecomposition(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        embed_dim: int,
        n_head: int,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        normalize_output: bool = False,
    ):
        super().__init__()

        assert embed_dim % n_head == 0

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.normalize_output = normalize_output

        if use_layernorm:
            self.text_norm = nn.LayerNorm(embed_dim)
            self.image_norm = nn.LayerNorm(vision_dim)
        else:
            self.text_norm = nn.Identity()
            self.image_norm = nn.Identity()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(vision_dim, embed_dim)
        self.v_proj = nn.Linear(vision_dim, embed_dim)

        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        self.residual_proj = nn.Linear(embed_dim, embed_dim)

        self.semantic_norm = nn.LayerNorm(embed_dim)
        self.residual_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def forward(
        self,
        image_token: torch.Tensor,
        text_cue: torch.Tensor,
        return_map: bool = True,
    ):
        B, N, _ = image_token.shape
        _, S, C, _ = text_cue.shape

        H = self.n_head
        Dh = self.head_dim

        image_x = self.image_norm(image_token)
        cue_x = self.text_norm(text_cue)

        q = self.q_proj(cue_x).view(B, S, C, H, Dh)
        k = self.k_proj(image_x).view(B, N, H, Dh)
        v = self.v_proj(image_x).view(B, N, H, Dh)

        logits = torch.einsum(
            "b s c h d, b n h d -> b s c n h",
            q,
            k,
        ) / math.sqrt(Dh)

        scale = self.logit_scale.clamp(0.1, 10.0)
        semantic_map = torch.sigmoid(logits * scale + self.logit_bias)
        residual_map = 1.0 - semantic_map

        semantic_cue = torch.einsum(
            "b s c n h, b n h d -> b s c h d",
            semantic_map,
            v,
        )

        residual_cue = torch.einsum(
            "b s c n h, b n h d -> b s c h d",
            residual_map,
            v,
        )

        sem_denom = semantic_map.sum(dim=3).unsqueeze(-1).clamp_min(1e-6)
        res_denom = residual_map.sum(dim=3).unsqueeze(-1).clamp_min(1e-6)

        semantic_cue = semantic_cue / sem_denom
        residual_cue = residual_cue / res_denom

        semantic_cue = semantic_cue.contiguous().view(B, S, C, self.embed_dim)
        residual_cue = residual_cue.contiguous().view(B, S, C, self.embed_dim)

        semantic = self.semantic_norm(self.semantic_proj(semantic_cue).mean(dim=2))
        residual = self.residual_norm(self.residual_proj(residual_cue).mean(dim=2))

        if return_map:
            return semantic, residual, semantic_map, residual_map

        return semantic, residual


class SPVD(nn.Module):
    """SPVD model class."""

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg | dict[str, Any],
        text_cfg: CLIPTextCfg | dict[str, Any],
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: torch.dtype | None = None,
        cue_num: int = 4,
        spvd_cfg: dict[str, Any] | None = None,
        output_dict: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        spvd_cfg = spvd_cfg or {}
        cue_num = int(spvd_cfg.get("num_soft_cues", cue_num))
        if isinstance(vision_cfg, dict):
            vision_cfg = dict(vision_cfg)
            vision_cfg["output_tokens"] = True
        else:
            vision_cfg.output_tokens = True
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)

        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        self.scale = text.width ** -0.5
        self.text_projection = nn.Parameter(self.scale * torch.randn(text.width, embed_dim))

        self.cue_encoder = TextCueEncoder(cue_num=cue_num, embed_dim=embed_dim)
        self.decom = CueDecomposition(vision_dim=vision_cfg['width'], embed_dim=embed_dim, n_head=text_cfg['heads'])

    def encode_image(self, image: Tensor) -> Tensor:
        image_embedding, image_token = self.visual(image)

        return image_embedding, image_token


    def encode_text(self, text: Tensor) -> Tensor:
        if text.ndim == 2:
            text = text.unsqueeze(1)
        if text.ndim != 3:
            raise ValueError(f"text must have shape [B, L] or [B, K, L], got {tuple(text.shape)}")
        B, K, L = text.shape
        cast_dtype = self.transformer.get_cast_dtype()

        text = text.view(-1, L)  # (B*K, 77)

        x = self.token_embedding(text).to(cast_dtype)
        D = x.shape[-1]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        text_embedding, text_tokens = text_global_pool(x, text, self.text_pool_type)

        text_mask = text.ne(0).reshape(B, K, L)
        text_tokens = text_tokens.view(B, K, L, D)
        text_cue = self.cue_encoder(text_tokens, text_mask)

        text_embedding = text_embedding @ self.text_projection
        text_cue = text_cue @ self.text_projection

        return text_embedding.view(B, K, D), text_cue.view(B, K, -1, D)

    def forward(self, image, text):
        _, image_token = self.encode_image(image)
        text_embedding, text_cue = self.encode_text(text)

        semantic, residual, semantic_map, _ = self.decom(image_token, text_cue)

        out_dict = {
            "semantic": semantic,
            "residual": residual,
            "text_features": text_embedding,
            "semantic_map": semantic_map.mean(-1),
            "logit_scale": self.logit_scale.exp(),
        }

        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias
        return out_dict


SPVDModel = SPVD
