"""Project-local CLIP tower construction utilities.

These helpers keep SPVD independent while reusing OpenCLIP building blocks for
vision towers, text towers, precision conversion, and preprocessing metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Union

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


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: Optional[dict[str, Any]] = None
    norm_kwargs: Optional[dict[str, Any]] = None

    timm_model_name: Optional[str] = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: Optional[float] = None


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict[str, Any]] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: Optional[dict[str, Any]] = None
    norm_kwargs: Optional[dict[str, Any]] = None

    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"


def get_cast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def get_input_dtype(precision: str) -> torch.dtype | None:
    if precision in ("bf16", "pure_bf16"):
        return torch.bfloat16
    if precision in ("fp16", "pure_fp16"):
        return torch.float16
    return None


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


def get_model_preprocess_cfg(model: nn.Module) -> dict[str, Any]:
    module = getattr(model, "visual", model)
    preprocess_cfg = getattr(module, "preprocess_cfg", {})
    if not preprocess_cfg:
        size = getattr(module, "image_size")
        if size is not None:
            preprocess_cfg["size"] = size
        mean = getattr(module, "image_mean", None)
        if mean is not None:
            preprocess_cfg["mean"] = mean
        std = getattr(module, "image_std", None)
        if std is not None:
            preprocess_cfg["std"] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model: nn.Module, preprocess_cfg: dict[str, Any]) -> None:
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg["mean"]
    module.image_std = preprocess_cfg["std"]
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


def get_model_tokenize_cfg(model: nn.Module) -> dict[str, Any]:
    module = getattr(model, "text", model)
    cfg: dict[str, Any] = {}
    context_length = getattr(module, "context_length", None)
    if context_length is not None:
        cfg["context_length"] = context_length
    vocab_size = getattr(module, "vocab_size", None)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return cfg
