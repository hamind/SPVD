"""SPVD model definition."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import partial
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


class SoftCueSigmoidDecomposition(nn.Module):
    """Text-conditioned visual decomposition with sigmoid-gated token pooling."""

    def __init__(
        self,
        visual_dim: int,
        embed_dim: int,
        gate_temperature: float = 1.0,
        gate_bias_init: float = 0.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        self.visual_dim = int(visual_dim)
        self.embed_dim = int(embed_dim)
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.semantic_value_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.residual_value_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))
        self.out_semantic_norm = nn.LayerNorm(self.embed_dim)
        self.out_residual_norm = nn.LayerNorm(self.embed_dim)
        self.out_semantic_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_residual_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, visual_tokens: Tensor, soft_cues: Tensor) -> dict[str, Tensor]:
        """Decompose visual tokens [B, M, D_v] using cues [B, S, D] or [B, K, S, D]."""
        if soft_cues.ndim not in {3, 4}:
            raise ValueError("soft_cues must have shape [B, S, D] or [B, K, S, D].")
        single_caption = soft_cues.ndim == 3
        if single_caption:
            soft_cues = soft_cues.unsqueeze(1)

        visual_tokens = visual_tokens.to(dtype=self.key_proj.weight.dtype)
        soft_cues = soft_cues.to(device=visual_tokens.device, dtype=self.query_proj.weight.dtype)

        queries = self.query_proj(soft_cues)
        keys = self.key_proj(visual_tokens)
        semantic_values = self.semantic_value_proj(visual_tokens)
        residual_values = self.residual_value_proj(visual_tokens)
        scale = math.sqrt(float(self.embed_dim))

        similarities = torch.einsum("bksd,bnd->bksn", queries, keys) / scale
        gate_logits = (similarities + self.gate_bias.to(dtype=similarities.dtype)) / self.gate_temperature

        sigmoid_map = torch.sigmoid(gate_logits)
        residual_map = 1.0 - sigmoid_map

        semantic_den = sigmoid_map.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        residual_den = residual_map.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        cue_semantic = torch.einsum("bksn,bnd->bksd", sigmoid_map, semantic_values) / semantic_den
        cue_residual = torch.einsum("bksn,bnd->bksd", residual_map, residual_values) / residual_den

        semantic_features = cue_semantic.mean(dim=2)
        residual_features = cue_residual.mean(dim=2)

        semantic_features = self.out_semantic_proj(self.out_semantic_norm(semantic_features))
        residual_features = self.out_residual_proj(self.out_residual_norm(residual_features))

        if single_caption:
            cue_semantic = cue_semantic.squeeze(1)
            cue_residual = cue_residual.squeeze(1)
            semantic_features = semantic_features.squeeze(1)
            residual_features = residual_features.squeeze(1)
            sigmoid_map = sigmoid_map.squeeze(1)
            residual_map = residual_map.squeeze(1)
            gate_logits = gate_logits.squeeze(1)

        return {
            "semantic_features": semantic_features,
            "residual_features": residual_features,
            "sigmoid_map": sigmoid_map,
            "residual_map": residual_map,
            "gate_logits": gate_logits,
        }


class SPVDModel(nn.Module):
    """SPVD bi-encoder with an optional soft-cue decomposition branch."""

    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: dict[str, Any],
        text_cfg: dict[str, Any],
        spvd_cfg: dict[str, Any] | None = None,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: float | None = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: torch.dtype | None = None,
        output_dict: bool = False,
    ) -> None:
        super().__init__()
        cfg = spvd_cfg or {}
        vision_cfg = dict(vision_cfg)
        vision_cfg["output_tokens"] = True

        self.output_dict = bool(output_dict)
        self.embed_dim = int(embed_dim)
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.text_eos_id = getattr(text, "eos_id", None)
        self.text_pad_id = int(getattr(text, "pad_id", 0))
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

        self.return_patch_tokens = bool(cfg.get("return_patch_tokens", True))
        self.use_global_image_head = bool(cfg.get("use_global_image_head", True))
        self.normalize_outputs = bool(cfg.get("normalize_outputs", cfg.get("normalize", True)))
        self.visual_dim = self._infer_visual_token_dim()
        self.text_dim = int(self.token_embedding.embedding_dim)
        self.embed_dim = int(embed_dim)

        self.use_finegrained_text_cue = bool(cfg.get("use_finegrained_text_cue", True))
        self.text_cue_type = str(cfg.get("text_cue_type", "soft_cue")).lower()
        self.enable_soft_cue_decomp = bool(cfg.get("enable_soft_cue_decomp", False))
        self.uses_soft_cue_extractor = self.use_finegrained_text_cue and self.text_cue_type not in {"pooled", "eot", "global"}
        if self.enable_soft_cue_decomp:
            self._freeze_unused_visual_projection()
            if self.uses_soft_cue_extractor:
                self.soft_cue_extractor = SoftCueExtractor(
                    text_dim=self.text_dim,
                    embed_dim=self.embed_dim,
                    num_soft_cues=int(cfg.get("num_soft_cues", 4)),
                    num_heads=int(cfg.get("soft_cue_num_heads", 4)),
                    num_layers=int(cfg.get("soft_cue_num_layers", 1)),
                    dropout=float(cfg.get("soft_cue_dropout", 0.0)),
                )
            self.soft_cue_decomposition = SoftCueSigmoidDecomposition(
                visual_dim=self.visual_dim,
                embed_dim=self.embed_dim,
                gate_temperature=float(cfg.get("gate_temperature", cfg.get("routing_temperature", 1.0))),
                gate_bias_init=float(cfg.get("gate_bias_init", 0.0)),
            )

    def _freeze_unused_visual_projection(self) -> None:
        """Freeze the global visual projection when SPVD only trains token routes."""
        visual_projection = getattr(self.visual, "proj", None)
        if isinstance(visual_projection, nn.Parameter):
            visual_projection.requires_grad_(False)
        elif isinstance(visual_projection, nn.Module):
            for parameter in visual_projection.parameters():
                parameter.requires_grad_(False)

    def lock_image_tower(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False) -> None:
        """OpenCLIP-compatible image tower freezing hook."""
        lock = getattr(self.visual, "lock", None)
        if callable(lock):
            lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable checkpointing on towers that expose the OpenCLIP hook."""
        visual_setter = getattr(self.visual, "set_grad_checkpointing", None)
        if callable(visual_setter):
            visual_setter(enable)
        if hasattr(self.transformer, "grad_checkpointing"):
            self.transformer.grad_checkpointing = enable

    def _infer_visual_token_dim(self) -> int:
        """Infer the local visual-token width from the OpenCLIP visual tower."""
        class_embedding = getattr(self.visual, "class_embedding", None)
        if class_embedding is not None:
            return int(class_embedding.shape[-1])
        conv1 = getattr(self.visual, "conv1", None)
        if conv1 is not None:
            return int(conv1.out_channels)
        raise AttributeError("Unable to infer OpenCLIP visual token width.")

    def _text_cast_dtype(self) -> torch.dtype:
        get_cast_dtype = getattr(self.transformer, "get_cast_dtype", None)
        if callable(get_cast_dtype):
            return get_cast_dtype()
        return self.token_embedding.weight.dtype

    def _text_attn_mask(self, length: int, device: torch.device) -> Tensor | None:
        attn_mask = getattr(self, "attn_mask", None)
        if attn_mask is None:
            return None
        return attn_mask[:length, :length].to(device=device)

    def encode_image(
        self,
        image: Tensor,
        normalize: bool = False,
        return_tokens: bool = False,
        return_patch_tokens: bool | None = None,
    ) -> Tensor | dict[str, Tensor | None]:
        """OpenCLIP-style image encoding with optional local visual tokens.

        The visual tower itself returns the pooled image representation and local
        image tokens; this method only decides whether to expose the token path.
        """
        if return_patch_tokens is not None:
            return_tokens = bool(return_patch_tokens)
        visual_outputs = self.visual(image)
        if isinstance(visual_outputs, tuple):
            image_global, image_tokens = visual_outputs
        else:
            image_global, image_tokens = visual_outputs, None

        if normalize and image_global is not None:
            image_global = F.normalize(image_global, dim=-1)
        if not return_tokens:
            return image_global
        if image_tokens is None:
            raise RuntimeError("OpenCLIP visual tower did not return local image tokens.")
        return {
            "image_global": image_global if self.use_global_image_head else None,
            "image_tokens": image_tokens,
        }

    def encode_text(
        self,
        text: Tensor,
        normalize: bool = False,
        return_tokens: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """OpenCLIP-style text encoding with optional multi-caption input.

        Text may be shaped [B, L] or [B, K, L]. Multi-caption
        inputs are flattened to [B*K, L] for the text tower, then pooled
        back to one text feature per image while exposing all caption tokens to
        the SPVD cue extractor.
        """
        if text.ndim not in {2, 3}:
            raise ValueError(f"Text must have shape [B, L] or [B, K, L], got {tuple(text.shape)}.")

        multi_caption = text.ndim == 3
        if multi_caption:
            batch_size, captions_per_sample, context_length = text.shape
            text_for_encoder = text.reshape(batch_size * captions_per_sample, context_length)
        else:
            batch_size, context_length = text.shape
            captions_per_sample = 1
            text_for_encoder = text

        if context_length > int(self.positional_embedding.shape[0]):
            raise ValueError(f"Text length {context_length} exceeds context length {int(self.positional_embedding.shape[0])}.")
        text_attention_mask = text_for_encoder.ne(self.text_pad_id)

        cast_dtype = self._text_cast_dtype()
        x = self.token_embedding(text_for_encoder).to(cast_dtype)
        x = x + self.positional_embedding[:context_length].to(device=x.device, dtype=cast_dtype)
        text_tokens = self.transformer(x, attn_mask=self._text_attn_mask(context_length, x.device))
        text_tokens = self.ln_final(text_tokens)

        text_global = text_global_pool(
            text_tokens,
            text_for_encoder,
            self.text_pool_type,
            eos_token_id=getattr(self, "text_eos_id", None),
        )
        text_global = apply_projection(text_global, self.text_projection)

        if multi_caption:
            text_tokens = text_tokens.reshape(batch_size, captions_per_sample, context_length, text_tokens.shape[-1])
            text_global = text_global.reshape(batch_size, captions_per_sample, -1)
            text_attention_mask = text_attention_mask.reshape(batch_size, captions_per_sample, context_length)
        if normalize:
            text_global = F.normalize(text_global, dim=-1)

        if not return_tokens:
            return text_global

        outputs = {
            "text_global": text_global,
            "text_tokens": text_tokens,
            "text_attention_mask": text_attention_mask,
        }
        if self.enable_soft_cue_decomp:
            if self.uses_soft_cue_extractor:
                if text_tokens.ndim == 4:
                    batch_size, captions_per_sample, context_length, token_dim = text_tokens.shape
                    flat_text_tokens = text_tokens.reshape(batch_size * captions_per_sample, context_length, token_dim)
                    flat_attention_mask = text_attention_mask.reshape(batch_size * captions_per_sample, context_length)
                    flat_cue = self.soft_cue_extractor(flat_text_tokens, attention_mask=flat_attention_mask)
                    cue = flat_cue.reshape(batch_size, captions_per_sample, flat_cue.shape[1], flat_cue.shape[2])
                else:
                    cue = self.soft_cue_extractor(text_tokens, attention_mask=text_attention_mask)
            else:
                if text_global.ndim == 3:
                    cue = text_global.unsqueeze(2).to(dtype=text_tokens.dtype)
                else:
                    cue = text_global.unsqueeze(1).to(dtype=text_tokens.dtype)
            outputs["cue"] = cue
            outputs["soft_cues"] = cue
        return outputs

    def get_logits_as_clip(self, image: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        """Return global CLIP-style logits."""
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if not torch.is_tensor(image_features) or not torch.is_tensor(text_features):
            raise RuntimeError("Global CLIP logits require tensor image and text features.")
        if text_features.ndim == 3:
            text_features = F.normalize(text_features.mean(dim=1), dim=-1)
        image_logits = self.logit_scale.exp() * image_features @ text_features.t()
        if self.logit_bias is not None:
            image_logits = image_logits + self.logit_bias
        return image_logits, image_logits.t()

    def get_logits(self, image: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        """Expose the standard global scoring path for interface compatibility."""
        return self.get_logits_as_clip(image, text)

    def _maybe_normalize(self, value: Tensor | None) -> Tensor | None:
        if value is None or not self.normalize_outputs:
            return value
        return F.normalize(value, dim=-1)

    def forward_spvd(self, image: Tensor, text: Tensor) -> dict[str, Any]:
        """Return CLIP-style features, with optional soft-cue decomposition."""
        if not self.enable_soft_cue_decomp:
            image_features = self.encode_image(image, normalize=self.normalize_outputs)
            text_features = self.encode_text(text, normalize=self.normalize_outputs)
            if not torch.is_tensor(image_features) or not torch.is_tensor(text_features):
                raise RuntimeError("Baseline SPVD forward requires tensor image and text features.")
            if text_features.ndim == 3:
                text_features = F.normalize(text_features.mean(dim=1), dim=-1) if self.normalize_outputs else text_features.mean(dim=1)
            outputs: dict[str, Any] = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                outputs["logit_bias"] = self.logit_bias
            return outputs

        image_outputs = self.encode_image(image, normalize=False, return_tokens=True)
        text_outputs = self.encode_text(text, normalize=self.normalize_outputs, return_tokens=True)
        image_tokens = image_outputs["image_tokens"]
        image_global = image_outputs["image_global"]
        text_tokens = text_outputs["text_tokens"]
        text_global = text_outputs["text_global"]
        soft_cues = text_outputs.get("cue")
        if soft_cues is None:
            soft_cues = text_outputs.get("soft_cues")
        if soft_cues is None:
            raise RuntimeError("encode_text must return cue when soft-cue decomposition is enabled.")
        if image_tokens is None:
            raise RuntimeError("Visual encoder did not return patch tokens.")

        caption_text_features = text_global if text_global.ndim == 3 else None
        if text_global.ndim == 3:
            text_features = F.normalize(text_global.mean(dim=1), dim=-1) if self.normalize_outputs else text_global.mean(dim=1)
        else:
            text_features = text_global

        decomp_outputs = self.soft_cue_decomposition(image_tokens, soft_cues)
        shared_global = decomp_outputs["semantic_features"]
        residual_global = decomp_outputs["residual_features"]
        outputs = {
            "image_features": shared_global,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "shared_visual_features": shared_global,
            "residual_visual_features": residual_global,
            "caption_text_features": caption_text_features,
            "sigmoid_map": decomp_outputs["sigmoid_map"],
            "residual_map": decomp_outputs["residual_map"],
            "gate_logits": decomp_outputs["gate_logits"],
        }
        if self.logit_bias is not None:
            outputs["logit_bias"] = self.logit_bias
        return outputs

    def forward(
        self,
        image: Tensor | None = None,
        text: Tensor | None = None,
        output_dict: bool | None = None,
    ) -> dict[str, Any] | tuple[Tensor | None, Tensor | None, Tensor]:
        """OpenCLIP-compatible forward with an SPVD dictionary extension."""
        use_dict = self.output_dict if output_dict is None else bool(output_dict)
        if image is None or text is None:
            image_features = self.encode_image(image, normalize=True) if image is not None else None
            text_features = self.encode_text(text, normalize=True) if text is not None else None
            if use_dict:
                out: dict[str, Any] = {
                    "image_features": image_features,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale.exp(),
                }
                if self.logit_bias is not None:
                    out["logit_bias"] = self.logit_bias
                return out
            return image_features, text_features, self.logit_scale.exp()

        outputs = self.forward_spvd(image, text)
        if use_dict:
            return outputs
        if self.logit_bias is not None:
            return outputs["image_features"], outputs["text_features"], outputs["logit_scale"], self.logit_bias
        return outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
