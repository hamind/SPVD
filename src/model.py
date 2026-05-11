"""SPVD model definition."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from clip_components import _build_text_tower, _build_vision_tower, text_global_pool
from conditional_decomposition import SoftCueBidirectionalDecomposition
from soft_cue import SoftCueExtractor


def apply_projection(x: Tensor, projection: Any) -> Tensor:
    """Apply an OpenCLIP-style projection module or parameter matrix."""
    if projection is None:
        return x
    if isinstance(projection, nn.Linear):
        return projection(x)
    return x @ projection


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
            self.soft_cue_decomposition = SoftCueBidirectionalDecomposition(
                visual_dim=self.visual_dim,
                embed_dim=self.embed_dim,
                relevance_temperature=float(cfg.get("relevance_temperature", 1.0)),
                routing_temperature=float(cfg.get("routing_temperature", 1.0)),
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
        return {"text_global": text_global, "text_tokens": text_tokens, "text_attention_mask": text_attention_mask}

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
        text_attention_mask = text_outputs["text_attention_mask"]
        if image_tokens is None:
            raise RuntimeError("Visual encoder did not return patch tokens.")

        caption_text_features = text_global if text_global.ndim == 3 else None
        if text_global.ndim == 3:
            text_features = F.normalize(text_global.mean(dim=1), dim=-1) if self.normalize_outputs else text_global.mean(dim=1)
        else:
            text_features = text_global

        if self.uses_soft_cue_extractor:
            if text_tokens.ndim == 4:
                batch_size, captions_per_sample, context_length, token_dim = text_tokens.shape
                flat_text_tokens = text_tokens.reshape(batch_size * captions_per_sample, context_length, token_dim)
                flat_attention_mask = text_attention_mask.reshape(batch_size * captions_per_sample, context_length)
                flat_soft_cues = self.soft_cue_extractor(flat_text_tokens, attention_mask=flat_attention_mask)
                soft_cues = flat_soft_cues.reshape(batch_size, captions_per_sample, flat_soft_cues.shape[1], flat_soft_cues.shape[2])
            else:
                soft_cues = self.soft_cue_extractor(text_tokens, attention_mask=text_attention_mask)
        else:
            if text_global.ndim == 3:
                soft_cues = text_global.unsqueeze(2).to(dtype=text_tokens.dtype)
            else:
                soft_cues = text_global.unsqueeze(1).to(dtype=text_tokens.dtype)
        decomp_outputs = self.soft_cue_decomposition(image_tokens, soft_cues)
        shared_global = decomp_outputs["shared_visual_features"]
        residual_global = decomp_outputs["residual_visual_features"]
        outputs = {
            "image_features": shared_global,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "z_v_s": shared_global,
            "z_v_p": residual_global,
            "z_t": text_features,
            "shared_visual_features": shared_global,
            "residual_visual_features": residual_global,
            "cue_visual_features": decomp_outputs["cue_visual_features"],
            "cue_residual_features": decomp_outputs["cue_residual_features"],
            "soft_cues": soft_cues,
            "relevance_scores": decomp_outputs["relevance_scores"],
            "shared_routing": decomp_outputs["shared_routing"],
            "residual_routing": decomp_outputs["residual_routing"],
            "cue_weights": decomp_outputs["cue_weights"],
            "image_attention": decomp_outputs["image_attention"],
            "image_tokens": image_tokens if self.return_patch_tokens else None,
            "text_tokens": text_tokens,
            "text_global": text_global,
            "caption_text_features": caption_text_features,
            "caption_shared_visual_features": decomp_outputs.get("caption_shared_visual_features"),
            "caption_residual_visual_features": decomp_outputs.get("caption_residual_visual_features"),
            "image_global": image_global,
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
