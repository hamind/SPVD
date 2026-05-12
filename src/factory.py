"""Factories for SPVD models, OpenCLIP baselines, losses, optimizers, and tokenizers."""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs"]
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {}


def _natural_key(string_: str) -> list[int | str]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs() -> None:
    """Populate the local model architecture registry from JSON configs."""
    global _MODEL_CONFIGS
    config_files: list[Path] = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix == ".json":
            config_files.append(config_path)
        elif config_path.is_dir():
            config_files.extend(config_path.glob("*.json"))

    configs: dict[str, dict[str, Any]] = {}
    for cfg_file in config_files:
        with cfg_file.open("r", encoding="utf-8") as handle:
            model_cfg = json.load(handle)
        if all(key in model_cfg for key in ("embed_dim", "vision_cfg", "text_cfg")) and "spvd_cfg" in model_cfg:
            configs[cfg_file.stem] = model_cfg
    _MODEL_CONFIGS = {k: v for k, v in sorted(configs.items(), key=lambda item: _natural_key(item[0]))}


_rescan_model_configs()


def list_models() -> list[str]:
    """Enumerate locally registered SPVD architecture names."""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path: str | Path) -> None:
    """Add a model config file or directory and refresh the local registry."""
    _MODEL_CONFIG_PATHS.append(Path(path))
    _rescan_model_configs()


def get_model_config(model_name: str) -> dict[str, Any] | None:
    """Return a deep copy of a locally registered SPVD model config."""
    return deepcopy(_MODEL_CONFIGS.get(model_name))


def _local_base_tokenizer_name(model_name: str) -> str:
    cfg = get_model_config(_normalize_model_name(model_name))
    if cfg is None:
        return _normalize_model_name(model_name)
    return str(cfg.get("tokenizer_model") or cfg.get("openclip_model") or "ViT-B-16")


def _tokenizer_context_length(model_name: str) -> int:
    model_name = _normalize_model_name(model_name)
    cfg = get_model_config(model_name)
    if cfg is not None:
        text_cfg = cfg.get("text_cfg", {})
        if isinstance(text_cfg, dict) and "context_length" in text_cfg:
            return int(text_cfg["context_length"])

    try:
        import open_clip

        base_cfg = open_clip.get_model_config(_local_base_tokenizer_name(model_name))
        text_cfg = base_cfg.get("text_cfg", {}) if isinstance(base_cfg, dict) else {}
        if isinstance(text_cfg, dict) and "context_length" in text_cfg:
            return int(text_cfg["context_length"])
    except Exception:
        pass
    return 77


def _normalize_model_name(model_name: str) -> str:
    """Normalize project aliases without routing OpenCLIP CLIP into local configs."""
    model_name = str(model_name).replace("/", "-")
    aliases = {
        "clip_vitb16": "ViT-B-16",
        "vitb16": "ViT-B-16",
        "CLIP-ViT-B-16": "ViT-B-16",
    }
    return aliases.get(model_name, model_name)


def _create_spvd_model(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    force_image_size: int | tuple[int, int] | None = None,
    output_dict: bool | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    """Create the local SPVD model directly from a project JSON architecture config."""
    from open_clip.factory import load_checkpoint
    from open_clip.transform import PreprocessCfg, merge_preprocess_dict

    from model import SPVDModel, convert_weights_to_lp, get_cast_dtype, set_model_preprocess_cfg

    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise RuntimeError(f"SPVD model config for {model_name} not found.")
    if force_image_size is not None:
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    spvd_override = model_kwargs.pop("spvd_cfg", None)
    if spvd_override:
        model_cfg.setdefault("spvd_cfg", {}).update(spvd_override)
    model_cfg.update(model_kwargs)
    model_cfg.pop("tokenizer_model", None)
    model_cfg.pop("openclip_model", None)
    model_cfg.pop("model_config", None)

    cast_dtype = get_cast_dtype(precision)
    if output_dict is not None:
        model_cfg["output_dict"] = bool(output_dict)
    model = SPVDModel(**model_cfg, cast_dtype=cast_dtype)

    device = torch.device(device)
    if precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        model.to(device=device)
        convert_weights_to_lp(model, dtype=dtype)
    elif precision in {"pure_fp16", "pure_bf16"}:
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    if pretrained:
        if os.path.exists(pretrained):
            logging.info("Loading local model weights from %s", pretrained)
            load_checkpoint(model, pretrained, strict=False)
        else:
            raise RuntimeError(f"Pretrained weights not found for SPVD model {model_name}: {pretrained}")

    preprocess_cfg = asdict(PreprocessCfg())
    if getattr(model.visual, "image_size", None) is not None:
        preprocess_cfg["size"] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(asdict(PreprocessCfg()), preprocess_cfg))
    return model


def create_model(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    force_image_size: int | tuple[int, int] | None = None,
    output_dict: bool | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    """Create SPVD locally and delegate ordinary CLIP/SigLIP models to OpenCLIP."""
    model_name = _normalize_model_name(model_name)
    config_dict = model_kwargs.pop("config_dict", {}) or {}
    siglip = bool(model_kwargs.pop("siglip", False))
    model_cfg = config_dict.get("model", {}) if isinstance(config_dict, dict) and isinstance(config_dict.get("model", {}), dict) else {}
    loss_cfg = config_dict.get("loss", {}) if isinstance(config_dict, dict) and isinstance(config_dict.get("loss", {}), dict) else {}
    align_loss = str(loss_cfg.get("align_loss", "")).lower()
    sigmoid_alignment = siglip or align_loss in {"siglip", "sigmoid", "sigmoid_pairwise"}

    if model_name in _MODEL_CONFIGS:
        for key in ("init_logit_scale", "init_logit_bias", "nonscalar_logit_scale"):
            if key in model_cfg:
                model_kwargs.setdefault(key, model_cfg[key])
        if sigmoid_alignment:
            model_kwargs.setdefault("init_logit_bias", -10)
        spvd_cfg = deepcopy(model_cfg.get("spvd", {}))
        for key in (
            "enable_soft_cue_decomp",
            "use_finegrained_text_cue",
            "text_cue_type",
            "num_soft_cues",
            "soft_cue_num_heads",
            "soft_cue_num_layers",
            "soft_cue_dropout",
            "relevance_temperature",
            "routing_temperature",
            "use_global_image_head",
            "normalize_outputs",
            "return_patch_tokens",
        ):
            if key in model_cfg:
                spvd_cfg[key] = model_cfg[key]
        if spvd_cfg:
            model_kwargs["spvd_cfg"] = spvd_cfg
        return _create_spvd_model(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            force_image_size=force_image_size,
            output_dict=output_dict,
            **model_kwargs,
        )

    import open_clip

    openclip_kwargs: dict[str, Any] = {}
    if siglip:
        openclip_kwargs["init_logit_scale"] = np.log(10)
        openclip_kwargs["init_logit_bias"] = -10
    return open_clip.create_model(
        model_name,
        pretrained=str(pretrained or ""),
        precision=precision,
        device=device,
        force_image_size=force_image_size,
        output_dict=output_dict,
        **openclip_kwargs,
        **model_kwargs,
    )


def create_model_and_transforms(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    force_image_size: int | tuple[int, int] | None = None,
    output_dict: bool | None = None,
    **model_kwargs: Any,
) -> tuple[nn.Module, Any, Any]:
    """Create a model and preprocessing transforms by architecture name."""
    from open_clip.transform import AugmentationCfg, PreprocessCfg, image_transform_v2

    aug_cfg = model_kwargs.pop("aug_cfg", None)
    model_kwargs.pop("image_mean", None)
    model_kwargs.pop("image_std", None)
    model_kwargs.pop("image_interpolation", None)
    model_kwargs.pop("image_resize_mode", None)
    model = create_model(
        str(model_name),
        pretrained=pretrained,
        precision=precision,
        device=device,
        force_image_size=force_image_size,
        output_dict=output_dict,
        **model_kwargs,
    )
    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)
    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    preprocess_train = image_transform_v2(pp_cfg, is_train=True, aug_cfg=aug_cfg)
    preprocess_val = image_transform_v2(pp_cfg, is_train=False)
    return model, preprocess_train, preprocess_val


def create_tokenizer(model_name: str = "ViT-B-16", **kwargs: Any):
    """Create a cached project-local BPE tokenizer."""
    from tokenizer import get_bpe_tokenizer

    kwargs.setdefault("context_length", _tokenizer_context_length(_normalize_model_name(model_name)))
    return get_bpe_tokenizer(**kwargs)


get_tokenizer = create_tokenizer


def create_loss(args: object) -> nn.Module:
    """Create CLIP, SigLIP, or SPVD loss."""
    from open_clip.loss import SigLipLoss
    from losses import InfoNCELoss, SPVDLoss

    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))
    loss_name = str(getattr(args, "loss_name", "clip")).lower()
    if loss_name == "spvd":
        return SPVDLoss(
            local_loss=bool(getattr(args, "local_loss", True)),
            gather_with_grad=bool(getattr(args, "gather_with_grad", True)),
            cache_labels=True,
            rank=rank,
            world_size=world_size,
            decomp_loss_weight=float(getattr(args, "decomp_loss_weight", 0.0)) if bool(getattr(args, "use_route_loss", True)) else 0.0,
            route_positive_constraint=bool(getattr(args, "route_positive_constraint", True)),
            route_negative_constraint=bool(getattr(args, "route_negative_constraint", True)),
            residual_loss_weight=float(getattr(args, "residual_loss_weight", 0.0)),
            orth_loss_weight=float(getattr(args, "orth_loss_weight", 0.0)),
            detach_relevance=bool(getattr(args, "detach_relevance", True)),
            residual_variance_gamma=float(getattr(args, "residual_variance_gamma", 1.0)),
            align_weight=float(getattr(args, "align_weight", 1.0)),
            global_align_weight=float(getattr(args, "global_align_weight", 1.0)),
            caption_align_weight=float(getattr(args, "caption_align_weight", 1.0)),
            loss_dist_impl=getattr(args, "loss_dist_impl", None),
            debug_finite_checks=bool(getattr(args, "debug_finite_checks", False)),
        )
    if bool(getattr(args, "siglip", False)) or loss_name in {"siglip", "sigmoid", "sigmoid_pairwise"}:
        return SigLipLoss(
            cache_labels=True,
            rank=rank,
            world_size=world_size,
            dist_impl=getattr(args, "loss_dist_impl", None),
        )
    return InfoNCELoss(
        local_loss=bool(getattr(args, "local_loss", True)),
        gather_with_grad=bool(getattr(args, "gather_with_grad", True)),
        cache_labels=True,
        rank=rank,
        world_size=world_size,
    )


def create_optimizer(model: nn.Module, args: object) -> torch.optim.Optimizer:
    """Create AdamW with CLIP-style optimizer defaults."""
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": float(getattr(args, "wd", 0.2))},
        ],
        lr=float(getattr(args, "lr", 5.0e-4)),
        betas=(float(getattr(args, "beta1", 0.9)), float(getattr(args, "beta2", 0.98))),
        eps=float(getattr(args, "eps", 1.0e-6)),
    )
