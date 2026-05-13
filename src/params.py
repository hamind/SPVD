"""Argument and YAML config handling.

The entrypoint delegates to package-local params, factory, data, and training
modules. YAML files provide defaults, while CLI flags can override them.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Sequence

import yaml


def get_default_params(model_name: str) -> dict[str, float]:
    """CLIP-style default optimizer params keyed by architecture family."""
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    """Parse ``KEY=VALUE`` CLI overrides."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: list[str], option_string: str | None = None) -> None:
        kwargs = {}
        for value in values:
            key, raw_value = value.split("=", 1)
            try:
                kwargs[key] = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                kwargs[key] = raw_value
        setattr(namespace, self.dest, kwargs)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: str | None) -> dict[str, Any]:
    """Read a YAML config if provided, resolving optional ``extends`` chains."""
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    extends = payload.get("extends")
    if not extends:
        return payload
    parents = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _deep_update(merged, _read_yaml(str(parent_path)))
    return _deep_update(merged, payload)


def _default_from_config(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Fetch a nested config default."""
    value = config.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def _default_loss_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Fetch loss defaults from top-level keys or the nested loss section."""
    if key in config:
        return config[key]
    return _default_from_config(config, "loss", key, default)


def _default_training_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Fetch training defaults from top-level keys or the nested training section."""
    if key in config:
        return config[key]
    return _default_from_config(config, "training", key, default)


def _default_model_name(config: dict[str, Any]) -> str:
    """Resolve the OpenCLIP-compatible model name from YAML."""
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    model_name = model_cfg.get("openclip_model") or model_cfg.get("model") or model_cfg.get("name") or "ViT-B-16"
    aliases = {
        "clip_vitb16": "ViT-B-16",
        "vitb16": "ViT-B-16",
    }
    return aliases.get(str(model_name), str(model_name))


def _default_loss_is_siglip(config: dict[str, Any]) -> bool:
    """Infer whether SigLIP loss should be enabled from YAML."""
    return _default_loss_name(config) in {"siglip", "sigmoid", "sigmoid_pairwise"}


def _default_loss_name(config: dict[str, Any]) -> str:
    """Resolve the loss name from either OpenCLIP-style or flat config keys."""
    loss_cfg = config.get("loss", {}) if isinstance(config.get("loss", {}), dict) else {}
    return str(config.get("loss_name") or loss_cfg.get("name", "clip")).lower()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with optional YAML defaults."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    config = _read_yaml(config_args.config)

    data_type = _default_from_config(config, "data", "type", "auto")
    if data_type == "manifest":
        data_type = _default_from_config(config, "data", "file_format", "csv")

    model_name = _default_model_name(config)
    default_optim = get_default_params(model_name)
    optim_betas = _default_from_config(config, "optim", "betas")
    beta1 = optim_betas[0] if isinstance(optim_betas, list) and len(optim_betas) > 0 else default_optim["beta1"]
    beta2 = optim_betas[1] if isinstance(optim_betas, list) and len(optim_betas) > 1 else default_optim["beta2"]

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--train-data", default=_default_from_config(config, "data", "train_data"))
    parser.add_argument("--val-data", default=_default_from_config(config, "data", "val_data"))
    parser.add_argument("--train-num-samples", type=int, default=_default_from_config(config, "data", "train_num_samples"))
    parser.add_argument("--val-num-samples", type=int, default=_default_from_config(config, "data", "val_num_samples"))
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "tsv", "json", "jsonl", "parquet", "synthetic", "auto"], default=data_type)
    parser.add_argument("--image-root", default=_default_from_config(config, "data", "image_root") or _default_from_config(config, "data", "root"))
    parser.add_argument("--csv-img-key", default=_default_from_config(config, "data", "image_key", "filepath"))
    parser.add_argument("--csv-caption-key", default=_default_from_config(config, "data", "caption_key", "title"))
    parser.add_argument("--csv-separator", default="\t" if data_type == "tsv" else ",")
    parser.add_argument("--image-key", default=_default_from_config(config, "data", "image_key", "jpg"))
    parser.add_argument("--caption-key", default=_default_from_config(config, "data", "caption_key", "txt"))
    parser.add_argument("--metadata-key", default=_default_from_config(config, "data", "metadata_key"))
    parser.add_argument("--filter-relabel-success", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "data", "filter_relabel_success", False)))
    parser.add_argument("--relabel-success-key", default=_default_from_config(config, "data", "relabel_success_key", "longSV"))
    parser.add_argument("--strict-caption-match", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "data", "strict_caption_match", True)))
    parser.add_argument("--num-sampled-captions", type=int, default=_default_from_config(config, "data", "num_sampled_captions", 4))
    parser.add_argument("--max-merged-num", type=int, default=_default_from_config(config, "data", "max_merged_num", 3))
    parser.add_argument("--tar-mode", default=_default_from_config(config, "data", "tar_mode", "r|*"))
    parser.add_argument("--caption-relabel-file", "--long-caption-file", dest="caption_relabel_file", default=_default_from_config(config, "data", "caption_relabel_file") or _default_from_config(config, "data", "long_caption_file"))
    parser.add_argument("--caption-relabel-index", "--long-caption-index", dest="caption_relabel_index", default=_default_from_config(config, "data", "caption_relabel_index") or _default_from_config(config, "data", "long_caption_index"))
    parser.add_argument("--caption-relabel-file-key", "--long-caption-file-key", dest="caption_relabel_file_key", default=_default_from_config(config, "data", "caption_relabel_file_key", _default_from_config(config, "data", "long_caption_file_key", "Image Path")))
    parser.add_argument("--caption-relabel-caption-key", "--long-caption-caption-key", dest="caption_relabel_caption_key", default=_default_from_config(config, "data", "caption_relabel_caption_key", _default_from_config(config, "data", "long_caption_caption_key", "longSV_captions")))
    parser.add_argument("--caption-relabel-fallback-keys", "--long-caption-fallback-keys", dest="caption_relabel_fallback_keys", nargs="*", default=_default_from_config(config, "data", "caption_relabel_fallback_keys", _default_from_config(config, "data", "long_caption_fallback_keys", [])))
    parser.add_argument("--caption-relabel-sample-key", "--long-caption-sample-key", dest="caption_relabel_sample_key", choices=["url", "key", "caption"], default=_default_from_config(config, "data", "caption_relabel_sample_key", _default_from_config(config, "data", "long_caption_sample_key", "url")))
    parser.add_argument("--caption-relabel-missing", "--long-caption-missing", dest="caption_relabel_missing", choices=["fallback", "skip", "error"], default=_default_from_config(config, "data", "caption_relabel_missing", _default_from_config(config, "data", "long_caption_missing", "fallback")))
    parser.add_argument("--caption-relabel-build-index", "--long-caption-build-index", dest="caption_relabel_build_index", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "data", "caption_relabel_build_index", _default_from_config(config, "data", "long_caption_build_index", True))))
    parser.add_argument("--batch-size", type=int, default=_default_from_config(config, "data", "batch_size", 256))
    parser.add_argument("--workers", type=int, default=_default_from_config(config, "data", "num_workers", 8))
    parser.add_argument("--prefetch-factor", type=int, default=_default_from_config(config, "data", "prefetch_factor", 2))
    parser.add_argument("--persistent-workers", action="store_true", default=bool(_default_from_config(config, "data", "persistent_workers", False)))
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "data", "pin_memory", True)))
    parser.add_argument("--sample-shuffle-size", type=int, default=_default_from_config(config, "data", "sample_shuffle_size", 5000))
    parser.add_argument("--sample-shuffle-initial", type=int, default=_default_from_config(config, "data", "sample_shuffle_initial", 1000))
    parser.add_argument("--shard-shuffle-size", type=int, default=_default_from_config(config, "data", "shard_shuffle_size", 2000))
    parser.add_argument("--shard-shuffle-initial", type=int, default=_default_from_config(config, "data", "shard_shuffle_initial", 500))

    parser.add_argument("--model", default=model_name)
    parser.add_argument("--pretrained", default=_default_from_config(config, "model", "pretrained", ""))
    parser.add_argument("--embed-dim", type=int, default=_default_from_config(config, "model", "embed_dim", 512))
    parser.add_argument("--image-size", "--force-image-size", dest="image_size", type=int, default=_default_from_config(config, "model", "image_size", 224))
    parser.add_argument("--precision", default="amp" if _default_from_config(config, "optim", "amp", True) else "fp32", choices=["amp", "fp32", "bf16", "amp_bf16"])
    parser.add_argument("--grad-checkpointing", action="store_true", default=bool(_default_from_config(config, "model", "grad_checkpointing", False)))
    parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)

    parser.add_argument("--loss-name", choices=["clip", "spvd", "siglip", "sigmoid", "sigmoid_pairwise"], default=_default_loss_name(config))
    parser.add_argument("--siglip", action="store_true", default=_default_loss_is_siglip(config))
    parser.add_argument("--loss-dist-impl", default=_default_loss_value(config, "dist_impl", "bidir"))
    parser.add_argument("--local-loss", action="store_true", default=bool(_default_loss_value(config, "local_loss", True)))
    parser.add_argument("--gather-with-grad", action="store_true", default=bool(_default_loss_value(config, "gather_with_grad", True)))
    parser.add_argument("--align-weight", type=float, default=_default_loss_value(config, "align_weight", 1.0))
    parser.add_argument("--global-align-weight", type=float, default=_default_loss_value(config, "global_align_weight", 1.0))
    parser.add_argument("--caption-align-weight", type=float, default=_default_loss_value(config, "caption_align_weight", 1.0))
    parser.add_argument("--caption-same-image-mode", default=_default_loss_value(config, "caption_same_image_mode", "ignore"))
    parser.add_argument("--branch-bce-weight", type=float, default=_default_loss_value(config, "branch_bce_weight", 0.0))
    parser.add_argument("--branch-logit-scale", type=float, default=_default_loss_value(config, "branch_logit_scale", 5.0))
    parser.add_argument("--residual-negative-weight", type=float, default=_default_loss_value(config, "residual_negative_weight", 0.25))
    parser.add_argument("--detach-text-for-residual", action=argparse.BooleanOptionalAction, default=bool(_default_loss_value(config, "detach_text_for_residual", True)))
    parser.add_argument("--residual-variance-weight", type=float, default=_default_loss_value(config, "residual_variance_weight", 0.0))
    parser.add_argument("--residual-variance-gamma", type=float, default=_default_loss_value(config, "residual_variance_gamma", 1.0))

    parser.add_argument("--epochs", type=int, default=_default_from_config(config, "optim", "epochs", 32))
    parser.add_argument("--lr", type=float, default=_default_from_config(config, "optim", "lr", default_optim["lr"]))
    parser.add_argument("--wd", "--weight-decay", dest="wd", type=float, default=_default_from_config(config, "optim", "weight_decay", 0.2))
    parser.add_argument("--beta1", type=float, default=beta1)
    parser.add_argument("--beta2", type=float, default=beta2)
    parser.add_argument("--eps", type=float, default=_default_from_config(config, "optim", "eps", default_optim["eps"]))
    parser.add_argument("--warmup", type=int, default=_default_from_config(config, "optim", "warmup_steps", 2000))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=bool(_default_training_value(config, "torch_compile", False)))
    parser.add_argument("--torch-compile-backend", default=_default_training_value(config, "torch_compile_backend", "inductor"))
    parser.add_argument("--torch-compile-mode", default=_default_training_value(config, "torch_compile_mode", None))

    parser.add_argument("--logs", "--logs-dir", dest="logs_dir", default=_default_from_config(config, "experiment", "output_dir", "outputs"))
    parser.add_argument("--name", default=_default_from_config(config, "experiment", "name"))
    parser.add_argument("--save-frequency", type=int, default=_default_from_config(config, "logging", "save_interval", 1), help="Save epoch checkpoints every N epochs; the final checkpoint is always saved.")
    parser.add_argument("--log-every-n-steps", type=int, default=_default_from_config(config, "logging", "log_interval", 50))
    parser.add_argument("--report-to", default=_default_from_config(config, "logging", "report_to", "tensorboard"))
    parser.add_argument("--diagnostics-enabled", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "diagnostics", "enabled", True)))
    parser.add_argument("--diagnostics-log-interval", type=int, default=_default_from_config(config, "diagnostics", "log_interval", 50))
    parser.add_argument("--diagnostics-heavy-log-interval", type=int, default=_default_from_config(config, "diagnostics", "heavy_log_interval", 500))
    parser.add_argument("--diagnostics-per-layer", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "diagnostics", "per_layer", True)))
    parser.add_argument("--diagnostics-log-update-rms", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "diagnostics", "log_update_rms", True)))
    parser.add_argument("--diagnostics-log-effective-rank", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "diagnostics", "log_effective_rank", True)))
    parser.add_argument("--diagnostics-log-redundancy", action=argparse.BooleanOptionalAction, default=bool(_default_from_config(config, "diagnostics", "log_redundancy", True)))
    parser.add_argument("--diagnostics-max-logged-layers", type=int, default=_default_from_config(config, "diagnostics", "max_logged_layers", 80))
    parser.add_argument("--diagnostics-eps", type=float, default=_default_from_config(config, "diagnostics", "eps", 1.0e-12))
    parser.add_argument(
        "--debug-finite-checks",
        action=argparse.BooleanOptionalAction,
        default=bool(
            _default_from_config(
                config,
                "training",
                "debug_finite_checks",
                _default_from_config(config, "logging", "debug_finite_checks", False),
            )
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=_default_from_config(config, "experiment", "seed", 42))
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--dist-backend", default=_default_from_config(config, "distributed", "backend", "nccl"))
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--horovod", action="store_true", default=False)
    args = parser.parse_args(remaining, namespace=config_args)
    args.config_dict = config
    return args
