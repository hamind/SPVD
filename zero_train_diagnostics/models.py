from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .schema import ModelLoadResult


def _read_checkpoint_state_dict(checkpoint_path: Path, weights_only: bool) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint: Any = load_file(str(checkpoint_path), device="cpu")
    else:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "module"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint {checkpoint_path} did not contain a state dict")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        for prefix in ("module.", "model."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        state_dict[normalized_key] = value
    return state_dict


def _load_open_clip_checkpoint_compat(model: torch.nn.Module, checkpoint_path: Path, weights_only: bool) -> None:
    state_dict = _read_checkpoint_state_dict(checkpoint_path, weights_only)
    target_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logging.getLogger(__name__).warning(
        "Loaded %s with compatibility checkpoint path: kept=%d skipped=%d missing=%d unexpected=%d",
        checkpoint_path,
        len(filtered_state),
        len(skipped),
        len(missing),
        len(unexpected),
    )


class FrozenVLMWrapper:
    def __init__(self, name: str, model_type: str, device: str, dtype: str = "fp32") -> None:
        self.name = name
        self.model_type = model_type
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model: torch.nn.Module | None = None

    def score(self, image: Image.Image, text: str) -> float:
        return float(self.score_batch([image], [text])[0, 0].detach().cpu().item())

    def score_batch(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def score_pairs(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        scores = []
        for image, text in zip(images, texts):
            scores.append(self.score_batch([image], [text])[0, 0])
        return torch.stack(scores).detach().cpu()

    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    @property
    def supports_feature_cache(self) -> bool:
        return False

    @property
    def supports_retrieval_cache(self) -> bool:
        return self.supports_feature_cache

    @property
    def supports_text_conditioned_retrieval(self) -> bool:
        return False

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        assert self.model is not None
        image_tensor = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device, non_blocking=True)
        image_features = self.model.encode_image(image_tensor, normalize=True)
        return image_features.float().detach().cpu()

    @torch.no_grad()
    def encode_images_with_cues(self, images: list[Image.Image], cue_texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        image_tensor = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device, non_blocking=True)
        text_tokens = self.tokenizer(cue_texts).to(self.device)
        outputs = self.model(image_tensor, text_tokens, output_dict=True)
        image_features = outputs["image_features"].float()
        return F.normalize(image_features, dim=-1).detach().cpu()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens, normalize=True)
        return text_features.float().detach().cpu()

    @torch.no_grad()
    def score_pairs(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        image_tensor = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device, non_blocking=True)
        text_tokens = self.tokenizer(texts).to(self.device)
        outputs = self.model(image_tensor, text_tokens, output_dict=True)
        image_features = outputs["image_features"].float()
        text_features = outputs["text_features"].float()
        return (image_features * text_features).sum(dim=-1).detach().cpu()

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        pair_images: list[Image.Image] = []
        pair_texts: list[str] = []
        for image in images:
            for text in texts:
                pair_images.append(image)
                pair_texts.append(text)
        scores = self.score_pairs(pair_images, pair_texts)
        return scores.reshape(len(images), len(texts)).detach().cpu()


def _find_model_dir(model_root: Path, model_cfg: dict[str, Any]) -> Path | None:
    explicit = model_cfg.get("local_dir") or model_cfg.get("checkpoint_dir")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    name = model_cfg["name"]
    direct = model_root / name
    if direct.exists():
        return direct
    matches = sorted(p for p in model_root.iterdir() if p.is_dir() and name.lower() in p.name.lower()) if model_root.exists() else []
    return matches[0] if matches else None


def _find_checkpoint(model_dir: Path | None, model_cfg: dict[str, Any]) -> Path | None:
    explicit = model_cfg.get("checkpoint_path")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    if model_dir is None:
        return None
    checkpoint_file = model_cfg.get("checkpoint_file")
    if checkpoint_file and (model_dir / checkpoint_file).exists():
        return model_dir / checkpoint_file
    for pattern in ("*.pt", "*.safetensors", "*.bin", "*.pth", "*.ckpt"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _dummy_forward(wrapper: FrozenVLMWrapper) -> None:
    image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    score = wrapper.score(image, "a diagnostic image")
    if not isinstance(score, float):
        raise RuntimeError("dummy forward did not return a float")


def load_model(model_cfg: dict[str, Any], model_root: Path, device: str, dtype: str, logger: logging.Logger | None = None) -> ModelLoadResult:
    log = logger or logging.getLogger(__name__)
    name = str(model_cfg["name"])
    model_type = str(model_cfg.get("model_type") or model_cfg.get("type") or "")
    model_dir = _find_model_dir(model_root, model_cfg)
    checkpoint = _find_checkpoint(model_dir, model_cfg)
    try:
        if model_type == "open_clip":
            if checkpoint is None:
                raise FileNotFoundError(f"no local checkpoint found for {name} under {model_root}")
            wrapper = OpenClipWrapper(
                name=name,
                model_name=str(model_cfg["model_name"]),
                checkpoint_path=checkpoint,
                device=device,
                dtype=dtype,
                force_quick_gelu=bool(model_cfg.get("force_quick_gelu", model_cfg.get("pretrained") == "openai")),
            )
        elif model_type == "siglip":
            if model_dir is None:
                raise FileNotFoundError(f"no local HuggingFace directory found for {name} under {model_root}")
            wrapper = SigLIPWrapper(name=name, local_dir=model_dir, device=device, dtype=dtype)
        elif model_type == "blip_itm":
            if model_dir is None:
                raise FileNotFoundError(f"no local HuggingFace directory found for {name} under {model_root}")
            wrapper = BLIPITMWrapper(name=name, local_dir=model_dir, device=device, dtype=dtype)
        elif model_type in {"spvd", "spvd_clip"}:
            if checkpoint is None:
                raise FileNotFoundError(f"no SPVD checkpoint found for {name}")
            wrapper = SPVDWrapper(
                name=name,
                model_name=str(model_cfg.get("model_name") or "SPVD-ViT-B-16"),
                checkpoint_path=checkpoint,
                device=device,
                dtype=dtype,
                image_size=int(model_cfg.get("image_size", 224)),
            )
        else:
            raise ValueError(f"unsupported model_type for {name}: {model_type}")
        _dummy_forward(wrapper)
        result = ModelLoadResult(
            name=name,
            model_type=model_type,
            status="ok",
            wrapper=wrapper,
            checkpoint_path=checkpoint,
            local_dir=model_dir,
            parameter_count=wrapper.parameter_count(),
            device=str(wrapper.device),
            dtype=dtype,
            dummy_forward_success=True,
        )
        log.info("Loaded model %s: type=%s params=%s device=%s checkpoint=%s local_dir=%s", name, model_type, result.parameter_count, result.device, checkpoint, model_dir)
        return result
    except Exception as exc:
        log.exception("Failed to load model %s", name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ModelLoadResult(
            name=name,
            model_type=model_type,
            status="failed",
            checkpoint_path=checkpoint,
            local_dir=model_dir,
            device=device,
            dtype=dtype,
            error=repr(exc),
        )
