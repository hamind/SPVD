from __future__ import annotations

import gc
import logging
import sys
from contextlib import nullcontext
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


def _cache_dtype(dtype: str) -> torch.dtype:
    normalized = str(dtype).lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _autocast_dtype(dtype: str) -> torch.dtype | None:
    normalized = str(dtype).lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def _to_cache_tensor(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=_cache_dtype(dtype))
    return tensor.contiguous()


class FrozenVLMWrapper:
    def __init__(self, name: str, model_type: str, device: str, dtype: str = "fp32") -> None:
        self.name = name
        self.model_type = model_type
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model: torch.nn.Module | None = None
        self.preprocess: Any = None
        self.tokenizer: Any = None

    def _autocast_context(self):
        cast_dtype = _autocast_dtype(self.dtype)
        if self.device.type == "cuda" and cast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=cast_dtype)
        return nullcontext()

    def _image_tensor(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device, non_blocking=True)

    def _text_tokens(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts).to(self.device)

    def tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts)

    def score(self, image: Image.Image, text: str) -> float:
        return float(self.score_batch([image], [text])[0, 0].detach().cpu().item())

    @torch.inference_mode()
    def score_batch(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        pair_images: list[Image.Image] = []
        pair_texts: list[str] = []
        for image in images:
            for text in texts:
                pair_images.append(image)
                pair_texts.append(text)
        scores = self.score_pairs(pair_images, pair_texts)
        return scores.reshape(len(images), len(texts)).detach().cpu()

    @torch.inference_mode()
    def score_pairs(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)
        return self.score_encoded_pairs(image_features, text_features)

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

    @property
    def supports_text_conditioned_pair_cache(self) -> bool:
        return False

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        assert self.model is not None
        image_tensor = self._image_tensor(images)
        with self._autocast_context():
            image_features = self.model.encode_image(image_tensor, normalize=True)
        return image_features.float().detach().cpu()

    @torch.inference_mode()
    def encode_images_with_cues(self, images: list[Image.Image], cue_texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        image_tensor = self._image_tensor(images)
        text_tokens = self._text_tokens(cue_texts)
        with self._autocast_context():
            outputs = self.model(image_tensor, text_tokens, output_dict=True)
            image_features = outputs["image_features"].float()
        return F.normalize(image_features, dim=-1).detach().cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        text_tokens = self._text_tokens(texts)
        with self._autocast_context():
            text_features = self.model.encode_text(text_tokens, normalize=True)
        return text_features.float().detach().cpu()

    @torch.inference_mode()
    def score_encoded_pairs(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.to(self.device, non_blocking=True).float()
        text_features = text_features.to(self.device, non_blocking=True).float()
        scores = (F.normalize(image_features, dim=-1) * F.normalize(text_features, dim=-1)).sum(dim=-1)
        return scores.detach().cpu()

    @torch.inference_mode()
    def encode_image_tokens(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not expose image-token caching")

    @torch.inference_mode()
    def encode_image_tokens_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not expose image-token caching")

    @torch.inference_mode()
    def encode_text_cues(self, texts: list[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(f"{self.name} does not expose text-cue caching")

    @torch.inference_mode()
    def encode_text_cues_from_tokens(self, text_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError(f"{self.name} does not expose text-cue caching")

    @torch.inference_mode()
    def score_cached_conditioned_pairs(
        self,
        image_tokens: torch.Tensor,
        soft_cues: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not support text-conditioned pair scoring")

    @torch.inference_mode()
    def score_conditioned_retrieval_chunk(
        self,
        image_tokens: torch.Tensor,
        soft_cues: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        image_count = int(image_tokens.shape[0])
        text_count = int(soft_cues.shape[0])
        if image_count == 0 or text_count == 0:
            return torch.empty((image_count, text_count), dtype=torch.float32)

        image_pairs = (
            image_tokens.unsqueeze(1)
            .expand(image_count, text_count, *image_tokens.shape[1:])
            .reshape(image_count * text_count, *image_tokens.shape[1:])
            .contiguous()
        )
        soft_cue_pairs = (
            soft_cues.unsqueeze(0)
            .expand(image_count, text_count, *soft_cues.shape[1:])
            .reshape(image_count * text_count, *soft_cues.shape[1:])
            .contiguous()
        )
        text_feature_pairs = (
            text_features.unsqueeze(0)
            .expand(image_count, text_count, *text_features.shape[1:])
            .reshape(image_count * text_count, *text_features.shape[1:])
            .contiguous()
        )
        scores = self.score_cached_conditioned_pairs(image_pairs, soft_cue_pairs, text_feature_pairs)
        return scores.reshape(image_count, text_count).detach().cpu().float()

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenClipWrapper(FrozenVLMWrapper):
    def __init__(
        self,
        name: str,
        model_name: str,
        checkpoint_path: Path,
        device: str,
        dtype: str = "fp32",
        force_quick_gelu: bool = False,
    ) -> None:
        super().__init__(name=name, model_type="open_clip", device=device, dtype=dtype)
        import open_clip

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained="",
            device=self.device,
            force_quick_gelu=force_quick_gelu,
            output_dict=True,
        )
        _load_open_clip_checkpoint_compat(model, Path(checkpoint_path), weights_only=True)
        self.model = model.eval()
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def supports_feature_cache(self) -> bool:
        return True


class HFCLIPWrapper(FrozenVLMWrapper):
    def __init__(self, name: str, local_dir: Path, device: str, dtype: str = "fp32") -> None:
        super().__init__(name=name, model_type="hf_clip", device=device, dtype=dtype)
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(str(local_dir), local_files_only=True)
        self.model = CLIPModel.from_pretrained(str(local_dir), local_files_only=True).to(self.device).eval()

    @property
    def supports_feature_cache(self) -> bool:
        return True

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        assert self.model is not None
        inputs = self.processor(images=[img.convert("RGB") for img in images], return_tensors="pt")
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        with self._autocast_context():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            features = self.model.visual_projection(vision_outputs.pooler_output)
        return F.normalize(features.float(), dim=-1).detach().cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        with self._autocast_context():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            features = self.model.text_projection(text_outputs.pooler_output)
        return F.normalize(features.float(), dim=-1).detach().cpu()


class SigLIPWrapper(FrozenVLMWrapper):
    def __init__(self, name: str, local_dir: Path, device: str, dtype: str = "fp32") -> None:
        super().__init__(name=name, model_type="siglip", device=device, dtype=dtype)
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(local_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(local_dir), local_files_only=True).to(self.device).eval()

    @property
    def supports_feature_cache(self) -> bool:
        return True

    def _feature_tensor(self, outputs: Any, preferred_attr: str) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        for attr in (preferred_attr, "pooler_output", "last_hidden_state"):
            value = getattr(outputs, attr, None)
            if torch.is_tensor(value):
                if attr == "last_hidden_state" and value.ndim >= 3:
                    return value[:, 0]
                return value
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"{self.name} returned no tensor feature in {type(outputs).__name__}")

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        assert self.model is not None
        inputs = self.processor(images=[img.convert("RGB") for img in images], return_tensors="pt")
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        with self._autocast_context():
            if hasattr(self.model, "get_image_features"):
                features = self._feature_tensor(self.model.get_image_features(**inputs), "image_embeds")
            else:
                outputs = self.model(**inputs)
                features = self._feature_tensor(outputs, "image_embeds")
        return F.normalize(features.float(), dim=-1).detach().cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        with self._autocast_context():
            if hasattr(self.model, "get_text_features"):
                features = self._feature_tensor(self.model.get_text_features(**inputs), "text_embeds")
            else:
                outputs = self.model(**inputs)
                features = self._feature_tensor(outputs, "text_embeds")
        return F.normalize(features.float(), dim=-1).detach().cpu()


class BLIPITMWrapper(FrozenVLMWrapper):
    def __init__(self, name: str, local_dir: Path, device: str, dtype: str = "fp32") -> None:
        super().__init__(name=name, model_type="blip_itm", device=device, dtype=dtype)
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(str(local_dir), local_files_only=True)
        self.model = BlipForImageTextRetrieval.from_pretrained(str(local_dir), local_files_only=True).to(self.device).eval()

    @torch.inference_mode()
    def score_pairs(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        inputs = self.processor(
            images=[img.convert("RGB") for img in images],
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        with self._autocast_context():
            outputs = self.model(**inputs, use_itm_head=True)
        logits = getattr(outputs, "itm_score", None)
        if logits is None:
            logits = getattr(outputs, "logits_per_image", None)
        if logits is None:
            logits = outputs[0]
        if logits.ndim == 2 and logits.shape[-1] == 2:
            scores = logits[:, 1]
        else:
            scores = logits.reshape(len(images), -1)[:, 0]
        return scores.float().detach().cpu()


class SPVDWrapper(FrozenVLMWrapper):
    def __init__(
        self,
        name: str,
        model_name: str,
        checkpoint_path: Path,
        device: str,
        dtype: str = "fp32",
        image_size: int = 224,
    ) -> None:
        super().__init__(name=name, model_type="spvd", device=device, dtype=dtype)
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from checkpoint import load_checkpoint as load_spvd_checkpoint
        from factory import create_model_and_transforms, create_tokenizer

        precision = str(dtype).lower()
        if precision not in {"fp32", "fp16", "bf16", "pure_fp16", "pure_bf16"}:
            precision = "fp32"
        model, _, preprocess_val = create_model_and_transforms(
            model_name,
            pretrained=None,
            precision=precision,
            device=self.device,
            force_image_size=image_size,
            output_dict=True,
            spvd_cfg={
                "enable_soft_cue_decomp": True,
                "use_finegrained_text_cue": True,
                "text_cue_type": "soft_cue",
                "num_soft_cues": 4,
                "return_patch_tokens": True,
            },
        )
        load_spvd_checkpoint(str(checkpoint_path), model, map_location=self.device)
        self.model = model.eval()
        self.preprocess = preprocess_val
        self.tokenizer = create_tokenizer(model_name)

    @property
    def supports_text_conditioned_pair_cache(self) -> bool:
        return True

    @torch.inference_mode()
    def encode_image_tokens(self, images: list[Image.Image]) -> torch.Tensor:
        image_tensor = self._image_tensor(images)
        return self.encode_image_tokens_from_tensor(image_tensor)

    @torch.inference_mode()
    def encode_image_tokens_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        with self._autocast_context():
            outputs = self.model.encode_image(image_tensor, normalize=False, return_tokens=True)
        image_tokens = outputs["image_tokens"]
        return _to_cache_tensor(image_tokens, self.dtype)

    @torch.inference_mode()
    def encode_text_cues(self, texts: list[str]) -> dict[str, torch.Tensor]:
        text_tokens = self.tokenize_texts(texts)
        return self.encode_text_cues_from_tokens(text_tokens)

    @torch.inference_mode()
    def encode_text_cues_from_tokens(self, text_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.model is not None
        text_tokens = text_tokens.to(self.device, non_blocking=True)
        with self._autocast_context():
            outputs = self.model.encode_text(text_tokens, normalize=True, return_tokens=True)
        result = {
            "text_features": _to_cache_tensor(outputs["text_global"], self.dtype),
            "soft_cues": _to_cache_tensor(outputs["cue"], self.dtype),
        }
        attention_mask = outputs.get("text_attention_mask")
        if attention_mask is not None:
            result["text_attention_mask"] = _to_cache_tensor(attention_mask, self.dtype)
        return result

    @torch.inference_mode()
    def score_cached_conditioned_pairs(
        self,
        image_tokens: torch.Tensor,
        soft_cues: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model is not None
        image_tokens = image_tokens.to(self.device, non_blocking=True)
        soft_cues = soft_cues.to(self.device, non_blocking=True)
        text_features = text_features.to(self.device, non_blocking=True)
        with self._autocast_context():
            decomp_outputs = self.model.soft_cue_decomposition(image_tokens, soft_cues)
            shared_features = decomp_outputs["shared_visual_features"]
        scores = (shared_features.float() * text_features.float()).sum(dim=-1)
        return scores

    @torch.inference_mode()
    def score_pairs(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        image_tokens = self.encode_image_tokens(images)
        text_outputs = self.encode_text_cues(texts)
        return self.score_cached_conditioned_pairs(
            image_tokens=image_tokens,
            soft_cues=text_outputs["soft_cues"],
            text_features=text_outputs["text_features"],
        )


def _find_model_dir(model_root: Path, model_cfg: dict[str, Any]) -> Path | None:
    explicit = model_cfg.get("local_dir") or model_cfg.get("checkpoint_dir")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    if not model_root.exists():
        return None

    names: list[str] = []
    for key in ("name", "hf_id", "model_name", "pretrained"):
        value = model_cfg.get(key)
        if value:
            names.append(str(value))
    candidates: list[Path] = []
    for name in names:
        slashless = name.replace("/", "--")
        candidates.extend(
            [
                model_root / name,
                model_root / slashless,
                model_root / f"models--{slashless}",
                model_root / Path(name).name,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    fragments = {name.lower() for name in names}
    fragments.update(name.replace("/", "--").lower() for name in names)
    fragments.update(Path(name).name.lower() for name in names)
    matches = sorted(
        p
        for p in model_root.iterdir()
        if p.is_dir() and any(fragment and fragment in p.name.lower() for fragment in fragments)
    )
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
        elif model_type in {"hf_clip", "clip", "transformers_clip"}:
            if model_dir is None:
                raise FileNotFoundError(f"no local HuggingFace CLIP directory found for {name} under {model_root}")
            wrapper = HFCLIPWrapper(name=name, local_dir=model_dir, device=device, dtype=dtype)
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
        log.info(
            "Loaded model %s: type=%s params=%s device=%s checkpoint=%s local_dir=%s",
            name,
            model_type,
            result.parameter_count,
            result.device,
            checkpoint,
            model_dir,
        )
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
