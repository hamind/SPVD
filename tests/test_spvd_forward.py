from __future__ import annotations

import torch
from open_clip.model import CLIP

from factory import create_model_and_transforms, create_tokenizer, get_model_config, list_models
from model import SPVDModel


def test_spvd_model_registry_contains_vitb16_config() -> None:
    assert "SPVD-ViT-B-16" in list_models()
    cfg = get_model_config("SPVD-ViT-B-16")
    assert cfg is not None
    assert cfg["embed_dim"] == 512
    assert "spvd_cfg" in cfg


def test_spvd_default_forward_is_clip_style() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
    )
    assert isinstance(model, SPVDModel)
    assert isinstance(model, torch.nn.Module)
    assert not isinstance(model, CLIP)

    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer(["a small image", "another tiny image"])
    outputs = model(images, texts, output_dict=True)

    assert outputs["image_features"].shape == (2, model.embed_dim)
    assert outputs["text_features"].shape == (2, model.embed_dim)
    assert outputs["logit_scale"].ndim == 0
    assert "relevance_scores" not in outputs
    assert "shared_mask" not in outputs


def test_spvd_soft_cue_forward_shapes() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
        config_dict={
            "model": {
                "enable_soft_cue_decomp": True,
                "num_soft_cues": 4,
                "soft_cue_num_heads": 4,
                "soft_cue_num_layers": 1,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer(["a small image", "another tiny image"])
    outputs = model(images, texts, output_dict=True)

    assert outputs["image_features"].shape == (2, model.embed_dim)
    assert outputs["text_features"].shape == (2, model.embed_dim)
    assert outputs["cue"].shape == (2, 4, model.embed_dim)
    assert outputs["soft_cues"].shape == (2, 4, model.embed_dim)
    assert torch.allclose(outputs["cue"], outputs["soft_cues"])
    assert outputs["relevance_scores"].shape == (2, 4, 4)
    assert outputs["shared_routing"].shape == (2, 4, 4)
    assert outputs["residual_routing"].shape == (2, 4, 4)
    assert outputs["cue_visual_features"].shape == (2, 4, model.embed_dim)
    assert outputs["cue_residual_features"].shape == (2, 4, model.embed_dim)
    assert outputs["cue_weights"].shape == (2, 4)
    assert not model.visual.proj.requires_grad
    assert torch.allclose(outputs["shared_routing"] + outputs["residual_routing"], torch.ones_like(outputs["shared_routing"]), atol=1.0e-5)


def test_spvd_soft_cue_forward_accepts_multi_caption_text() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
        config_dict={
            "model": {
                "enable_soft_cue_decomp": True,
                "num_soft_cues": 4,
                "soft_cue_num_heads": 4,
                "soft_cue_num_layers": 1,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer([
        "a small image",
        "a tiny picture",
        "a compact scene",
        "another tiny image",
        "another small picture",
        "another compact scene",
    ]).reshape(2, 3, -1)

    outputs = model(images, texts, output_dict=True)

    assert outputs["image_features"].shape == (2, model.embed_dim)
    assert outputs["text_features"].shape == (2, model.embed_dim)
    assert outputs["text_tokens"].shape == (2, 3, texts.shape[-1], model.text_dim)
    assert outputs["caption_text_features"].shape == (2, 3, model.embed_dim)
    assert outputs["cue"].shape == (2, 3, 4, model.embed_dim)
    assert outputs["soft_cues"].shape == (2, 3, 4, model.embed_dim)
    assert torch.allclose(outputs["cue"], outputs["soft_cues"])
    assert outputs["image_attention"].shape == (2, 3, 4, 4)
    assert outputs["relevance_scores"].shape == (2, 3, 4, 4)
    assert outputs["cue_visual_features"].shape == (2, 3, 4, model.embed_dim)
    assert outputs["caption_shared_visual_features"].shape == (2, 3, model.embed_dim)


def test_spvd_no_soft_cue_does_not_register_unused_extractor() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
        config_dict={
            "model": {
                "enable_soft_cue_decomp": True,
                "use_finegrained_text_cue": False,
                "text_cue_type": "pooled",
                "num_soft_cues": 4,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer([
        "a small image",
        "a tiny picture",
        "a compact scene",
        "another tiny image",
        "another small picture",
        "another compact scene",
    ]).reshape(2, 3, -1)

    outputs = model(images, texts, output_dict=True)

    assert not hasattr(model, "soft_cue_extractor")
    assert outputs["cue"].shape == (2, 3, 1, model.embed_dim)
    assert outputs["soft_cues"].shape == (2, 3, 1, model.embed_dim)
    assert torch.allclose(outputs["cue"], outputs["soft_cues"])
    assert outputs["caption_shared_visual_features"].shape == (2, 3, model.embed_dim)


def test_encode_text_returns_text_global_and_cue() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
        config_dict={
            "model": {
                "enable_soft_cue_decomp": True,
                "num_soft_cues": 4,
                "soft_cue_num_heads": 4,
                "soft_cue_num_layers": 1,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    texts = tokenizer([
        "a small image",
        "a tiny picture",
        "a compact scene",
        "another tiny image",
        "another small picture",
        "another compact scene",
    ]).reshape(2, 3, -1)

    outputs = model.encode_text(texts, normalize=True, return_tokens=True)

    assert outputs["text_global"].shape == (2, 3, model.embed_dim)
    assert outputs["cue"].shape == (2, 3, 4, model.embed_dim)
    assert torch.allclose(outputs["cue"], outputs["soft_cues"])
    assert outputs["text_tokens"].shape == (2, 3, texts.shape[-1], model.text_dim)
