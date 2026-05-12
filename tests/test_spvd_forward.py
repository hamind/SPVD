from __future__ import annotations

import torch
from open_clip.model import CLIP

from factory import create_model_and_transforms, create_tokenizer, get_model_config, list_models
from losses import SPVDLoss
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
    assert outputs["shared_visual_features"].shape == (2, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, model.embed_dim)
    assert outputs["text_features"].shape == (2, model.embed_dim)
    assert outputs["caption_text_features"] is None
    assert outputs["sigmoid_map"].shape == (2, 4, 4)
    assert outputs["residual_map"].shape == (2, 4, 4)
    assert outputs["gate_logits"].shape == (2, 4, 4)
    assert "relevance_scores" not in outputs
    assert "routing_logits" not in outputs
    assert "shared_routing" not in outputs
    assert "residual_routing" not in outputs
    assert "image_attention" not in outputs
    assert "cue" not in outputs
    assert "soft_cues" not in outputs
    assert "cue_visual_features" not in outputs
    assert "cue_residual_features" not in outputs
    assert "global_image_features" not in outputs
    assert not model.visual.proj.requires_grad
    assert torch.allclose(outputs["sigmoid_map"] + outputs["residual_map"], torch.ones_like(outputs["sigmoid_map"]), atol=1.0e-5)


def test_spvd_sigmoid_branch_bce_one_batch_backward() -> None:
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
                "gate_temperature": 1.0,
                "gate_bias_init": 0.0,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer(["a small image", "another tiny image"])
    outputs = model(images, texts, output_dict=True)

    for key in ("shared_visual_features", "residual_visual_features", "sigmoid_map", "residual_map", "gate_logits"):
        assert key in outputs
    assert outputs["shared_visual_features"].shape == (2, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, model.embed_dim)
    assert outputs["sigmoid_map"].shape == (2, 4, 4)
    assert torch.isfinite(outputs["sigmoid_map"]).all()
    gate = outputs["sigmoid_map"].detach()
    assert 0.01 < float(gate.mean()) < 0.99
    assert float(gate.std(unbiased=False)) > 0

    loss_fn = SPVDLoss(
        rank=0,
        world_size=1,
        branch_bce_weight=0.05,
        branch_logit_scale=5.0,
        residual_negative_weight=0.25,
        detach_text_for_residual=True,
        residual_variance_weight=0.05,
        residual_variance_gamma=1.0,
    )
    loss, loss_dict = loss_fn(outputs)
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_branch"])
    assert torch.isfinite(loss_dict["loss_residual_variance"])
    assert torch.isfinite(loss_dict["branch_gap_s_minus_r"])
    loss.backward()

    decomp = model.soft_cue_decomposition
    assert decomp.query_proj.weight.grad is not None
    assert decomp.key_proj.weight.grad is not None
    assert decomp.semantic_value_proj.weight.grad is not None
    assert decomp.residual_value_proj.weight.grad is not None
    assert decomp.gate_bias.grad is not None


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

    assert outputs["image_features"].shape == (2, 3, model.embed_dim)
    assert outputs["shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["text_features"].shape == (2, model.embed_dim)
    assert outputs["caption_text_features"].shape == (2, 3, model.embed_dim)
    assert outputs["sigmoid_map"].shape == (2, 3, 4, 4)
    assert outputs["residual_map"].shape == (2, 3, 4, 4)
    assert outputs["gate_logits"].shape == (2, 3, 4, 4)
    assert "text_tokens" not in outputs
    assert "cue" not in outputs
    assert "soft_cues" not in outputs
    assert "image_attention" not in outputs
    assert "relevance_scores" not in outputs
    assert "cue_visual_features" not in outputs
    assert "caption_shared_visual_features" not in outputs
    assert "caption_residual_visual_features" not in outputs
    assert torch.allclose(outputs["sigmoid_map"] + outputs["residual_map"], torch.ones_like(outputs["sigmoid_map"]), atol=1.0e-5)


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
    assert "cue" not in outputs
    assert "soft_cues" not in outputs
    assert outputs["shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["sigmoid_map"].shape == (2, 3, 1, 4)


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
