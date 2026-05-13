from __future__ import annotations

import math
import os
import pytest
import torch
from open_clip.model import CLIP

from factory import create_model_and_transforms, create_tokenizer, get_model_config, list_models
from losses import SPVDLoss
from model import SPVDModel, SoftCueSigmoidDecomposition
from training import clamp_logit_scale_, clip_gradients_


def _compile_for_test(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")
    backend = os.environ.get("SPVD_TEST_COMPILE_BACKEND", "eager")
    kwargs = {"backend": backend, "fullgraph": False, "dynamic": False}
    if backend == "inductor":
        kwargs["mode"] = "reduce-overhead"
    return torch.compile(model, **kwargs)


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


def test_spvd_training_forward_rejects_single_caption_text() -> None:
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
    with pytest.raises(ValueError, match="SPVD training forward expects multi-sub-caption text"):
        model(images, texts, output_dict=True)
    assert not model.visual.proj.requires_grad


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
    texts = tokenizer([
        "a small image",
        "a tiny picture",
        "a compact scene",
        "another tiny image",
        "another small picture",
        "another compact scene",
    ]).reshape(2, 3, -1)
    outputs = model(images, texts, output_dict=True)

    for key in (
        "caption_semantic_visual_features",
        "caption_shared_visual_features",
        "caption_residual_visual_features",
        "residual_visual_features",
        "caption_text_features",
        "text_features",
        "sigmoid_map",
        "residual_map",
        "gate_logits",
    ):
        assert key in outputs
    assert "image_features" not in outputs
    assert outputs["caption_semantic_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_text_features"].shape == (2, 3, model.embed_dim)
    assert outputs["text_features"].shape == (2, 3, model.embed_dim)
    assert outputs["sigmoid_map"].shape == (2, 3, 4, 4)
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


def test_spvd_wrapper_cached_conditioned_score_uses_semantic_features() -> None:
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
                "use_finegrained_text_cue": True,
                "text_cue_type": "soft_cue",
                "num_soft_cues": 4,
                "soft_cue_num_heads": 4,
                "soft_cue_num_layers": 1,
            }
        },
    )
    tokenizer = create_tokenizer("SPVD-ViT-B-16")
    images = torch.randn(2, 3, 32, 32)
    texts = tokenizer(["a small image", "another tiny image"])

    pairwise = model.score_pairwise(images, texts, return_dict=True)
    assert isinstance(pairwise, dict)
    scores = pairwise["score"]
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert pairwise["semantic_features"].shape == (2, model.embed_dim)
    assert pairwise["text_features"].shape == (2, model.embed_dim)
    assert pairwise["sigmoid_map"].ndim == 3


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

    assert "image_features" not in outputs
    assert outputs["caption_semantic_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["text_features"].shape == (2, 3, model.embed_dim)
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
    assert outputs["caption_semantic_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["caption_residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["shared_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["residual_visual_features"].shape == (2, 3, model.embed_dim)
    assert outputs["sigmoid_map"].shape == (2, 3, 1, 4)


def test_encode_text_multi_returns_text_global_and_cue() -> None:
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

    outputs = model.encode_text_multi(texts, normalize=True, return_tokens=True)

    assert outputs["text_global"].shape == (2, 3, model.embed_dim)
    assert outputs["cue"].shape == (2, 3, 4, model.embed_dim)
    assert torch.allclose(outputs["cue"], outputs["soft_cues"])
    assert outputs["text_tokens"].shape == (2, 3, texts.shape[-1], model.text_dim)


def test_soft_cue_sigmoid_decomposition_requires_multi_caption_cues() -> None:
    decomp = SoftCueSigmoidDecomposition(visual_dim=16, embed_dim=8)
    visual_tokens = torch.randn(2, 5, 16)
    soft_cues = torch.randn(2, 3, 4, 8)

    outputs = decomp(visual_tokens, soft_cues)

    assert outputs["semantic_features"].shape == (2, 3, 8)
    assert outputs["residual_features"].shape == (2, 3, 8)
    assert outputs["sigmoid_map"].shape == (2, 3, 4, 5)
    assert outputs["residual_map"].shape == (2, 3, 4, 5)
    assert outputs["gate_logits"].shape == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="soft_cues must have shape"):
        decomp(visual_tokens, torch.randn(2, 4, 8))



def test_logit_scale_clamp_uses_exp_scale_bounds() -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(100.0)))

    model = DummyModel()
    compiled_model = _compile_for_test(model)
    assert clamp_logit_scale_(compiled_model, min_scale=1.0, max_scale=30.0)
    assert model.logit_scale.exp().item() <= 30.0 + 1.0e-5


def test_grad_clip_helper_is_optional() -> None:
    model = torch.nn.Linear(4, 2)
    loss = model(torch.ones(3, 4)).sum()
    loss.backward()

    assert not clip_gradients_(model, None)
    assert clip_gradients_(model, 1.0)



def test_spvd_torch_compile_soft_cue_forward_and_masked_loss() -> None:
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
                "use_finegrained_text_cue": True,
                "text_cue_type": "soft_cue",
                "num_soft_cues": 4,
                "soft_cue_num_heads": 4,
                "soft_cue_num_layers": 1,
            }
        },
    )
    compiled_model = _compile_for_test(model)
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

    outputs = compiled_model(images, texts, output_dict=True)
    loss_fn = SPVDLoss(
        rank=0,
        world_size=1,
        global_align_weight=0.0,
        caption_align_weight=1.0,
        caption_loss_impl="masked_sigmoid",
        branch_bce_weight=0.05,
        residual_variance_weight=0.02,
        branch_bce_warmup_steps=10,
        residual_variance_warmup_steps=10,
    )
    loss_fn.set_global_step(10)
    loss, loss_dict = loss_fn(outputs)

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_align_caption"])
    assert loss_dict["loss_align_global"].item() == 0.0
    assert loss_dict["caption_masked_same_image_pairs"].item() > 0
    loss.backward()
    raw_model = compiled_model._orig_mod
    assert raw_model.soft_cue_decomposition.query_proj.weight.grad is not None
