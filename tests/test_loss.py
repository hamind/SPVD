from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F
from open_clip.loss import ClipLoss
from open_clip.loss import SigLipLoss as OpenCLIPSigLipLoss

from factory import create_loss, create_model_and_transforms
from losses import BranchBCELoss, GateMapStats, ResidualVarianceLoss, SPVDLoss


def test_clip_loss_returns_scalar() -> None:
    args = Namespace(siglip=False, local_loss=True, gather_with_grad=True, rank=0, world_size=1)
    loss_fn = create_loss(args)
    assert isinstance(loss_fn, ClipLoss)
    image = F.normalize(torch.randn(4, 8), dim=-1)
    text = F.normalize(torch.randn(4, 8), dim=-1)
    loss = loss_fn(image, text, torch.tensor(10.0))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_siglip_loss_returns_scalar() -> None:
    args = Namespace(siglip=True, loss_dist_impl="bidir", rank=0, world_size=1)
    loss_fn = create_loss(args)
    assert isinstance(loss_fn, OpenCLIPSigLipLoss)
    image = F.normalize(torch.randn(4, 8), dim=-1)
    text = F.normalize(torch.randn(4, 8), dim=-1)
    loss = loss_fn(image, text, torch.tensor(10.0), torch.tensor(-10.0))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_openclip_siglip_loss_matches_logsigmoid_formula() -> None:
    image = F.normalize(torch.randn(3, 8), dim=-1)
    text = F.normalize(torch.randn(3, 8), dim=-1)
    logit_scale = torch.tensor(10.0)
    logit_bias = torch.tensor(-1.0)
    loss_fn = OpenCLIPSigLipLoss(rank=0, world_size=1)

    loss = loss_fn(image, text, logit_scale, logit_bias)
    logits = loss_fn.get_logits(image, text, logit_scale, logit_bias)
    labels = loss_fn.get_ground_truth(image.device, image.dtype, image.shape[0])
    expected = -F.logsigmoid(labels * logits).sum() / image.shape[0]

    assert torch.allclose(loss, expected)


def test_spvd_sigmoid_alignment_initializes_siglip_bias() -> None:
    model, _, _ = create_model_and_transforms(
        "SPVD-ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
        config_dict={"loss": {"align_loss": "sigmoid"}},
    )

    assert model.logit_bias is not None
    assert torch.allclose(model.logit_bias.detach(), torch.tensor(-10.0))
    assert torch.allclose(model.logit_scale.detach(), torch.tensor(1 / 0.07).log())


def test_spvd_loss_without_decomposition_uses_alignment_only() -> None:
    args = Namespace(
        loss_name="spvd",
        local_loss=True,
        gather_with_grad=True,
        rank=0,
        world_size=1,
        residual_variance_gamma=1.0,
    )
    loss_fn = create_loss(args)
    assert isinstance(loss_fn, SPVDLoss)
    outputs = {
        "image_features": torch.randn(4, 8, requires_grad=True),
        "text_features": torch.randn(4, 8, requires_grad=True),
        "logit_scale": torch.tensor(10.0),
    }

    loss, loss_dict = loss_fn(outputs)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_align"])
    assert loss_dict["loss_branch"].item() == 0.0
    assert loss_dict["loss_residual_variance"].item() == 0.0
    loss.backward()
    assert outputs["image_features"].grad is not None
    assert outputs["text_features"].grad is not None


def test_spvd_loss_prefers_flattened_caption_sigmoid_alignment() -> None:
    image_features = torch.randn(2, 8, requires_grad=True)
    text_features = torch.randn(2, 8, requires_grad=True)
    caption_visual_features = torch.randn(2, 3, 8, requires_grad=True)
    caption_text_features = torch.randn(2, 3, 8, requires_grad=True)
    logit_scale = torch.tensor(10.0)
    outputs = {
        "image_features": image_features,
        "text_features": text_features,
        "caption_shared_visual_features": caption_visual_features,
        "caption_text_features": caption_text_features,
        "logit_scale": logit_scale,
    }
    loss_fn = SPVDLoss(rank=0, world_size=1)

    loss, loss_dict = loss_fn(outputs)
    expected_global = OpenCLIPSigLipLoss(rank=0, world_size=1)(
        image_features,
        text_features,
        logit_scale,
        None,
    )
    expected_caption = OpenCLIPSigLipLoss(rank=0, world_size=1)(
        caption_visual_features.reshape(-1, 8),
        caption_text_features.reshape(-1, 8),
        logit_scale,
        None,
    )
    expected_loss = (expected_global + expected_caption) / 2

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_align"])
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(loss_dict["loss_align_global"], expected_global.detach())
    assert torch.allclose(loss_dict["loss_align_caption"], expected_caption.detach())
    loss.backward()
    assert caption_visual_features.grad is not None
    assert caption_text_features.grad is not None
    assert image_features.grad is not None
    assert text_features.grad is not None


def test_spvd_loss_sigmoid_branch_terms_are_finite() -> None:
    image_features = torch.randn(5, 8, requires_grad=True)
    text_features = torch.randn(5, 8, requires_grad=True)
    residual_features = torch.randn(5, 8, requires_grad=True)
    gate_logits = torch.randn(5, 4, 6)
    sigmoid_map = torch.sigmoid(gate_logits)
    outputs = {
        "image_features": image_features,
        "text_features": text_features,
        "residual_visual_features": residual_features,
        "gate_logits": gate_logits,
        "sigmoid_map": sigmoid_map,
        "residual_map": 1.0 - sigmoid_map,
        "logit_scale": torch.tensor(10.0),
    }
    loss_fn = SPVDLoss(
        rank=0,
        world_size=1,
        branch_bce_weight=0.05,
        residual_variance_weight=0.05,
    )
    assert isinstance(loss_fn.branch_bce, BranchBCELoss)
    assert isinstance(loss_fn.residual_variance, ResidualVarianceLoss)
    assert isinstance(loss_fn.gate_stats, GateMapStats)

    loss, loss_dict = loss_fn(outputs)

    assert torch.isfinite(loss)
    for key in (
        "loss_align",
        "loss_branch",
        "loss_branch_s_text",
        "loss_branch_r_text",
        "branch_sim_s_text",
        "branch_sim_r_text",
        "branch_gap_s_minus_r",
        "loss_residual_variance",
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
    ):
        assert torch.isfinite(loss_dict[key])
    loss.backward()
    assert image_features.grad is not None
    assert text_features.grad is not None
    assert residual_features.grad is not None
