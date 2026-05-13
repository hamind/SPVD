from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F
from open_clip.loss import ClipLoss
from open_clip.loss import SigLipLoss as OpenCLIPSigLipLoss

from factory import create_loss, create_model_and_transforms
from losses import BranchBCELoss, GateMapStats, MaskedCaptionSigLipLoss, ResidualVarianceLoss, SPVDLoss


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
        global_align_weight=1.0,
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


def test_spvd_loss_uses_explicit_caption_sigmoid_alignment() -> None:
    caption_visual_features = torch.randn(2, 3, 8, requires_grad=True)
    caption_text_features = torch.randn(2, 3, 8, requires_grad=True)
    logit_scale = torch.tensor(10.0)
    outputs = {
        "caption_semantic_visual_features": caption_visual_features,
        "caption_text_features": caption_text_features,
        "logit_scale": logit_scale,
    }
    loss_fn = SPVDLoss(rank=0, world_size=1, global_align_weight=0.0, caption_loss_impl="openclip_siglip")

    loss, loss_dict = loss_fn(outputs)
    expected_caption = OpenCLIPSigLipLoss(rank=0, world_size=1)(
        F.normalize(caption_visual_features.reshape(-1, 8).float(), dim=-1),
        F.normalize(caption_text_features.reshape(-1, 8).float(), dim=-1),
        logit_scale,
        None,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_align"])
    assert torch.allclose(loss, expected_caption)
    assert loss_dict["loss_align_global"].item() == 0.0
    assert torch.allclose(loss_dict["loss_align_caption"], expected_caption.detach())
    loss.backward()
    assert caption_visual_features.grad is not None
    assert caption_text_features.grad is not None


def test_spvd_loss_rejects_flat_caption_alignment_features() -> None:
    outputs = {
        "caption_semantic_visual_features": torch.randn(2, 8, requires_grad=True),
        "caption_text_features": torch.randn(2, 8, requires_grad=True),
        "logit_scale": torch.tensor(10.0),
    }
    loss_fn = SPVDLoss(rank=0, world_size=1)

    with pytest.raises(ValueError, match=r"\[B, K, D\]"):
        loss_fn(outputs)


def test_spvd_loss_sigmoid_branch_terms_are_finite() -> None:
    image_features = torch.randn(2, 3, 8, requires_grad=True)
    text_features = torch.randn(2, 3, 8, requires_grad=True)
    residual_features = torch.randn(2, 3, 8, requires_grad=True)
    gate_logits = torch.randn(2, 3, 4, 6)
    sigmoid_map = torch.sigmoid(gate_logits)
    outputs = {
        "caption_semantic_visual_features": image_features,
        "caption_text_features": text_features,
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



def test_masked_caption_siglip_masks_same_image_subcaptions() -> None:
    bsz, num_captions, dim = 2, 3, 8
    image = torch.randn(bsz, num_captions, dim, requires_grad=True)
    text = torch.randn(bsz, num_captions, dim, requires_grad=True)
    loss_fn = MaskedCaptionSigLipLoss()

    loss, stats = loss_fn(image, text, torch.tensor(10.0), logit_bias=torch.tensor(-1.0))

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert int(stats["caption_num_pairs"].item()) == (bsz * num_captions) * (bsz * num_captions)
    assert int(stats["caption_masked_same_image_pairs"].item()) == bsz * num_captions * (num_captions - 1)
    assert int(stats["caption_num_valid_pairs"].item()) == int(stats["caption_num_pairs"].item()) - int(stats["caption_masked_same_image_pairs"].item())
    assert 0.0 < float(stats["caption_valid_negative_fraction"].item()) < 1.0
    loss.backward()
    assert image.grad is not None
    assert text.grad is not None


def test_masked_caption_siglip_rejects_flat_caption_features() -> None:
    loss_fn = MaskedCaptionSigLipLoss()
    image = torch.randn(2, 8)
    text = torch.randn(2, 8)

    with pytest.raises(ValueError, match=r"\[B, K, D\]"):
        loss_fn(image, text, torch.tensor(10.0))


def test_masked_caption_signed_uses_iou_positive_and_low_overlap_negative() -> None:
    bsz, num_captions, num_soft_cues, num_tokens, dim = 1, 3, 1, 4, 8
    image = torch.randn(bsz, num_captions, dim, requires_grad=True)
    text = torch.randn(bsz, num_captions, dim, requires_grad=True)
    sigmoid_map = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]]]
    ).reshape(bsz, num_captions, num_soft_cues, num_tokens)
    loss_fn = MaskedCaptionSigLipLoss(
        caption_same_image_mode="signed",
        same_image_iou_threshold=0.3,
    )

    loss, stats = loss_fn(image, text, torch.tensor(10.0), sigmoid_map=sigmoid_map)
    labels, weights, _ = loss_fn.build_labels_and_weights(
        torch.zeros(bsz * num_captions, bsz * num_captions),
        bsz,
        num_captions,
        sigmoid_map,
    )

    assert torch.isfinite(loss)
    assert labels[0, 1].item() > 0.3
    assert labels[0, 2].item() == -1.0
    assert weights[0, 1].item() == labels[0, 1].item()
    assert weights[0, 2].item() == 1.0
    assert stats["caption_same_image_mode_code"].item() == 2.0
    loss.backward()
    assert image.grad is not None
    assert text.grad is not None


def test_masked_caption_positive_threshold_gap_is_ignored() -> None:
    image = torch.randn(1, 2, 8, requires_grad=True)
    text = torch.randn(1, 2, 8, requires_grad=True)
    sigmoid_map = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.2, 0.8, 0.0, 0.0]]]])
    loss_fn = MaskedCaptionSigLipLoss(
        caption_same_image_mode="positive",
        same_image_iou_threshold=0.3,
    )

    loss, stats = loss_fn(image, text, torch.tensor(10.0), sigmoid_map=sigmoid_map)
    labels, weights, _ = loss_fn.build_labels_and_weights(torch.zeros(2, 2), 1, 2, sigmoid_map)

    assert torch.isfinite(loss)
    assert labels[0, 1].item() == 0.0
    assert weights[0, 1].item() == 0.0
    assert stats["caption_num_ignored_pairs"].item() == 2.0


def test_masked_caption_positive_mode_has_no_same_image_negative() -> None:
    image = torch.randn(1, 3, 8, requires_grad=True)
    text = torch.randn(1, 3, 8, requires_grad=True)
    sigmoid_map = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]]]
    )
    loss_fn = MaskedCaptionSigLipLoss(
        caption_same_image_mode="positive",
        same_image_iou_threshold=0.3,
    )

    loss, stats = loss_fn(image, text, torch.tensor(10.0), sigmoid_map=sigmoid_map)
    labels, weights, _ = loss_fn.build_labels_and_weights(torch.zeros(3, 3), 1, 3, sigmoid_map)

    assert torch.isfinite(loss)
    assert labels[0, 1].item() > 0.3
    assert weights[0, 2].item() == 0.0
    assert labels[0, 2].item() == 0.0
    assert stats["caption_same_image_mode_code"].item() == 1.0


def test_masked_caption_ignore_mode_keeps_region_terms_zero() -> None:
    image = torch.randn(1, 3, 8, requires_grad=True)
    text = torch.randn(1, 3, 8, requires_grad=True)
    sigmoid_map = torch.ones(1, 3, 1, 4)
    loss_fn = MaskedCaptionSigLipLoss(caption_same_image_mode="ignore")

    loss, stats = loss_fn(image, text, torch.tensor(10.0), sigmoid_map=sigmoid_map)
    labels, weights, _ = loss_fn.build_labels_and_weights(torch.zeros(3, 3), 1, 3, sigmoid_map)

    assert torch.isfinite(loss)
    assert torch.all(weights[torch.tensor([[False, True, True], [True, False, True], [True, True, False]])] == 0)
    assert torch.all(labels.diag() == 1)
    assert torch.all(labels[torch.tensor([[False, True, True], [True, False, True], [True, True, False]])] == 0)
    assert stats["caption_same_image_mode_code"].item() == 0.0


def test_masked_caption_ddp_adds_negative_only_text_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    bsz, num_captions, dim = 2, 2, 8
    image = torch.randn(bsz, num_captions, dim, requires_grad=True)
    text = torch.randn(bsz, num_captions, dim, requires_grad=True)
    other_text = F.normalize(torch.randn(bsz * num_captions, dim), dim=-1)
    loss_fn = MaskedCaptionSigLipLoss(world_size=2, rank=0)

    monkeypatch.setattr(loss_fn, "_negative_text_feature_batches", lambda _: [other_text])
    loss, stats = loss_fn(image, text, torch.tensor(10.0))

    assert torch.isfinite(loss)
    assert stats["caption_ddp_negative_loss"].item() > 0.0
    loss.backward()
    assert image.grad is not None
    assert text.grad is not None


def test_spvd_loss_masked_caption_path_with_global_disabled() -> None:
    bsz, num_captions, dim = 2, 3, 8
    gate_logits = torch.randn(bsz, num_captions, 4, 6)
    sigmoid_map = torch.sigmoid(gate_logits)
    outputs = {
        "caption_semantic_visual_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "caption_text_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "text_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "residual_visual_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "gate_logits": gate_logits,
        "sigmoid_map": sigmoid_map,
        "residual_map": 1.0 - sigmoid_map,
        "logit_scale": torch.tensor(10.0),
    }
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
    loss_fn.set_global_step(5)

    loss, loss_dict = loss_fn(outputs)

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["loss_align_caption"])
    assert loss_dict["loss_align_global"].item() == 0.0
    assert loss_dict["global_align_enabled"].item() == 0.0
    assert loss_dict["caption_align_enabled"].item() == 1.0
    assert loss_dict["caption_masked_same_image_pairs"].item() > 0
    assert "loss_branch_weight_effective" in loss_dict
    assert "loss_residual_variance_weight_effective" in loss_dict
    assert torch.allclose(loss_dict["loss_branch_weight_effective"], torch.tensor(0.025))
    assert torch.allclose(loss_dict["loss_residual_variance_weight_effective"], torch.tensor(0.01))
    loss.backward()
    assert outputs["caption_semantic_visual_features"].grad is not None
    assert outputs["caption_text_features"].grad is not None


def test_spvd_loss_passes_sigmoid_map_to_region_caption_loss() -> None:
    bsz, num_captions, dim = 1, 3, 8
    gate_logits = torch.randn(bsz, num_captions, 1, 4)
    sigmoid_map = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]]]
    )
    outputs = {
        "caption_semantic_visual_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "caption_text_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "text_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "residual_visual_features": torch.randn(bsz, num_captions, dim, requires_grad=True),
        "gate_logits": gate_logits,
        "sigmoid_map": sigmoid_map,
        "residual_map": 1.0 - torch.sigmoid(gate_logits),
        "logit_scale": torch.tensor(10.0),
    }
    loss_fn = SPVDLoss(
        rank=0,
        world_size=1,
        global_align_weight=0.0,
        caption_align_weight=1.0,
        caption_loss_impl="masked_sigmoid",
        caption_same_image_mode="positive",
        branch_bce_weight=0.0,
        residual_variance_weight=0.0,
    )
    loss_fn.set_global_step(5)

    loss, loss_dict = loss_fn(outputs)

    assert torch.isfinite(loss)
    assert loss_dict["caption_same_image_mode_code"].item() == 1.0
    assert loss_dict["caption_num_positive_pairs"].item() > bsz * num_captions
    loss.backward()
    assert outputs["caption_semantic_visual_features"].grad is not None
    assert outputs["caption_text_features"].grad is not None
