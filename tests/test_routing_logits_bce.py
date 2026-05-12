from __future__ import annotations

import pytest
import torch

try:
    from losses import bidirectional_routing_bce_with_logits_loss
except ImportError:
    from src.losses import bidirectional_routing_bce_with_logits_loss


def test_routing_bce_logits_single_caption_shape_backward():
    B, S, M = 2, 4, 16
    routing_logits = torch.randn(B, S, M, requires_grad=True)
    relevance_scores = torch.rand(B, S, M)
    cue_weights = torch.softmax(torch.randn(B, S), dim=-1)

    loss = bidirectional_routing_bce_with_logits_loss(
        routing_logits=routing_logits,
        relevance_scores=relevance_scores,
        cue_weights=cue_weights,
        positive_constraint=True,
        negative_constraint=True,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert routing_logits.grad is not None
    assert torch.isfinite(routing_logits.grad).all()


def test_routing_bce_logits_multi_caption_shape_backward():
    B, K, S, M = 2, 3, 4, 16
    routing_logits = torch.randn(B, K, S, M, requires_grad=True)
    relevance_scores = torch.rand(B, K, S, M)
    cue_weights = torch.softmax(torch.randn(B, K, S), dim=-1)

    loss = bidirectional_routing_bce_with_logits_loss(
        routing_logits=routing_logits,
        relevance_scores=relevance_scores,
        cue_weights=cue_weights,
        positive_constraint=True,
        negative_constraint=True,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert routing_logits.grad is not None
    assert torch.isfinite(routing_logits.grad).all()


def test_routing_bce_logits_extreme_values_are_finite():
    routing_logits = torch.tensor([[[-50.0, 50.0, 0.0]]], requires_grad=True)
    relevance_scores = torch.tensor([[[0.99, 0.01, 0.5]]])

    loss = bidirectional_routing_bce_with_logits_loss(
        routing_logits=routing_logits,
        relevance_scores=relevance_scores,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert routing_logits.grad is not None
    assert torch.isfinite(routing_logits.grad).all()


def test_routing_bce_logits_constraint_modes():
    B, S, M = 2, 4, 8
    routing_logits = torch.randn(B, S, M)
    relevance_scores = torch.rand(B, S, M)

    for pos, neg in [(True, False), (False, True), (True, True)]:
        logits = routing_logits.detach().clone().requires_grad_(True)
        loss = bidirectional_routing_bce_with_logits_loss(
            routing_logits=logits,
            relevance_scores=relevance_scores,
            positive_constraint=pos,
            negative_constraint=neg,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_routing_bce_logits_no_constraints_returns_zero():
    routing_logits = torch.randn(2, 4, 8, requires_grad=True)
    relevance_scores = torch.rand(2, 4, 8)

    loss = bidirectional_routing_bce_with_logits_loss(
        routing_logits=routing_logits,
        relevance_scores=relevance_scores,
        positive_constraint=False,
        negative_constraint=False,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_routing_bce_logits_shape_mismatch_raises():
    routing_logits = torch.randn(2, 4, 8)
    relevance_scores = torch.rand(2, 4, 7)

    with pytest.raises(ValueError):
        bidirectional_routing_bce_with_logits_loss(
            routing_logits=routing_logits,
            relevance_scores=relevance_scores,
        )


def test_routing_bce_logits_invalid_cue_weight_shape_raises_single_caption():
    routing_logits = torch.randn(2, 4, 8)
    relevance_scores = torch.rand(2, 4, 8)
    cue_weights = torch.rand(2, 3, 4)

    with pytest.raises(ValueError):
        bidirectional_routing_bce_with_logits_loss(
            routing_logits=routing_logits,
            relevance_scores=relevance_scores,
            cue_weights=cue_weights,
        )


def test_routing_bce_logits_invalid_cue_weight_shape_raises_multi_caption():
    routing_logits = torch.randn(2, 3, 4, 8)
    relevance_scores = torch.rand(2, 3, 4, 8)
    cue_weights = torch.rand(2, 4)

    with pytest.raises(ValueError):
        bidirectional_routing_bce_with_logits_loss(
            routing_logits=routing_logits,
            relevance_scores=relevance_scores,
            cue_weights=cue_weights,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_routing_bce_logits_cuda_autocast_smoke():
    B, S, M = 2, 4, 16
    routing_logits = torch.randn(B, S, M, device="cuda", requires_grad=True)
    relevance_scores = torch.rand(B, S, M, device="cuda")
    cue_weights = torch.softmax(torch.randn(B, S, device="cuda"), dim=-1)

    with torch.autocast(device_type="cuda"):
        loss = bidirectional_routing_bce_with_logits_loss(
            routing_logits=routing_logits,
            relevance_scores=relevance_scores,
            cue_weights=cue_weights,
        )

    assert torch.isfinite(loss)
    loss.backward()
    assert routing_logits.grad is not None
    assert torch.isfinite(routing_logits.grad).all()
