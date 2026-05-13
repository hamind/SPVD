from __future__ import annotations

import torch

from losses import (
    BranchBCELoss,
    ResidualVarianceLoss,
)
from model import SoftCueExtractor, SoftCueSigmoidDecomposition


def test_soft_cue_decomposition_shapes_bounds_and_backward() -> None:
    batch_size = 2
    num_tokens = 196
    text_len = 32
    dim = 512
    num_cues = 4

    extractor = SoftCueExtractor(
        text_dim=dim,
        embed_dim=dim,
        num_soft_cues=num_cues,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    decomposition = SoftCueSigmoidDecomposition(
        visual_dim=dim,
        embed_dim=dim,
        gate_temperature=1.0,
        gate_bias_init=0.0,
    )

    visual_tokens = torch.randn(batch_size, num_tokens, dim)
    text_tokens = torch.randn(batch_size, text_len, dim)
    text_features = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)

    soft_cues = extractor(text_tokens)
    outputs = decomposition(visual_tokens, soft_cues)

    sigmoid_map = outputs["sigmoid_map"]
    residual_map = outputs["residual_map"]
    gate_logits = outputs["gate_logits"]
    z_s = outputs["semantic_features"]
    z_r = outputs["residual_features"]

    assert soft_cues.shape == (batch_size, num_cues, dim)
    assert sigmoid_map.shape == (batch_size, num_cues, num_tokens)
    assert residual_map.shape == (batch_size, num_cues, num_tokens)
    assert gate_logits.shape == (batch_size, num_cues, num_tokens)
    assert z_s.shape == (batch_size, dim)
    assert z_r.shape == (batch_size, dim)

    assert torch.allclose(sigmoid_map + residual_map, torch.ones_like(sigmoid_map), atol=1.0e-5)
    assert "image_attention" not in outputs
    assert "relevance_scores" not in outputs
    assert "shared_routing" not in outputs
    assert "residual_routing" not in outputs
    assert "routing_logits" not in outputs
    assert sigmoid_map.min() >= 0
    assert sigmoid_map.max() <= 1
    assert residual_map.min() >= 0
    assert residual_map.max() <= 1
    assert torch.isfinite(sigmoid_map).all()
    assert sigmoid_map.std(unbiased=False) > 0

    loss_branch = BranchBCELoss()(z_s, z_r, text_features)["loss_branch"]
    loss_res = ResidualVarianceLoss()(z_r)
    total_loss = loss_branch + loss_res
    total_loss.backward()

    assert extractor.soft_cue_slots.grad is not None
    assert decomposition.query_proj.weight.grad is not None
    assert decomposition.key_proj.weight.grad is not None
    assert decomposition.semantic_value_proj.weight.grad is not None
    assert decomposition.residual_value_proj.weight.grad is not None
    assert decomposition.gate_bias.grad is not None


def test_soft_cue_extractor_masks_padding_tokens() -> None:
    dim = 32
    extractor = SoftCueExtractor(
        text_dim=dim,
        embed_dim=dim,
        num_soft_cues=4,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    extractor.eval()
    text_tokens = torch.randn(2, 6, dim)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    changed_padding = text_tokens.clone()
    changed_padding[~attention_mask] = torch.randn_like(changed_padding[~attention_mask]) * 1000

    with torch.no_grad():
        cues = extractor(text_tokens, attention_mask=attention_mask)
        cues_changed = extractor(changed_padding, attention_mask=attention_mask)

    assert torch.allclose(cues, cues_changed, atol=1.0e-5)
