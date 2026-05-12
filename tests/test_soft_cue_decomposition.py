from __future__ import annotations

import torch

from losses import (
    bidirectional_routing_bce_loss,
    residual_preservation_loss,
    shared_residual_decorrelation_loss,
)
from model import SoftCueBidirectionalDecomposition, SoftCueExtractor


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
    decomposition = SoftCueBidirectionalDecomposition(
        visual_dim=dim,
        embed_dim=dim,
        relevance_temperature=1.0,
        routing_temperature=1.0,
    )

    visual_tokens = torch.randn(batch_size, num_tokens, dim)
    text_tokens = torch.randn(batch_size, text_len, dim)

    soft_cues = extractor(text_tokens)
    outputs = decomposition(visual_tokens, soft_cues)

    rho = outputs["relevance_scores"]
    routing_logits = outputs["routing_logits"]
    m_s = outputs["shared_routing"]
    m_r = outputs["residual_routing"]
    routing_probs = outputs["routing_probs"]
    routing_pair_logits = outputs["routing_pair_logits"]
    z_s_k = outputs["cue_visual_features"]
    z_r_k = outputs["cue_residual_features"]
    z_s = outputs["shared_visual_features"]
    z_r = outputs["residual_visual_features"]
    alpha = outputs["cue_weights"]

    assert soft_cues.shape == (batch_size, num_cues, dim)
    assert rho.shape == (batch_size, num_cues, num_tokens)
    assert routing_logits.shape == (batch_size, num_cues, num_tokens)
    assert m_s.shape == (batch_size, num_cues, num_tokens)
    assert m_r.shape == (batch_size, num_cues, num_tokens)
    assert routing_probs.shape == (batch_size, num_cues, num_tokens, 2)
    assert routing_pair_logits.shape == (batch_size, num_cues, num_tokens, 2)
    assert z_s_k.shape == (batch_size, num_cues, dim)
    assert z_r_k.shape == (batch_size, num_cues, dim)
    assert z_s.shape == (batch_size, dim)
    assert z_r.shape == (batch_size, dim)
    assert alpha.shape == (batch_size, num_cues)

    assert torch.allclose(m_s + m_r, torch.ones_like(m_s), atol=1.0e-5)
    assert m_s.min() >= 0
    assert m_s.max() <= 1
    assert m_r.min() >= 0
    assert m_r.max() <= 1
    assert rho.min() >= 0
    assert rho.max() <= 1

    loss_decomp = bidirectional_routing_bce_loss(m_s, m_r, rho, alpha)
    loss_res = residual_preservation_loss(z_r)
    loss_orth = shared_residual_decorrelation_loss(z_s, z_r)
    total_loss = loss_decomp + loss_res + loss_orth
    total_loss.backward()

    assert extractor.soft_cue_slots.grad is not None
    assert decomposition.router.q_proj.weight.grad is not None
    assert decomposition.router.v_proj.weight.grad is not None
    assert decomposition.shared_out_proj.weight.grad is not None


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
