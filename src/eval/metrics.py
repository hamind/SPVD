"""Retrieval metrics."""

from __future__ import annotations

import torch


def retrieval_metrics(similarity: torch.Tensor) -> dict[str, float]:
    """Compute paired image-text retrieval metrics."""
    n = similarity.shape[0]
    target = torch.arange(n)
    i2t_order = similarity.argsort(dim=1, descending=True)
    t2i_order = similarity.t().argsort(dim=1, descending=True)

    def ranks(order: torch.Tensor) -> torch.Tensor:
        return (order == target[:, None]).nonzero()[:, 1] + 1

    i2t = ranks(i2t_order)
    t2i = ranks(t2i_order)
    out: dict[str, float] = {}
    for prefix, value in [("i2t", i2t), ("t2i", t2i)]:
        for k in (1, 5, 10):
            out[f"{prefix}_r@{k}"] = float((value <= k).float().mean().item() * 100)
        out[f"{prefix}_mean_rank"] = float(value.float().mean().item())
        out[f"{prefix}_median_rank"] = float(value.float().median().item())
    return out
