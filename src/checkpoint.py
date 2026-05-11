"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer, epoch: int, step: int, args: object) -> None:
    """Save a training checkpoint."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer | None = None, map_location: Any = "cpu") -> dict[str, Any]:
    """Load model and optionally optimizer state."""
    payload = torch.load(path, map_location=map_location)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload if isinstance(payload, dict) else {"state_dict": state_dict}
