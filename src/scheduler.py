"""Learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer


def cosine_lr(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable[[int], float]:
    """Return a step-wise cosine LR scheduler closure."""
    steps = max(int(steps), 1)
    warmup_length = max(int(warmup_length), 0)

    def _lr_adjuster(step: int) -> float:
        if warmup_length > 0 and step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            progress = (step - warmup_length) / max(1, steps - warmup_length)
            lr = 0.5 * (1.0 + math.cos(math.pi * progress)) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster
