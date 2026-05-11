"""Small distributed helpers inspired by OpenCLIP's training utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedInfo:
    """Runtime distributed state."""

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def init_distributed_device(args: object) -> DistributedInfo:
    """Initialize torch.distributed if launched with torchrun."""
    world_size = _int_env("WORLD_SIZE", 1)
    rank = _int_env("RANK", 0)
    local_rank = _int_env("LOCAL_RANK", 0)
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        backend = getattr(args, "dist_backend", "nccl")
        if device.type == "cpu" and backend == "nccl":
            backend = "gloo"
        init_kwargs = {
            "backend": backend,
            "init_method": getattr(args, "dist_url", "env://"),
        }
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
    return DistributedInfo(device=device, rank=rank, local_rank=local_rank, world_size=world_size, distributed=distributed)


def is_master(args: object | None = None) -> bool:
    """Return True on rank zero."""
    if args is not None and hasattr(args, "rank"):
        return int(getattr(args, "rank")) == 0
    return get_rank() == 0


def get_rank() -> int:
    """Return current rank."""
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    """Return current world size."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def cleanup_distributed() -> None:
    """Tear down distributed state."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module for DDP-wrapped models."""
    return model.module if hasattr(model, "module") else model
