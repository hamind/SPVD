"""Benchmark evaluation launcher for SPVD semantic-preservation diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zero_train_diagnostics.evaluator import run_diagnostics


def _args_with_defaults(**kwargs) -> argparse.Namespace:
    defaults = {
        "config": "configs/benchmark_eval.yaml",
        "dataset_root": None,
        "model_root": None,
        "output_dir": None,
        "device": None,
        "dtype": None,
        "batch_size": None,
        "num_workers": None,
        "seed": None,
        "limit": None,
        "dry_run": False,
        "models": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPVD benchmark semantic-preservation evaluation.")
    parser.add_argument("--config", default="configs/benchmark_eval.yaml")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--models", default=None, help="Optional comma-separated model-name subset.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_dir = run_diagnostics(parse_args(argv))
    print(f"RUN_DIR={run_dir}")


def eval_aro(**kwargs):
    """Compatibility wrapper around the benchmark launcher."""
    return run_diagnostics(_args_with_defaults(**kwargs))


def eval_sugarcrepe(**kwargs):
    """Compatibility wrapper around the benchmark launcher."""
    return run_diagnostics(_args_with_defaults(**kwargs))


def eval_winoground(**kwargs):
    """Compatibility wrapper around the benchmark launcher."""
    return run_diagnostics(_args_with_defaults(**kwargs))


if __name__ == "__main__":
    main()
