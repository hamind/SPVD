"""Make the local src/ layout importable when running from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir():
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
