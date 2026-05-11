"""Project-local BPE tokenizer helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH, SimpleTokenizer


_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
_BPE_PATH = _ASSET_DIR / _BPE_FILENAME


def default_bpe_path() -> str:
    """Return the vendored OpenAI CLIP BPE vocabulary path."""
    if not _BPE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing BPE vocabulary at {_BPE_PATH}. "
            "Copy bpe_simple_vocab_16e6.txt.gz into src/assets/."
        )
    return str(_BPE_PATH)


@lru_cache(maxsize=16)
def _cached_bpe_tokenizer(
    context_length: int,
    clean: str,
    reduction_mask: str,
    additional_special_tokens: tuple[str, ...],
) -> SimpleTokenizer:
    """Build and cache a SimpleTokenizer so the gzip BPE file is read once."""
    return SimpleTokenizer(
        bpe_path=default_bpe_path(),
        additional_special_tokens=list(additional_special_tokens) or None,
        context_length=context_length,
        clean=clean,
        reduction_mask=reduction_mask,
    )


def get_bpe_tokenizer(
    context_length: int | None = None,
    clean: str = "lower",
    reduction_mask: str = "",
    additional_special_tokens: list[str] | tuple[str, ...] | None = None,
    **_: Any,
) -> SimpleTokenizer:
    """Return a cached project-local BPE tokenizer.

    This intentionally bypasses OpenCLIP's generic tokenizer dispatch and uses
    the vendored BPE vocabulary, avoiding repeated cache/download checks.
    """
    return _cached_bpe_tokenizer(
        int(context_length or DEFAULT_CONTEXT_LENGTH),
        str(clean),
        str(reduction_mask),
        tuple(additional_special_tokens or ()),
    )
