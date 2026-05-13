from __future__ import annotations

import torch

from factory import create_tokenizer
from tokenizer import default_bpe_path


def test_bpe_tokenizer_uses_vendored_vocab_and_is_cached() -> None:
    tokenizer_a = create_tokenizer("SPVD-ViT-B-16")
    tokenizer_b = create_tokenizer("SPVD-ViT-B-16")

    assert tokenizer_a is tokenizer_b
    assert default_bpe_path().endswith("src/assets/bpe_simple_vocab_16e6.txt.gz")

    tokens = tokenizer_a(["a small image"])
    assert tokens.shape == (1, 77)
    assert tokens.dtype == torch.long
