from __future__ import annotations

import torch
from open_clip.model import CLIP

from factory import create_model_and_transforms, create_tokenizer


def test_openclip_forward_smoke() -> None:
    model, _, _ = create_model_and_transforms(
        "ViT-B-16",
        pretrained="",
        precision="fp32",
        device="cpu",
        force_image_size=32,
        output_dict=True,
    )
    assert isinstance(model, CLIP)
    assert not hasattr(model, "visual_proj")
    tokenizer = create_tokenizer("ViT-B-16")
    images = torch.randn(1, 3, 32, 32)
    texts = tokenizer(["a tiny smoke test"])
    outputs = model(images, texts)
    assert outputs["image_features"].shape[0] == 1
    assert outputs["text_features"].shape[0] == 1
    assert outputs["logit_scale"].ndim == 0
