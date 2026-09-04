"""End-to-end DecoderLM construction and forward shapes."""

from __future__ import annotations

import torch

from tsl.model.factory import build_model


def _toy_cfg(**overrides) -> dict:
    cfg = {
        "model": {
            "vocab_size": 50,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 4,
            "d_ff": 64,
            "max_seq_len": 16,
            "dropout": 0.0,
            "norm": "layernorm",
            "positional": "sinusoidal",
            "feedforward": "relu",
            "attention": "mha",
        }
    }
    cfg["model"].update(overrides)
    return cfg


@torch.no_grad()
def test_baseline_forward_logits_shape() -> None:
    model = build_model(_toy_cfg())
    ids = torch.randint(0, 50, (2, 7))
    logits = model(ids)
    assert logits.shape == (2, 7, 50)
    assert torch.isfinite(logits).all()


@torch.no_grad()
def test_modern_stack_forward() -> None:
    model = build_model(
        _toy_cfg(
            norm="rmsnorm",
            positional="rope",
            feedforward="swiglu",
            attention="gqa",
            num_kv_heads=2,
        )
    )
    logits = model(torch.randint(0, 50, (1, 5)))
    assert logits.shape == (1, 5, 50)
    assert torch.isfinite(logits).all()
    assert model.count_parameters() > 0
