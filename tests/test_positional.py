"""Sinusoidal PE and RoPE shape / finite checks."""

from __future__ import annotations

import torch

from tsl.model.positional import (
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    apply_rotary_emb,
    build_positional,
    rotate_half,
)


@torch.no_grad()
def test_sinusoidal_preserves_shape() -> None:
    h, t = 32, 7
    pe = SinusoidalPositionalEncoding(h, max_seq_len=64)
    x = torch.randn(2, t, h)
    y = pe(x)
    assert y.shape == (2, t, h)
    assert torch.isfinite(y).all()
    # Additive: output differs from input unless PE is zero.
    assert not torch.allclose(y, x)


@torch.no_grad()
def test_rotary_tables_and_apply() -> None:
    head_dim, seq = 8, 5
    rope = RotaryEmbedding(head_dim, max_seq_len=32)
    cos, sin = rope(seq)
    assert cos.shape == (seq, head_dim)
    assert sin.shape == (seq, head_dim)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()

    q = torch.randn(2, 4, seq, head_dim)
    q_rot = apply_rotary_emb(q, cos, sin)
    assert q_rot.shape == q.shape
    assert torch.isfinite(q_rot).all()


@torch.no_grad()
def test_rotate_half_pairing() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    y = rotate_half(x)
    assert torch.allclose(y, torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))


@torch.no_grad()
def test_build_positional() -> None:
    assert isinstance(build_positional("sinusoidal", 16, max_seq_len=8), SinusoidalPositionalEncoding)
    assert build_positional("rope", 16, max_seq_len=8) is None
