"""ReLU MLP and SwiGLU feed-forward shape checks."""

from __future__ import annotations

import torch

from tsl.model.feedforward import ReLUFeedForward, SwiGLUFeedForward, build_feedforward


@torch.no_grad()
def test_relu_ff_shape() -> None:
    h, d_ff = 32, 64
    x = torch.randn(2, 5, h)
    y = ReLUFeedForward(h, d_ff)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_swiglu_ff_shape_and_parity_scaling() -> None:
    h, d_ff = 32, 96
    x = torch.randn(2, 5, h)
    ff = SwiGLUFeedForward(h, d_ff, scale_for_param_parity=True)
    assert ff.d_ff == int(round(2 * d_ff / 3)) + (int(round(2 * d_ff / 3)) % 2)
    y = ff(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_build_feedforward_both_variants() -> None:
    x = torch.randn(1, 3, 24)
    for kind in ("relu", "swiglu"):
        y = build_feedforward(kind, 24, d_ff=48)(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
