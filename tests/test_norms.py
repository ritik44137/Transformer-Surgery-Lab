"""LayerNorm / RMSNorm shape and numerical sanity."""

from __future__ import annotations

import torch

from tsl.model.norms import LayerNorm, RMSNorm, build_norm


@torch.no_grad()
def test_layernorm_shape_and_finite() -> None:
    h = 32
    x = torch.randn(2, 5, h)
    y = LayerNorm(h)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_rmsnorm_shape_and_finite() -> None:
    h = 32
    x = torch.randn(2, 5, h)
    y = RMSNorm(h)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_rmsnorm_rms_near_one() -> None:
    h = 64
    x = torch.randn(4, 8, h)
    y = RMSNorm(h)(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


@torch.no_grad()
def test_build_norm_variants() -> None:
    x = torch.randn(1, 3, 16)
    for kind in ("layernorm", "rmsnorm"):
        y = build_norm(kind, 16)(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
