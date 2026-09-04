"""MHA / GQA shapes and basic causal masking."""

from __future__ import annotations

import torch

from tsl.model.attention import CausalSelfAttention, build_attention, repeat_kv


@torch.no_grad()
def test_mha_output_shape() -> None:
    b, t, h, heads = 2, 6, 32, 4
    attn = CausalSelfAttention(h, heads, max_seq_len=16)
    x = torch.randn(b, t, h)
    y = attn(x)
    assert y.shape == (b, t, h)
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_gqa_output_shape() -> None:
    b, t, h, heads, kv = 2, 5, 48, 6, 2
    attn = CausalSelfAttention(h, heads, num_kv_heads=kv, max_seq_len=16)
    assert attn.num_queries_per_kv == 3
    y = attn(torch.randn(b, t, h))
    assert y.shape == (b, t, h)
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_gqa_with_rope() -> None:
    attn = build_attention("gqa", 32, 4, num_kv_heads=2, max_seq_len=16, use_rope=True)
    y = attn(torch.randn(1, 4, 32))
    assert y.shape == (1, 4, 32)
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_causal_mask_blocks_future() -> None:
    """With identity projections, future positions must not leak into past."""
    torch.manual_seed(0)
    h, heads, t = 8, 2, 4
    attn = CausalSelfAttention(h, heads, max_seq_len=8, dropout=0.0)
    # Force projections toward identity-ish so masking dominates the check.
    with torch.no_grad():
        eye = torch.eye(h)
        attn.q_proj.weight.copy_(eye)
        attn.k_proj.weight.copy_(eye)
        attn.v_proj.weight.copy_(eye)
        attn.out_proj.weight.copy_(eye)

    x = torch.zeros(1, t, h)
    x[0, -1] = 10.0  # spike only at the last position
    y = attn(x)
    # First token cannot see the spike at the end.
    assert y[0, 0].abs().max() < y[0, -1].abs().max()


@torch.no_grad()
def test_repeat_kv() -> None:
    x = torch.randn(2, 2, 5, 4)
    y = repeat_kv(x, 3)
    assert y.shape == (2, 6, 5, 4)
