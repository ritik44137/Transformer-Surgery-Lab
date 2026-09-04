"""Transformer block composition and residual shape preservation."""

from __future__ import annotations

import torch

from tsl.model.attention import build_attention
from tsl.model.block import TransformerBlock
from tsl.model.feedforward import build_feedforward
from tsl.model.norms import build_norm


def _make_block(
    *,
    hidden: int = 32,
    heads: int = 4,
    kv: int | None = None,
    norm: str = "layernorm",
    ff: str = "relu",
    use_rope: bool = False,
) -> TransformerBlock:
    return TransformerBlock(
        norm_attn=build_norm(norm, hidden),
        attention=build_attention(
            "mha" if kv is None or kv == heads else "gqa",
            hidden,
            heads,
            num_kv_heads=kv if kv is not None else heads,
            max_seq_len=32,
            use_rope=use_rope,
        ),
        norm_ff=build_norm(norm, hidden),
        feedforward=build_feedforward(ff, hidden, d_ff=hidden * 2),
    )


@torch.no_grad()
def test_block_baseline_shape() -> None:
    block = _make_block()
    x = torch.randn(2, 6, 32)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_block_modern_stack_shape() -> None:
    block = _make_block(norm="rmsnorm", ff="swiglu", kv=2, heads=4, use_rope=True)
    x = torch.randn(1, 5, 32)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
