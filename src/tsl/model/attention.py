"""Causal self-attention modules.

Baseline: standard multi-head attention (MHA).
Grouped-query attention (GQA) is added in Phase 5; ``num_kv_heads`` is already
accepted so the factory contract does not need to change later.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.constants import ATTN_MHA, ATTN_VARIANTS


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention for a decoder-only transformer.

    Tensor flow (MHA, ``num_kv_heads == num_heads``)::

        x:          (B, T, C)
        q, k, v:    (B, T, C)  -> reshape -> (B, H, T, Dh)
        scores:     (B, H, T, T) with causal mask
        context:    (B, H, T, Dh) -> (B, T, C)
        out:        (B, T, C)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout

        if self.num_kv_heads != self.num_heads:
            # GQA wiring lands in Phase 5; keep the constructor ready.
            raise NotImplementedError(
                "num_kv_heads != num_heads requires GQA (Phase 5). "
                f"Got num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}"
            )

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: True means "allowed to attend".
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def _shape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """(B, T, H*Dh) -> (B, H, T, Dh)."""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        if hidden != self.hidden_size:
            raise ValueError(f"Expected hidden={self.hidden_size}, got {hidden}")
        if seq > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {seq} exceeds attention max_seq_len={self.causal_mask.size(0)}"
            )

        q = self._shape_heads(self.q_proj(x), self.num_heads)
        k = self._shape_heads(self.k_proj(x), self.num_kv_heads)
        v = self._shape_heads(self.v_proj(x), self.num_kv_heads)

        # Scaled dot-product attention with explicit causal mask.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:seq, :seq]
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = torch.matmul(weights, v)  # (B, H, T, Dh)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)
        out = self.out_proj(context)
        return self.resid_dropout(out)


def build_attention(
    kind: str,
    hidden_size: int,
    num_heads: int,
    *,
    num_kv_heads: int | None = None,
    dropout: float = 0.0,
    max_seq_len: int = 2048,
) -> nn.Module:
    """Construct an attention module by config name.

    Phase 3 supports ``mha`` only. ``gqa`` lands in Phase 5.
    """
    kind = kind.lower()
    if kind not in ATTN_VARIANTS:
        raise ValueError(f"Unknown attention variant {kind!r}; expected one of {ATTN_VARIANTS}")
    if kind == ATTN_MHA:
        kv = num_kv_heads if num_kv_heads is not None else num_heads
        if kv != num_heads:
            raise ValueError("MHA requires num_kv_heads == num_heads")
        return CausalSelfAttention(
            hidden_size,
            num_heads,
            num_kv_heads=kv,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    raise NotImplementedError(f"Attention variant {kind!r} is not implemented yet (Phase 5).")
