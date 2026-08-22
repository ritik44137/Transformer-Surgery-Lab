"""Causal self-attention modules.

Swap axes:
- Attention: MHA (baseline) vs GQA (Phase 5)
- Positional: optional RoPE applied to Q/K (when ``use_rope=True``)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.constants import ATTN_GQA, ATTN_MHA, ATTN_VARIANTS
from tsl.model.positional import RotaryEmbedding, apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads so they align with query heads.

    Input:  ``(batch, num_kv_heads, seq, head_dim)``
    Output: ``(batch, num_kv_heads * n_rep, seq, head_dim)``

    For GQA, each KV head is shared by ``n_rep = num_heads // num_kv_heads``
    query heads.
    """
    if n_rep == 1:
        return x
    batch, n_kv, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv, n_rep, seq, head_dim)
    return x.reshape(batch, n_kv * n_rep, seq, head_dim)


class CausalSelfAttention(nn.Module):
    """Causal self-attention supporting MHA and GQA, with optional RoPE.

    Tensor flow::

        x:          (B, T, C)
        q:          (B, Hq, T, Dh)
        k, v:       (B, Hkv, T, Dh)  — repeated to Hq for GQA
        scores:     (B, Hq, T, T) with causal mask
        out:        (B, T, C)

    Constraints:
    - ``num_heads`` must be divisible by ``num_kv_heads``
    - MHA is the special case ``num_kv_heads == num_heads``
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = False,
        rope_base: float = 10000.0,
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
        self.use_rope = use_rope

        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: True means "allowed to attend".
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.rotary: RotaryEmbedding | None
        if use_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=max_seq_len, base=rope_base
            )
        else:
            self.rotary = None

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

        if self.rotary is not None:
            cos, sin = self.rotary(seq)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads to match query heads for GQA (no-op for MHA).
        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:seq, :seq]
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = torch.matmul(weights, v)  # (B, Hq, T, Dh)
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
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> nn.Module:
    """Construct an attention module by config name."""
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
            use_rope=use_rope,
            rope_base=rope_base,
        )

    if kind == ATTN_GQA:
        if num_kv_heads is None:
            raise ValueError("GQA requires explicit num_kv_heads in config")
        if num_kv_heads == num_heads:
            raise ValueError(
                "GQA expects num_kv_heads < num_heads; "
                f"got num_heads={num_heads}, num_kv_heads={num_kv_heads}"
            )
        return CausalSelfAttention(
            hidden_size,
            num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_base=rope_base,
        )

    raise ValueError(f"Unhandled attention variant {kind!r}")
