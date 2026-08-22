"""Positional encoding helpers.

Swap axis: additive sinusoidal (baseline) vs RoPE (Phase 5).

RoPE does **not** add vectors to embeddings. It rotates query/key pairs in
attention so relative position is encoded in the attention scores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsl.constants import POS_ROPE, POS_SINUSOIDAL, POS_VARIANTS


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings added to token embeddings.

    Expected input shape: ``(batch, seq, hidden_size)``
    Output shape: same as input.
    """

    def __init__(self, hidden_size: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for sinusoidal PE, got {hidden_size}")

        pe = torch.zeros(max_seq_len, hidden_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_seq_len, hidden_size) for easy broadcasting
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to *x*."""
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        return x + self.pe[:, :seq_len]


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for rotary position embeddings.

    For a 2D pair ``(x0, x1)`` at position ``t`` with angle ``θ_t``::

        x0' = x0 * cos(θ_t) - x1 * sin(θ_t)
        x1' = x0 * sin(θ_t) + x1 * cos(θ_t)

    Relative position falls out of the Q·K product after both are rotated.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        # Inverse frequencies for each pair dimension: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        # (max_seq_len, head_dim/2)
        freqs = torch.outer(positions, inv_freq)
        # Duplicate for even/odd interleaving: (max_seq_len, head_dim)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds RoPE max_seq_len={self.max_seq_len}")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``(..., x0, x1, x2, x3, ...)`` → ``(..., -x1, x0, -x3, x2, ...)``."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    # Stack as (-odd, even) then flatten last dim back to head_dim.
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a tensor of shape ``(batch, heads, seq, head_dim)``.

    ``cos`` / ``sin`` are ``(seq, head_dim)`` and broadcast over batch/heads.
    """
    # (1, 1, seq, head_dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    return (x * cos) + (rotate_half(x) * sin)


def build_positional(
    kind: str,
    hidden_size: int,
    max_seq_len: int = 2048,
) -> nn.Module | None:
    """Return an additive positional module, or ``None`` for RoPE.

    ``sinusoidal`` → :class:`SinusoidalPositionalEncoding`
    ``rope`` → ``None`` (rotation is applied inside attention)
    """
    kind = kind.lower()
    if kind not in POS_VARIANTS:
        raise ValueError(f"Unknown positional variant {kind!r}; expected one of {POS_VARIANTS}")
    if kind == POS_SINUSOIDAL:
        return SinusoidalPositionalEncoding(hidden_size, max_seq_len=max_seq_len)
    if kind == POS_ROPE:
        return None
    raise ValueError(f"Unhandled positional variant {kind!r}")
