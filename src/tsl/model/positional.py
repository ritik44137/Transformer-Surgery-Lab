"""Positional encoding helpers.

Baseline: additive sinusoidal encodings.
RoPE helpers are added in Phase 5 without mixing into the additive path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsl.constants import POS_SINUSOIDAL, POS_VARIANTS


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


def build_positional(
    kind: str,
    hidden_size: int,
    max_seq_len: int = 2048,
) -> nn.Module | None:
    """Return an additive positional module, or ``None`` for non-additive kinds.

    ``sinusoidal`` returns :class:`SinusoidalPositionalEncoding`.
    ``rope`` returns ``None`` here — rotary application lives with attention (Phase 5).
    """
    kind = kind.lower()
    if kind not in POS_VARIANTS:
        raise ValueError(f"Unknown positional variant {kind!r}; expected one of {POS_VARIANTS}")
    if kind == POS_SINUSOIDAL:
        return SinusoidalPositionalEncoding(hidden_size, max_seq_len=max_seq_len)
    # RoPE does not use additive embeddings; factory / attention handle it later.
    return None
