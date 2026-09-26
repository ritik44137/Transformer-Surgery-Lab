"""Token lookup. Sinusoidal PE is added here; RoPE is not."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        positional: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.positional = positional  # None on the RoPE path
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (batch, seq), got {tuple(input_ids.shape)}")
        x = self.token(input_ids)
        if self.positional is not None:
            x = self.positional(x)
        return self.dropout(x)
