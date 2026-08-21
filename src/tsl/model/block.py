"""Pre-norm transformer block with injected submodules."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: Norm → Attn → +res, then Norm → FFN → +res.

    Submodules are injected so architecture swaps never require editing this file.
    """

    def __init__(
        self,
        norm_attn: nn.Module,
        attention: nn.Module,
        norm_ff: nn.Module,
        feedforward: nn.Module,
    ) -> None:
        super().__init__()
        self.norm_attn = norm_attn
        self.attention = attention
        self.norm_ff = norm_ff
        self.feedforward = feedforward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual
        x = x + self.attention(self.norm_attn(x))
        # Feed-forward sublayer with residual
        x = x + self.feedforward(self.norm_ff(x))
        return x
