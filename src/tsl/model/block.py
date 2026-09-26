"""Pre-norm block. The factory passes in the norm, attention, and MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
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
        x = x + self.attention(self.norm_attn(x))
        x = x + self.feedforward(self.norm_ff(x))
        return x
