"""ReLU MLP and SwiGLU.

SwiGLU has a gate, an up projection, and a down projection, so three
matrices against the ReLU block's two. With scale_for_param_parity (the
default) the inner width is round(2/3 * d_ff), bumped to even, so the
param count stays near the ReLU MLP that used the same d_ff. Turn the
flag off to keep d_ff exactly and live with the extra parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.constants import FF_RELU, FF_SWIGLU, FF_VARIANTS


class ReLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.d_ff = d_ff
        self.fc_up = nn.Linear(hidden_size, d_ff)
        self.fc_down = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_up(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_down(x)
        x = self.dropout(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_ff: int,
        dropout: float = 0.0,
        *,
        scale_for_param_parity: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if scale_for_param_parity:
            d_ff_eff = max(hidden_size, int(round(2 * d_ff / 3)))
            # even width; some kernels dislike odd sizes
            d_ff_eff = d_ff_eff + (d_ff_eff % 2)
        else:
            d_ff_eff = d_ff
        self.d_ff = d_ff_eff
        self.scale_for_param_parity = scale_for_param_parity

        self.fc_gate = nn.Linear(hidden_size, d_ff_eff, bias=False)
        self.fc_up = nn.Linear(hidden_size, d_ff_eff, bias=False)
        self.fc_down = nn.Linear(d_ff_eff, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.fc_gate(x)) * self.fc_up(x)
        gated = self.dropout(gated)
        return self.dropout(self.fc_down(gated))


def build_feedforward(
    kind: str,
    hidden_size: int,
    d_ff: int,
    dropout: float = 0.0,
    *,
    scale_for_param_parity: bool = True,
) -> nn.Module:
    kind = kind.lower()
    if kind not in FF_VARIANTS:
        raise ValueError(f"Unknown feedforward variant {kind!r}; expected one of {FF_VARIANTS}")
    if kind == FF_RELU:
        return ReLUFeedForward(hidden_size, d_ff=d_ff, dropout=dropout)
    if kind == FF_SWIGLU:
        return SwiGLUFeedForward(
            hidden_size,
            d_ff=d_ff,
            dropout=dropout,
            scale_for_param_parity=scale_for_param_parity,
        )
    raise ValueError(f"Unhandled feedforward variant {kind!r}")
