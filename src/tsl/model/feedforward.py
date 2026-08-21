"""Feed-forward / MLP blocks.

Baseline: ReLU MLP. SwiGLU is added in Phase 5 beside this implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.constants import FF_RELU, FF_VARIANTS


class ReLUFeedForward(nn.Module):
    """Two-layer MLP with ReLU: ``hidden -> d_ff -> hidden``.

    Expected input shape: ``(batch, seq, hidden_size)``
    Output shape: same as input.
    """

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


def build_feedforward(
    kind: str,
    hidden_size: int,
    d_ff: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct a feed-forward block by config name.

    Phase 3 supports ``relu`` only. ``swiglu`` lands in Phase 5.
    """
    kind = kind.lower()
    if kind not in FF_VARIANTS:
        raise ValueError(f"Unknown feedforward variant {kind!r}; expected one of {FF_VARIANTS}")
    if kind == FF_RELU:
        return ReLUFeedForward(hidden_size, d_ff=d_ff, dropout=dropout)
    raise NotImplementedError(f"Feed-forward variant {kind!r} is not implemented yet (Phase 5).")
