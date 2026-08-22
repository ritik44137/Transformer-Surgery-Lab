"""Feed-forward / MLP blocks.

Swap axis: ReLU MLP (baseline) vs SwiGLU (Phase 5).

Parameterization policy (fair comparison)
-----------------------------------------
A ReLU MLP has two projections: ``hidden → d_ff → hidden`` (~ ``2 * H * d_ff`` params).
SwiGLU has three: gate, up, and down (~ ``3 * H * d_ff_eff`` params).

By default ``scale_for_param_parity=True`` sets::

    d_ff_eff = round(2/3 * d_ff)

so SwiGLU stays roughly parameter-matched to the ReLU baseline using the same
config ``d_ff``. Set ``scale_for_param_parity=False`` to keep ``d_ff`` fixed and
accept a larger SwiGLU (then report param counts explicitly in summaries).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.constants import FF_RELU, FF_SWIGLU, FF_VARIANTS


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


class SwiGLUFeedForward(nn.Module):
    """SwiGLU MLP: ``silu(gate(x)) * up(x)``, then down-project.

    Expected input shape: ``(batch, seq, hidden_size)``
    Output shape: same as input.
    """

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
        # See module docstring for the 2/3 scaling rationale.
        if scale_for_param_parity:
            d_ff_eff = max(hidden_size, int(round(2 * d_ff / 3)))
            # Keep even for friendlier kernel shapes.
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
    """Construct a feed-forward block by config name."""
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
