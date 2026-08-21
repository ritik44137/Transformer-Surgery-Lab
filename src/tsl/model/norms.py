"""Normalization layers for Transformer Surgery Lab.

Baseline: LayerNorm. RMSNorm is added in Phase 5 beside this implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsl.constants import NORM_LAYERNORM, NORM_VARIANTS


class LayerNorm(nn.Module):
    """Standard LayerNorm over the last dimension.

    Expected input shape: ``(..., hidden_size)``
    Output shape: same as input.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual implementation keeps the math visible for interviews.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


def build_norm(kind: str, hidden_size: int, eps: float = 1e-5) -> nn.Module:
    """Construct a normalization module by config name.

    Phase 3 supports ``layernorm`` only. ``rmsnorm`` lands in Phase 5.
    """
    kind = kind.lower()
    if kind not in NORM_VARIANTS:
        raise ValueError(f"Unknown norm variant {kind!r}; expected one of {NORM_VARIANTS}")
    if kind == NORM_LAYERNORM:
        return LayerNorm(hidden_size, eps=eps)
    raise NotImplementedError(f"Norm variant {kind!r} is not implemented yet (Phase 5).")
