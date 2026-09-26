"""LayerNorm and RMSNorm, both over the last dimension."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsl.constants import NORM_LAYERNORM, NORM_RMSNORM, NORM_VARIANTS


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


class RMSNorm(nn.Module):
    # no mean subtraction, no bias
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms in fp16 gets noisy; compute it in fp32
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x_f / rms).type_as(x)
        return self.weight * x_norm


def build_norm(kind: str, hidden_size: int, eps: float = 1e-5) -> nn.Module:
    kind = kind.lower()
    if kind not in NORM_VARIANTS:
        raise ValueError(f"Unknown norm variant {kind!r}; expected one of {NORM_VARIANTS}")
    if kind == NORM_LAYERNORM:
        return LayerNorm(hidden_size, eps=eps)
    if kind == NORM_RMSNORM:
        return RMSNorm(hidden_size, eps=eps)
    raise ValueError(f"Unhandled norm variant {kind!r}")
