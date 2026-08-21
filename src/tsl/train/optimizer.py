"""Optimizer factory (AdamW with light param grouping)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build AdamW with decay applied only to 2D+ weight tensors.

    Biases and LayerNorm/Embedding-style 1D parameters skip weight decay —
    a common, explainable grouping used by nanoGPT-style trainers.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps)


def param_group_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return a small summary of optimizer param groups (for logging)."""
    out: list[dict[str, Any]] = []
    for i, group in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in group["params"])
        out.append(
            {
                "group": i,
                "n_params": n,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
            }
        )
    return out
