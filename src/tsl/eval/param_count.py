"""Parameter counting helpers (dashboard-friendly)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Return total / trainable parameter counts, de-duplicating tied weights."""
    seen: set[int] = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return {
        "param_count": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(total - trainable),
    }
