"""Learning-rate schedule: linear warmup + optional cosine decay."""

from __future__ import annotations

import math
from typing import Callable


def build_lr_scheduler(
    *,
    learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
) -> Callable[[int], float]:
    """Return a function ``step -> lr``.

    ``schedule``:
      - ``cosine``: linear warmup then cosine decay to ``min_lr_ratio * lr``
      - ``constant``: linear warmup then constant ``learning_rate``
    """
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))
    schedule = schedule.lower()
    min_lr = learning_rate * min_lr_ratio

    def lr_at(step: int) -> float:
        # step is 1-indexed from the trainer after each optimizer step.
        if step < 1:
            step = 1
        if warmup_steps > 0 and step <= warmup_steps:
            return learning_rate * step / float(warmup_steps)
        if schedule == "constant":
            return learning_rate
        if schedule != "cosine":
            raise ValueError(f"Unknown schedule {schedule!r}; use 'cosine' or 'constant'")

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine

    return lr_at


def set_optimizer_lr(optimizer, lr: float) -> None:
    """Write *lr* into every param group."""
    for group in optimizer.param_groups:
        group["lr"] = lr
