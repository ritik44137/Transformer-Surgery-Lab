"""Deterministic seed helpers for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Integer seed shared across libraries.
    deterministic:
        If True, prefer deterministic CUDA algorithms when available.
        This can reduce throughput slightly but improves repeatability.

    Returns
    -------
    The seed that was applied.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make hash-based Python behavior deterministic where possible.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Prefer deterministic algorithms when supported; fall back silently.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch without warn_only.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    return seed
