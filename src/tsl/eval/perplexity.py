"""Perplexity from mean cross-entropy (nats)."""

from __future__ import annotations

import math


def loss_to_perplexity(mean_nll: float, *, max_nll: float = 20.0) -> float:
    """Convert average negative log-likelihood (nats) to perplexity.

    ``perplexity = exp(mean_nll)``. Cap ``mean_nll`` at *max_nll* so early
    unstable losses do not overflow to ``inf``.
    """
    return float(math.exp(min(float(mean_nll), max_nll)))
