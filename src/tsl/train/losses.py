"""Causal language modeling loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over next-token predictions.

    Parameters
    ----------
    logits:
        Model output of shape ``(batch, seq, vocab)``.
    labels:
        Target token ids of shape ``(batch, seq)``, already shifted so that
        ``labels[t]`` is the token that should be predicted from position ``t``.
        The dataset produces this alignment; this function does **not** shift.

    Returns
    -------
    Scalar mean cross-entropy loss.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, T, V), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (B, T), got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}"
        )

    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))
