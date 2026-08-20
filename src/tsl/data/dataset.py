"""Causal language modeling dataset over a flat token array."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    """Fixed-length next-token prediction windows.

    Given a 1D token array ``T``, sample ``i`` returns::

        input  = T[i * seq_len : i * seq_len + seq_len]
        target = T[i * seq_len + 1 : i * seq_len + seq_len + 1]

    Windows are non-overlapping for simple, efficient packing.
    """

    def __init__(self, tokens: np.ndarray | torch.Tensor, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        if isinstance(tokens, np.ndarray):
            tokens_t = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
        else:
            tokens_t = tokens.to(dtype=torch.long)

        if tokens_t.ndim != 1:
            raise ValueError(f"tokens must be 1D, got shape {tuple(tokens_t.shape)}")
        if len(tokens_t) < seq_len + 1:
            raise ValueError(
                f"Need at least seq_len+1={seq_len + 1} tokens, got {len(tokens_t)}"
            )

        self.tokens = tokens_t
        self.seq_len = seq_len
        self.n_windows = (len(tokens_t) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.n_windows:
            raise IndexError(index)
        start = index * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1].clone(),
            "labels": chunk[1:].clone(),
        }


def load_causal_dataset(path: str | Path, seq_len: int) -> CausalLMDataset:
    """Load a ``.npy`` token file into a :class:`CausalLMDataset`."""
    tokens = np.load(path)
    return CausalLMDataset(tokens, seq_len=seq_len)
