"""Decoder-only transformer language model (nanoGPT-like, modular)."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsl.model.block import TransformerBlock
from tsl.model.embeddings import TokenEmbeddings


class DecoderLM(nn.Module):
    """Small causal language model.

    Forward returns logits of shape ``(batch, seq, vocab_size)``.
    """

    def __init__(
        self,
        embeddings: TokenEmbeddings,
        blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        *,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.blocks = blocks
        self.final_norm = final_norm
        self.lm_head = lm_head

        if tie_weights:
            # Share input embedding and output projection weights.
            self.lm_head.weight = self.embeddings.token.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for *input_ids* ``(B, T)``."""
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def count_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        # Account for tied weights once.
        seen: set[int] = set()
        total = 0
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
