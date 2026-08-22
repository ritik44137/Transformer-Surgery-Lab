"""Tokens-per-second throughput measurement."""

from __future__ import annotations

import time
from typing import Any, Literal

import torch
import torch.nn as nn


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inference_throughput(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> dict[str, Any]:
    """Measure forward-pass tokens/sec on random token batches."""
    model.eval()
    model.to(device)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(num_warmup):
        _ = model(ids)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = model(ids)
    _sync(device)
    elapsed = max(time.perf_counter() - t0, 1e-8)

    tokens = batch_size * seq_len * num_iters
    return {
        "mode": "inference",
        "tokens_per_sec": tokens / elapsed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iters": num_iters,
        "elapsed_s": elapsed,
    }


def measure_train_throughput(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> dict[str, Any]:
    """Measure train-step tokens/sec (forward + backward + optimizer step)."""
    from tsl.train.losses import causal_lm_loss

    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one_step() -> None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)
        loss.backward()
        optimizer.step()

    for _ in range(num_warmup):
        one_step()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        one_step()
    _sync(device)
    elapsed = max(time.perf_counter() - t0, 1e-8)

    tokens = batch_size * seq_len * num_iters
    return {
        "mode": "train",
        "tokens_per_sec": tokens / elapsed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iters": num_iters,
        "elapsed_s": elapsed,
    }


def measure_throughput(
    model: nn.Module,
    *,
    mode: Literal["inference", "train"] = "inference",
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> dict[str, Any]:
    """Dispatch to inference or train throughput measurement."""
    if mode == "inference":
        return measure_inference_throughput(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )
    if mode == "train":
        return measure_train_throughput(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            num_warmup=max(1, num_warmup - 1),
            num_iters=max(1, num_iters // 2),
        )
    raise ValueError(f"Unknown throughput mode {mode!r}")
