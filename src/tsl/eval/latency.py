"""Forward and generation latency benchmarks."""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
import torch.nn as nn

from tsl.eval.generation import generate


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_ms(samples_s: list[float]) -> dict[str, float]:
    samples_ms = [s * 1000.0 for s in samples_s]
    return {
        "mean_ms": float(statistics.fmean(samples_ms)),
        "median_ms": float(statistics.median(samples_ms)),
        "stdev_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


@torch.no_grad()
def measure_forward_latency(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iters: int = 20,
) -> dict[str, Any]:
    """Time a single forward pass; report mean/median latency in ms."""
    model.eval()
    model.to(device)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(num_warmup):
        _ = model(ids)
    _sync(device)

    samples: list[float] = []
    for _ in range(num_iters):
        _sync(device)
        t0 = time.perf_counter()
        _ = model(ids)
        _sync(device)
        samples.append(time.perf_counter() - t0)

    stats = _summarize_ms(samples)
    return {
        "kind": "forward",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iters": num_iters,
        "forward_latency_ms": stats["mean_ms"],
        **stats,
    }


@torch.no_grad()
def measure_generation_latency(
    model: nn.Module,
    *,
    prompt_len: int,
    max_new_tokens: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 2,
    num_iters: int = 10,
) -> dict[str, Any]:
    """Time short autoregressive generation; report mean latency in ms."""
    model.eval()
    model.to(device)
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    for _ in range(num_warmup):
        _ = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    _sync(device)

    samples: list[float] = []
    for _ in range(num_iters):
        _sync(device)
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        _sync(device)
        samples.append(time.perf_counter() - t0)

    stats = _summarize_ms(samples)
    return {
        "kind": "generation",
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "num_iters": num_iters,
        "gen_latency_ms": stats["mean_ms"],
        "ms_per_token": stats["mean_ms"] / max(max_new_tokens, 1),
        **stats,
    }
