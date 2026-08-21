"""Canonical artifact shapes for experiment tracking.

These helpers build plain dicts (JSON/JSONL friendly) rather than heavy
schema frameworks — easy to explain and easy for the dashboard to consume.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def make_metadata(
    *,
    run_name: str,
    seed: int,
    device: str,
    param_count: int,
    trainable_params: int,
    model_variants: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run-level metadata written once at start."""
    meta: dict[str, Any] = {
        "run_name": run_name,
        "seed": seed,
        "device": device,
        "param_count": param_count,
        "trainable_params": trainable_params,
        "model_variants": dict(model_variants),
    }
    if extra:
        meta.update(dict(extra))
    return meta


def make_train_metrics(
    *,
    step: int,
    loss: float,
    lr: float,
    tokens_per_sec: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One train JSONL row."""
    row: dict[str, Any] = {
        "step": int(step),
        "loss": float(loss),
        "lr": float(lr),
    }
    if tokens_per_sec is not None:
        row["tokens_per_sec"] = float(tokens_per_sec)
    if extra:
        row.update(dict(extra))
    return row


def make_eval_metrics(
    *,
    step: int,
    val_loss: float,
    perplexity: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One eval JSONL row."""
    row: dict[str, Any] = {
        "step": int(step),
        "val_loss": float(val_loss),
    }
    if perplexity is not None:
        row["perplexity"] = float(perplexity)
    if extra:
        row.update(dict(extra))
    return row


def make_summary(
    *,
    run_name: str,
    best_val_loss: float | None,
    final_train_loss: float | None,
    best_step: int | None,
    total_steps: int,
    param_count: int,
    model_variants: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact end-of-run summary for comparison tooling."""
    summary: dict[str, Any] = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "best_step": best_step,
        "total_steps": int(total_steps),
        "param_count": int(param_count),
        "model_variants": dict(model_variants),
    }
    if best_val_loss is not None:
        summary["best_perplexity"] = float(math.exp(min(best_val_loss, 20.0)))
    if extra:
        summary.update(dict(extra))
    return summary


def make_benchmark(
    *,
    param_count: int,
    tokens_per_sec: float | None = None,
    forward_latency_ms: float | None = None,
    gen_latency_ms: float | None = None,
    peak_memory_mb: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Benchmark artifact shape (filled in Phase 6)."""
    out: dict[str, Any] = {"param_count": int(param_count)}
    if tokens_per_sec is not None:
        out["tokens_per_sec"] = float(tokens_per_sec)
    if forward_latency_ms is not None:
        out["forward_latency_ms"] = float(forward_latency_ms)
    if gen_latency_ms is not None:
        out["gen_latency_ms"] = float(gen_latency_ms)
    if peak_memory_mb is not None:
        out["peak_memory_mb"] = float(peak_memory_mb)
    if extra:
        out.update(dict(extra))
    return out
