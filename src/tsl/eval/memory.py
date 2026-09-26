"""CUDA peak memory. On CPU the number is just missing."""

from __future__ import annotations

from typing import Any, Callable

import torch


def reset_peak_memory(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def peak_memory_mb(device: torch.device | None = None) -> float | None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def measure_peak_memory(
    fn: Callable[[], Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        result = fn()
        return {
            "peak_memory_mb": None,
            "device": str(device),
            "available": False,
            "note": "Peak memory reporting requires CUDA",
            "result": result,
        }

    reset_peak_memory(device)
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "peak_memory_mb": peak_memory_mb(device),
        "device": str(device),
        "available": True,
        "result": result,
    }
