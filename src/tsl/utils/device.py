"""Device selection and runtime info helpers."""

from __future__ import annotations

from typing import Any

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return ``cuda`` if available and preferred, otherwise ``cpu``.

    Defaults to using CUDA when present so training scripts do not need
    special-casing. Pass ``prefer_cuda=False`` to force CPU (e.g. smoke tests).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device: torch.device | None = None) -> dict[str, Any]:
    """Return a small, JSON-friendly snapshot of runtime device state."""
    if device is None:
        device = get_device()

    info: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_index": idx,
                "gpu_name": props.name,
                "gpu_total_memory_mb": round(props.total_memory / (1024**2), 1),
            }
        )

    return info
