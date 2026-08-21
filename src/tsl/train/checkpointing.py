"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tsl.constants import CHECKPOINT_BEST, CHECKPOINT_LATEST
from tsl.utils.io import ensure_dir
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


def checkpoint_paths(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "latest": run_dir / CHECKPOINT_LATEST,
        "best": run_dir / CHECKPOINT_BEST,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int,
    best_val_loss: float | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint dict to *path*."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    logger.info("Saved checkpoint step=%s -> %s", step, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint; optionally restore model/optimizer in place."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    logger.info("Loaded checkpoint from %s (step=%s)", path, payload.get("step"))
    return payload
