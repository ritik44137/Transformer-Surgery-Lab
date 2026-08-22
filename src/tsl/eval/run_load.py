"""Shared helpers to rebuild a trained model from a run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tsl.constants import CHECKPOINT_BEST, CHECKPOINT_LATEST, CONFIG_RESOLVED_FILENAME
from tsl.model import build_model
from tsl.train.checkpointing import load_checkpoint
from tsl.utils.device import get_device
from tsl.utils.io import read_yaml
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_run_model(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    device: torch.device | None = None,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, dict[str, Any], Path]:
    """Load resolved config + model weights from a run directory.

    Returns ``(model, cfg, checkpoint_path)``.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / CONFIG_RESOLVED_FILENAME
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {cfg_path}")
    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config at {cfg_path}")

    if device is None:
        device = get_device(prefer_cuda=prefer_cuda)

    ckpt_name = CHECKPOINT_BEST if checkpoint == "best" else CHECKPOINT_LATEST
    if checkpoint not in ("best", "latest"):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            ckpt_path = run_dir / checkpoint
    else:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.is_file() and checkpoint == "best":
            # Fall back to latest if best was never written.
            alt = run_dir / CHECKPOINT_LATEST
            if alt.is_file():
                logger.warning("best checkpoint missing; using latest")
                ckpt_path = alt

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg)
    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s on %s", ckpt_path, device)
    return model, cfg, ckpt_path
