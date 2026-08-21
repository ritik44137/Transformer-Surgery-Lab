"""Write run artifacts: config, metadata, metrics, summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tsl.constants import (
    CONFIG_RESOLVED_FILENAME,
    METADATA_FILENAME,
    METRICS_EVAL_FILENAME,
    METRICS_TRAIN_FILENAME,
    SUMMARY_FILENAME,
)
from tsl.utils.io import append_jsonl, ensure_dir, write_json, write_yaml
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RunWriter:
    """Owns one run directory and appends metrics as JSONL."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.run_name = run_name
        self.run_dir = ensure_dir(Path(output_dir) / run_name)
        self.train_path = self.run_dir / METRICS_TRAIN_FILENAME
        self.eval_path = self.run_dir / METRICS_EVAL_FILENAME
        logger.info("Run directory: %s", self.run_dir)

    def save_config(self, cfg: Mapping[str, Any]) -> Path:
        path = self.run_dir / CONFIG_RESOLVED_FILENAME
        write_yaml(dict(cfg), path)
        return path

    def save_metadata(self, metadata: Mapping[str, Any]) -> Path:
        path = self.run_dir / METADATA_FILENAME
        write_json(dict(metadata), path)
        return path

    def log_train(self, row: Mapping[str, Any]) -> None:
        append_jsonl(row, self.train_path)

    def log_eval(self, row: Mapping[str, Any]) -> None:
        append_jsonl(row, self.eval_path)

    def save_summary(self, summary: Mapping[str, Any]) -> Path:
        path = self.run_dir / SUMMARY_FILENAME
        write_json(dict(summary), path)
        logger.info("Wrote summary to %s", path)
        return path
