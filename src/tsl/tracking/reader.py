"""Read experiment run artifacts for comparison tools and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tsl.constants import (
    BENCHMARK_FILENAME,
    CONFIG_RESOLVED_FILENAME,
    METADATA_FILENAME,
    METRICS_EVAL_FILENAME,
    METRICS_TRAIN_FILENAME,
    SAMPLES_FILENAME,
    SUMMARY_FILENAME,
)
from tsl.utils.io import read_json, read_jsonl, read_yaml


class RunReader:
    """Load a single run directory into normalized structures."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def _optional_json(self, name: str) -> dict[str, Any] | None:
        path = self.run_dir / name
        if not path.is_file():
            return None
        data = read_json(path)
        return data if isinstance(data, dict) else {"value": data}

    def _optional_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        if not path.is_file():
            return []
        return list(read_jsonl(path))

    def load_config(self) -> dict[str, Any] | None:
        path = self.run_dir / CONFIG_RESOLVED_FILENAME
        if not path.is_file():
            return None
        data = read_yaml(path)
        return data if isinstance(data, dict) else None

    def load_metadata(self) -> dict[str, Any] | None:
        return self._optional_json(METADATA_FILENAME)

    def load_summary(self) -> dict[str, Any] | None:
        return self._optional_json(SUMMARY_FILENAME)

    def load_benchmark(self) -> dict[str, Any] | None:
        return self._optional_json(BENCHMARK_FILENAME)

    def load_samples(self) -> dict[str, Any] | list[Any] | None:
        path = self.run_dir / SAMPLES_FILENAME
        if not path.is_file():
            return None
        return read_json(path)

    def load_train_metrics(self) -> list[dict[str, Any]]:
        return self._optional_jsonl(METRICS_TRAIN_FILENAME)

    def load_eval_metrics(self) -> list[dict[str, Any]]:
        return self._optional_jsonl(METRICS_EVAL_FILENAME)

    def load_all(self) -> dict[str, Any]:
        """Return a dashboard-friendly bundle for this run."""
        meta = self.load_metadata() or {}
        summary = self.load_summary() or {}
        benchmark = self.load_benchmark() or {}
        cfg = self.load_config() or {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

        variants = (
            summary.get("model_variants")
            or meta.get("model_variants")
            or {
                "norm": model_cfg.get("norm"),
                "positional": model_cfg.get("positional"),
                "feedforward": model_cfg.get("feedforward"),
                "attention": model_cfg.get("attention"),
            }
        )

        return {
            "run_dir": str(self.run_dir),
            "run_name": summary.get("run_name")
            or meta.get("run_name")
            or self.run_dir.name,
            "metadata": meta,
            "summary": summary,
            "benchmark": benchmark,
            "config": cfg,
            "model_variants": variants,
            "train_metrics": self.load_train_metrics(),
            "eval_metrics": self.load_eval_metrics(),
            "samples": self.load_samples(),
        }


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Convenience wrapper around :class:`RunReader.load_all`."""
    return RunReader(run_dir).load_all()


def load_runs(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """Load multiple run directories."""
    return [load_run(d) for d in run_dirs]


def comparison_row(bundle: dict[str, Any]) -> dict[str, Any]:
    """Flatten one run bundle into a compact comparison-table row."""
    summary = bundle.get("summary") or {}
    benchmark = bundle.get("benchmark") or {}
    meta = bundle.get("metadata") or {}
    variants = bundle.get("model_variants") or {}

    return {
        "run_name": bundle.get("run_name"),
        "run_dir": bundle.get("run_dir"),
        "norm": variants.get("norm"),
        "positional": variants.get("positional"),
        "feedforward": variants.get("feedforward"),
        "attention": variants.get("attention"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_perplexity": summary.get("best_perplexity"),
        "final_train_loss": summary.get("final_train_loss"),
        "total_steps": summary.get("total_steps"),
        "param_count": summary.get("param_count")
        or benchmark.get("param_count")
        or meta.get("param_count"),
        "tokens_per_sec": benchmark.get("tokens_per_sec"),
        "forward_latency_ms": benchmark.get("forward_latency_ms"),
        "gen_latency_ms": benchmark.get("gen_latency_ms"),
        "peak_memory_mb": benchmark.get("peak_memory_mb"),
    }
