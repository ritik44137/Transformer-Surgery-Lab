"""Load experiment run artifacts into dashboard-friendly structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tsl.tracking.reader import comparison_row, load_run, load_runs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_runs_dir() -> Path:
    return repo_root() / "runs"


def discover_runs(runs_dir: str | Path | None = None) -> list[Path]:
    """Return run directories that look like completed experiments."""
    root = Path(runs_dir) if runs_dir else default_runs_dir()
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / "summary.json").is_file() or (path / "config_resolved.yaml").is_file():
            runs.append(path)
    return runs


def load_dashboard_runs(
    run_dirs: list[str | Path] | None = None,
    *,
    runs_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load selected runs (or all discovered runs) as full bundles."""
    if run_dirs is None:
        run_dirs = discover_runs(runs_dir)
    if not run_dirs:
        return []
    bundles = load_runs(run_dirs)
    for b in bundles:
        b["comparison"] = comparison_row(b)
    return bundles


def loss_series(bundle: dict[str, Any]) -> dict[str, list[float]]:
    """Extract train/eval loss series for plotting."""
    train = bundle.get("train_metrics") or []
    eval_rows = bundle.get("eval_metrics") or []
    return {
        "train_steps": [float(r.get("step", i)) for i, r in enumerate(train, start=1)],
        "train_loss": [float(r["loss"]) for r in train if "loss" in r],
        "eval_steps": [float(r.get("step", i)) for i, r in enumerate(eval_rows, start=1)],
        "eval_loss": [float(r["val_loss"]) for r in eval_rows if "val_loss" in r],
    }


def overview_stats(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate KPIs across selected runs for the top metric row."""
    if not bundles:
        return {
            "n_runs": 0,
            "best_val_loss": None,
            "best_run": None,
            "total_params": None,
            "best_tok_s": None,
            "best_tok_run": None,
        }

    best_loss = None
    best_run = None
    best_tok = None
    best_tok_run = None
    params = []

    for b in bundles:
        name = b.get("run_name")
        summary = b.get("summary") or {}
        bench = b.get("benchmark") or {}
        loss = summary.get("best_val_loss")
        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = float(loss)
            best_run = name
        p = summary.get("param_count") or bench.get("param_count")
        if p is not None:
            params.append(int(p))
        tok = bench.get("tokens_per_sec")
        if tok is not None and (best_tok is None or tok > best_tok):
            best_tok = float(tok)
            best_tok_run = name

    return {
        "n_runs": len(bundles),
        "best_val_loss": best_loss,
        "best_run": best_run,
        "total_params": max(params) if params else None,
        "mean_params": sum(params) / len(params) if params else None,
        "best_tok_s": best_tok,
        "best_tok_run": best_tok_run,
    }
