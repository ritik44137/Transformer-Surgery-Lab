#!/usr/bin/env python3
"""Aggregate run artifacts into a compact dashboard-friendly export bundle.

Reuses :mod:`tsl.tracking.reader` so the export stays aligned with the
dashboard and ``compare_runs`` without duplicating load logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.tracking.reader import comparison_row, load_runs
from tsl.utils.io import ensure_dir, write_json
from tsl.utils.logging_utils import get_logger

logger = get_logger("export_dashboard_data")


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Return run directories that look complete enough to export."""
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / "summary.json").is_file() or (path / "config_resolved.yaml").is_file():
            found.append(path)
    return found


def build_bundle(bundles: list[dict]) -> dict:
    """Compact export: comparison rows + per-run metric series."""
    runs_out = []
    for b in bundles:
        row = comparison_row(b)
        train = b.get("train_metrics") or []
        eval_rows = b.get("eval_metrics") or []
        runs_out.append(
            {
                **row,
                "model_variants": b.get("model_variants") or {},
                "train_curve": [
                    {"step": r.get("step"), "loss": r.get("loss")}
                    for r in train
                    if "loss" in r
                ],
                "eval_curve": [
                    {"step": r.get("step"), "val_loss": r.get("val_loss")}
                    for r in eval_rows
                    if "val_loss" in r
                ],
                "benchmark": b.get("benchmark") or {},
                "summary": b.get("summary") or {},
            }
        )
    return {
        "n_runs": len(runs_out),
        "runs": runs_out,
        "columns": list(comparison_row(bundles[0]).keys()) if bundles else [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a compact dashboard data bundle from run artifacts."
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run directories (default: discover under --runs-root)",
    )
    p.add_argument(
        "--runs-root",
        default="runs",
        help="Root used when --runs is omitted (default: runs/)",
    )
    p.add_argument(
        "--out",
        default="experiments/comparisons/dashboard_export.json",
        help="Output JSON path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs:
        run_dirs = [Path(p) for p in args.runs]
    else:
        run_dirs = discover_run_dirs(Path(args.runs_root))

    if not run_dirs:
        logger.error("No runs found to export.")
        return 1

    bundles = load_runs(run_dirs)
    payload = build_bundle(bundles)
    out = Path(args.out)
    ensure_dir(out.parent)
    write_json(payload, out)
    logger.info("Exported %d run(s) → %s", payload["n_runs"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
