#!/usr/bin/env python3
"""Compare multiple TSL run directories into a compact table / JSON / CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.tracking.reader import comparison_row, load_runs
from tsl.utils.io import ensure_dir, write_json
from tsl.utils.logging_utils import get_logger

logger = get_logger("compare_runs")

COLUMNS = [
    "run_name",
    "norm",
    "positional",
    "feedforward",
    "attention",
    "best_val_loss",
    "best_perplexity",
    "final_train_loss",
    "param_count",
    "tokens_per_sec",
    "forward_latency_ms",
    "gen_latency_ms",
    "peak_memory_mb",
    "total_steps",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare TSL experiment runs.")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run directories under runs/",
    )
    p.add_argument(
        "--out-dir",
        default="experiments/comparisons",
        help="Directory for comparison.json / comparison.csv",
    )
    p.add_argument("--name", default="comparison", help="Output basename")
    return p.parse_args()


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> int:
    args = parse_args()
    bundles = load_runs(args.runs)
    rows = [comparison_row(b) for b in bundles]

    # Pretty CLI table
    widths = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=1)) for c in COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in COLUMNS))

    out_dir = ensure_dir(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    csv_path = out_dir / f"{args.name}.csv"

    write_json({"runs": rows, "columns": COLUMNS}, json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in COLUMNS})

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
