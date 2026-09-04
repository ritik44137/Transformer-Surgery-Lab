# Comparison artifacts

Derived tables and bundles produced from run directories under `runs/<name>/`.
**Source of truth** is always the individual run (config, metrics JSONL, summary, benchmark). Files here are convenience exports for CLI review, archiving, and demos.

## Generate

Compare selected runs (JSON + CSV):

```bash
make compare-runs RUNS='runs/baseline_mini_gate runs/rmsnorm_mini_gate'
# or
python scripts/compare_runs.py \
  --runs runs/baseline_mini_gate runs/rmsnorm_mini_gate \
  --out-dir experiments/comparisons \
  --name phase6_gate
```

Dashboard-friendly aggregate (curves + comparison fields):

```bash
python scripts/export_dashboard_data.py
# optional: --runs runs/a runs/b --out experiments/comparisons/dashboard_export.json
```

## Files in this directory

| File | Role |
|------|------|
| `<name>.json` | Compact comparison rows from `compare_runs.py` |
| `<name>.csv` | Same rows for spreadsheets |
| `dashboard_export.json` | Optional multi-run bundle with train/eval curves (`export_dashboard_data.py`) |

Example already present: `phase6_gate.json` / `phase6_gate.csv` (mini-gate baseline vs RMSNorm).

## Not source of truth

Do not edit these exports by hand. Re-run the scripts after training, evaluating, or benchmarking so they stay aligned with `runs/`.
