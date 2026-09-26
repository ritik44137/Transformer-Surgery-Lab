# Comparison artifacts

Tables copied out of `runs/<name>/` for the CLI and for keeping a snapshot around. If these disagree with the run directory, trust the run directory.

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

One JSON file with comparison fields and the loss curves:

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

Don't edit the JSON or CSV by hand. Re-run the scripts after you train or benchmark.
