# Dashboard

Streamlit app for comparing Transformer Surgery Lab experiment runs.

## Purpose

Load run artifacts from `runs/` and present:

- Overview and run selection
- Loss curves (train / validation)
- Benchmark tradeoffs (throughput, latency, memory, parameter count)
- Side-by-side comparison tables

## Launch

From repo root (once `dashboard/app.py` is implemented):

```bash
make dashboard
# or
docker compose -f docker/docker-compose.yml --profile dashboard up dashboard
```

Then open http://localhost:8501

## Expected inputs

The dashboard reads standardized artifacts produced by training and benchmark scripts:

- `runs/<run_id>/config_resolved.yaml`
- `runs/<run_id>/metadata.json`
- `runs/<run_id>/metrics_train.jsonl`
- `runs/<run_id>/metrics_eval.jsonl`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/benchmark.json` (optional)

See `src/tsl/tracking/schema.py` (Phase 4) for the canonical schema.

## Status

Scaffold only — implement `dashboard/app.py` in Phase 7 per the implementation plan.
