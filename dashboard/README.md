# Dashboard

Streamlit app for comparing **Transformer Surgery Lab** experiment runs.

Styled after the dark NeuralAI-style reference in `references/Dashboard.jpg`:
navy background, purple accents, rounded metric cards, loss curves, and run tiles.

## Purpose

Load artifacts from `runs/` and make architecture tradeoffs obvious in under a minute:

- Overview metric cards (best val loss, params, throughput, run count)
- Train / validation loss curves
- Benchmark bars (throughput, latency, params)
- Per-run variant tiles (norm / pos / ff / attention)

## Launch

From the repo root:

```bash
make dashboard
# or
docker compose -f docker/docker-compose.yml --profile dashboard up dashboard
```

Open http://localhost:8501

Without Docker (after `pip install -r requirements.txt`):

```bash
PYTHONPATH=src streamlit run dashboard/app.py
```

## Expected inputs

Each run directory under `runs/<run_name>/` should contain:

| File | Required |
|------|----------|
| `config_resolved.yaml` | yes |
| `metadata.json` | yes |
| `metrics_train.jsonl` | yes |
| `metrics_eval.jsonl` | recommended |
| `summary.json` | yes |
| `benchmark.json` | recommended for efficiency plots |
| `samples.json` | optional |

Produce benchmarks with:

```bash
make benchmark RUN_DIR=runs/<run_name>
```

## Views

1. **Sidebar** — brand, run multi-select, navigation hints
2. **Overview** — four KPI cards from selected runs
3. **Training progress** — loss curves + mini stats
4. **Tradeoffs** — throughput / latency / params charts
5. **Runs** — card grid for each selected experiment

## Status

Phase 7 — production dashboard wired to real run artifacts.
