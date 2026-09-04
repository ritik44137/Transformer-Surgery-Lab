# Transformer Surgery Lab

A config-driven PyTorch lab for **controlled transformer architecture experiments**. Swap one component at a time — or compare full classical vs modern stacks — and measure the tradeoffs with reproducible runs and a Streamlit dashboard.

### Four comparison axes

| Axis | Baseline (classical) | Swap (modern) |
|------|---------------------|---------------|
| Normalization | LayerNorm | RMSNorm |
| Positional encoding | Sinusoidal | RoPE |
| Feed-forward | ReLU MLP | SwiGLU |
| Attention | Multi-head (MHA) | Grouped-query (GQA) |

Recommended experiment order: **baseline → single-axis swaps → combined modern stack**.

---

## Prerequisites

- **WSL2** (Ubuntu 22.04+) or Linux
- **Docker** with Compose (`docker compose` or `docker-compose`)
- Optional: **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for full training (`make train`)

All commands below are run from the **repo root**.

---

## 1. First steps after cloning

```bash
# Build images (CPU is enough for smoke / mini gate / tests / dashboard)
make build-cpu

# Optional: GPU image for full TinyStories training
make build
```

Sanity-check the plumbing (no dataset download):

```bash
make smoke    # toy train/eval + temp artifacts
make test     # full pytest suite
```

Open a shell inside the container when you need an interactive session:

```bash
make shell-cpu   # CPU image
make shell       # GPU image
```

---

## 2. Prepare data

**Fast local path (recommended first):** tiny smoke corpus, no HuggingFace download.

```bash
make prepare-data-smoke
```

**Official TinyStories path** (larger download / preprocess):

```bash
make prepare-data
```

Artifacts land under `data/processed/` and `data/tokenizer/`.

---

## 3. Train

### Mini gate (CPU, smoke data + tiny model)

Use this to verify the full train → artifact path quickly:

```bash
make train-mini
# → runs/baseline_mini_gate/
```

RMSNorm mini counterpart (for a two-run compare):

```bash
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/train.py --config configs/experiments/rmsnorm_mini_gate.yaml --device cpu
# → runs/rmsnorm_mini_gate/
```

### Full baseline (GPU image)

```bash
make train
# uses configs/experiments/baseline_layernorm_sinusoidal_relu_mha.yaml
# → runs/<run_name>/
```

### Train any experiment config

```bash
docker compose -f docker/docker-compose.yml run --rm dev \
  python scripts/train.py --config configs/experiments/<name>.yaml

# Force CPU:
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/train.py --config configs/experiments/<name>.yaml --device cpu
```

Each run directory typically contains: `config_resolved.yaml`, `metadata.json`, `metrics_*.jsonl`, `summary.json`, and `checkpoints/`.

---

## 4. Architecture swaps

Configs live in `configs/experiments/`. Primary grid (see also `configs/experiments/full_grid.yaml`):

| Config | What changes |
|--------|----------------|
| `baseline_layernorm_sinusoidal_relu_mha.yaml` | Classical reference |
| `rmsnorm_sinusoidal_relu_mha.yaml` | Norm only → RMSNorm |
| `layernorm_rope_relu_mha.yaml` | Positional only → RoPE |
| `layernorm_sinusoidal_swiglu_mha.yaml` | FF only → SwiGLU |
| `layernorm_sinusoidal_relu_gqa.yaml` | Attention only → GQA |
| `rmsnorm_rope_swiglu_gqa.yaml` | Full modern stack |

Example — train the RoPE single-axis swap:

```bash
docker compose -f docker/docker-compose.yml run --rm dev \
  python scripts/train.py --config configs/experiments/layernorm_rope_relu_mha.yaml
```

Quick component-level demo (no training):

```bash
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/demo_swaps.py
```

---

## 5. Evaluate a run

```bash
make evaluate RUN_DIR=runs/baseline_mini_gate
```

Or with options:

```bash
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/evaluate.py \
    --run-dir runs/baseline_mini_gate \
    --device cpu \
    --generate \
    --checkpoint best
```

Updates val metrics / samples under that run directory.

---

## 6. Benchmark a run

Throughput, latency, params, memory:

```bash
make benchmark RUN_DIR=runs/baseline_mini_gate
```

Writes `benchmark.json` into the run directory. Repeat for each run you want on the dashboard / comparison table.

---

## 7. Compare runs

```bash
make compare-runs RUNS='runs/baseline_mini_gate runs/rmsnorm_mini_gate'
```

Or:

```bash
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/compare_runs.py \
    --runs runs/baseline_mini_gate runs/rmsnorm_mini_gate \
    --name phase6_gate
```

Exports a compact table plus JSON/CSV under `experiments/comparisons/`.

Bundle all discovered runs for demos/archives:

```bash
make export-dashboard
# → experiments/comparisons/dashboard_export.json
```

---

## 8. Dashboard

Requires run artifacts under `runs/` (train + preferably `benchmark.json`):

```bash
make dashboard
# → http://localhost:8501
```

Select runs in the sidebar to compare loss curves, throughput/latency, and model profiles. Stop with Ctrl+C in that terminal (or `docker compose -f docker/docker-compose.yml --profile dashboard down`).

---

## 9. Tests

```bash
make test     # all of tests/
make smoke    # top-level plumbing check (toy tensors, no dataset)
```

Single file / single case:

```bash
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  pytest tests/test_norms.py -v

docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  pytest tests/test_attention.py::test_mha_output_shape -v
```

Without Docker (host Python 3.11+ with deps installed):

```bash
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src python scripts/smoke_test.py --config configs/train/smoke.yaml
```

---

## 10. Lint / format

```bash
make lint
make format
```

---

## Suggested end-to-end path (CPU)

```bash
make build-cpu
make smoke && make test
make prepare-data-smoke
make train-mini
docker compose -f docker/docker-compose.yml run --rm dev-cpu \
  python scripts/train.py --config configs/experiments/rmsnorm_mini_gate.yaml --device cpu
make benchmark RUN_DIR=runs/baseline_mini_gate
make benchmark RUN_DIR=runs/rmsnorm_mini_gate
make compare-runs RUNS='runs/baseline_mini_gate runs/rmsnorm_mini_gate'
make export-dashboard
make dashboard
```

---

## Results

Mini-gate numbers from `experiments/comparisons/phase6_gate.json` (smoke corpus, tiny model, 20 steps — for plumbing, not paper claims). Fill the primary grid after full TinyStories runs.

| Run | Val loss | Params | Throughput | Fwd latency | Notes |
|-----|----------|--------|------------|-------------|-------|
| `baseline_mini_gate` | 44.80 | 98,560 | 182.7K tok/s | 1.82 ms | Classical stack (mini) |
| `rmsnorm_mini_gate` | 41.11 | 98,240 | 195.0K tok/s | 2.06 ms | RMSNorm only (mini) |
| Full baseline / RoPE / SwiGLU / GQA / modern | — | — | — | — | Train via configs in §4 |

---

## License

MIT — see [`LICENSE`](LICENSE).
