# Transformer Surgery Lab

A config-driven PyTorch lab for **controlled transformer architecture experiments**. Swap one component at a time — or compare full classical vs modern stacks — and measure the tradeoffs with reproducible runs and a polished dashboard.

## Core idea

Modern decoder-only transformers differ in a handful of high-leverage design choices. This project implements a small causal language model where each choice is **config-selectable**, so comparisons stay fair and explainable.

### Four comparison axes

| Axis | Baseline (classical) | Swap (modern) |
|------|---------------------|---------------|
| Normalization | LayerNorm | RMSNorm |
| Positional encoding | Sinusoidal | RoPE |
| Feed-forward | ReLU MLP | SwiGLU |
| Attention | Multi-head (MHA) | Grouped-query (GQA) |

Experiments follow a disciplined sequence: **baseline → single-axis swaps → combined modern stack**.

## Intended outputs

- Reproducible experiment configs and run artifacts
- Validation loss, perplexity, throughput, latency, memory, and parameter counts
- Streamlit dashboard for side-by-side comparison
- Evidence-based lesson notes for each swap family

## Environment

Development targets **WSL Ubuntu 22.04 + Docker**. The Makefile wraps common Docker workflows.

```bash
make build        # build dev image
make shell        # open interactive dev shell
make prepare-data # prepare dataset (once implemented)
make train        # launch training (once implemented)
make dashboard    # launch Streamlit (once implemented)
```

## Roadmap

- [x] Project scaffold and Docker environment
- [X] Config system, data pipeline, and TinyStories preparation
- [ ] Baseline decoder-only transformer (LayerNorm / sinusoidal / ReLU / MHA)
- [ ] Training loop, checkpoints, and artifact tracking
- [ ] Architectural swaps and experiment matrix
- [ ] Evaluation, benchmarking, and comparison tooling
- [ ] Streamlit dashboard
- [ ] Tests, lessons learned, and publication polish

## Results

<!-- Placeholder: populate after baseline and swap runs -->

| Run | Val loss | Perplexity | Params | Throughput | Notes |
|-----|----------|------------|--------|------------|-------|
| Baseline (classical) | — | — | — | — | LayerNorm + sinusoidal + ReLU + MHA |
| RMSNorm swap | — | — | — | — | Single-axis |
| RoPE swap | — | — | — | — | Single-axis |
| SwiGLU swap | — | — | — | — | Single-axis |
| GQA swap | — | — | — | — | Single-axis |
| Modern stack | — | — | — | — | RMSNorm + RoPE + SwiGLU + GQA |

## Dashboard

<!-- Placeholder: add screenshot after dashboard is built -->

```bash
make dashboard
```

## Lessons learned

See [`experiments/lessons/`](experiments/lessons/) for evidence-based writeups on each swap family.

## License

MIT (or your preferred license — update before publishing).
