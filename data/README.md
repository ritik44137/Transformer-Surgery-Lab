# Data layout

This directory holds local dataset artifacts. Raw and processed files are **gitignored**; only this README and `.gitkeep` markers are tracked.

## Directory structure

```
data/
├── raw/          # Downloaded or copied source text (TinyStories, smoke-test text)
├── processed/    # Tokenized train/val splits written by prepare_data
└── tokenizer/    # Saved tokenizer artifact (frozen across all experiments)
```

## Workflow

1. Place or download raw data under `data/raw/` (or use HuggingFace `datasets` download in `scripts/prepare_data.py`).
2. Run data preparation:
   - Main experiments: `python scripts/prepare_data.py --config configs/default.yaml configs/data/tinystories.yaml`
   - Local smoke path (no HF download): `python scripts/prepare_data.py --config configs/default.yaml configs/data/smoke.yaml`
3. Training scripts read from `data/processed/` and `data/tokenizer/` via config paths.

## Experiment policy

- **One fixed tokenizer** across all architecture comparisons.
- **One fixed dataset family** (TinyStories) for main comparison runs.
- A separate tiny smoke-test corpus lives under `data/raw/smoke/` for pipeline validation.

## Git ignore rules

See root `.gitignore`. Never commit large raw downloads, tokenized shards, or checkpoints.
