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

Use the same tokenizer and the same TinyStories split for every architecture comparison. `data/raw/smoke/` is only there to check the pipeline.

Raw downloads, token arrays, and the tokenizer json are gitignored. See the root `.gitignore`.
