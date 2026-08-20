"""Canonical names for artifacts, metrics, and architecture variants.

Keep all string identifiers here so configs, tracking, and the dashboard
stay consistent across experiments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Architecture variant names (must match config values)
# ---------------------------------------------------------------------------

NORM_LAYERNORM = "layernorm"
NORM_RMSNORM = "rmsnorm"
NORM_VARIANTS = (NORM_LAYERNORM, NORM_RMSNORM)

POS_SINUSOIDAL = "sinusoidal"
POS_ROPE = "rope"
POS_VARIANTS = (POS_SINUSOIDAL, POS_ROPE)

FF_RELU = "relu"
FF_SWIGLU = "swiglu"
FF_VARIANTS = (FF_RELU, FF_SWIGLU)

ATTN_MHA = "mha"
ATTN_GQA = "gqa"
ATTN_VARIANTS = (ATTN_MHA, ATTN_GQA)

# ---------------------------------------------------------------------------
# Artifact filenames inside a run directory
# ---------------------------------------------------------------------------

CONFIG_RESOLVED_FILENAME = "config_resolved.yaml"
METADATA_FILENAME = "metadata.json"
METRICS_TRAIN_FILENAME = "metrics_train.jsonl"
METRICS_EVAL_FILENAME = "metrics_eval.jsonl"
SUMMARY_FILENAME = "summary.json"
BENCHMARK_FILENAME = "benchmark.json"
CHECKPOINT_LATEST = "checkpoint_latest.pt"
CHECKPOINT_BEST = "checkpoint_best.pt"
SAMPLES_FILENAME = "samples.json"

# ---------------------------------------------------------------------------
# Directory names
# ---------------------------------------------------------------------------

RUNS_DIRNAME = "runs"
CHECKPOINTS_DIRNAME = "checkpoints"
DATA_RAW_DIRNAME = "raw"
DATA_PROCESSED_DIRNAME = "processed"
DATA_TOKENIZER_DIRNAME = "tokenizer"

# Processed data artifact names
TRAIN_TOKENS_FILENAME = "train_tokens.npy"
VAL_TOKENS_FILENAME = "val_tokens.npy"
DATA_META_FILENAME = "meta.json"
TOKENIZER_FILENAME = "tokenizer.json"

# ---------------------------------------------------------------------------
# Metric keys (train / eval / benchmark / summary)
# ---------------------------------------------------------------------------

METRIC_STEP = "step"
METRIC_LOSS = "loss"
METRIC_VAL_LOSS = "val_loss"
METRIC_PERPLEXITY = "perplexity"
METRIC_LR = "lr"
METRIC_TOKENS_PER_SEC = "tokens_per_sec"
METRIC_PARAM_COUNT = "param_count"
METRIC_TRAINABLE_PARAMS = "trainable_params"
METRIC_FORWARD_LATENCY_MS = "forward_latency_ms"
METRIC_GEN_LATENCY_MS = "gen_latency_ms"
METRIC_PEAK_MEMORY_MB = "peak_memory_mb"

REQUIRED_CONFIG_SECTIONS = ("run", "data", "model", "train")
