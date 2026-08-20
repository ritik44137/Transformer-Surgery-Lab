#!/usr/bin/env python3
"""Prepare dataset and tokenizer artifacts for Transformer Surgery Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package when PYTHONPATH is unset.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.config import load_config, save_yaml
from tsl.data.preprocess import prepare_dataset
from tsl.utils.io import ensure_dir
from tsl.utils.logging_utils import get_logger

logger = get_logger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tokenizer + processed token artifacts for TSL."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/default.yaml", "configs/data/tinystories.yaml"],
        help="One or more YAML configs to merge (later overrides earlier).",
    )
    parser.add_argument(
        "--force-retrain-tokenizer",
        action="store_true",
        help="Retrain and overwrite the tokenizer even if an artifact exists.",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Optional override for data.max_texts (useful for smoke runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(*args.config)

    if args.max_texts is not None:
        cfg.setdefault("data", {})["max_texts"] = args.max_texts

    data = cfg["data"]
    logger.info("Preparing dataset=%s", data.get("dataset"))
    logger.info("raw_dir=%s", data.get("raw_dir"))
    logger.info("processed_dir=%s", data.get("processed_dir"))
    logger.info("tokenizer_dir=%s", data.get("tokenizer_dir"))

    result = prepare_dataset(cfg, force_retrain_tokenizer=args.force_retrain_tokenizer)

    # Persist the resolved data config next to processed artifacts for provenance.
    processed_dir = ensure_dir(result["processed_dir"])
    save_yaml({"data": data}, processed_dir / "config_used.yaml")

    logger.info(
        "Done. docs=%d train_tokens=%d val_tokens=%d vocab=%d",
        result["num_documents"],
        result["train_tokens"],
        result["val_tokens"],
        result["vocab_size"],
    )
    for name, path in result["paths"].items():
        logger.info("  %s -> %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
