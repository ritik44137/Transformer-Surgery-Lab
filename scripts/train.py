#!/usr/bin/env python3
"""Train a Transformer Surgery Lab experiment from YAML config(s)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.config import load_config
from tsl.data import build_dataloaders
from tsl.data.tokenizer import load_tokenizer
from tsl.model import build_model
from tsl.tracking.schema import make_metadata
from tsl.tracking.writer import RunWriter
from tsl.train.optimizer import build_optimizer, param_group_summary
from tsl.train.trainer import Trainer
from tsl.utils.device import device_info, get_device
from tsl.utils.logging_utils import get_logger
from tsl.utils.seed import set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TSL experiment.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/experiments/baseline_layernorm_sinusoidal_relu_mha.yaml"],
        help="YAML config path(s). Later files override earlier; includes are expanded.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (default: auto).",
    )
    return parser.parse_args()


def _resolve_device(choice: str):
    if choice == "cpu":
        return get_device(prefer_cuda=False)
    if choice == "cuda":
        return get_device(prefer_cuda=True)
    return get_device(prefer_cuda=True)


def _sync_vocab_from_tokenizer(cfg: dict) -> None:
    """Align model.vocab_size with the frozen tokenizer when available."""
    tok_dir = cfg.get("data", {}).get("tokenizer_dir")
    if not tok_dir:
        return
    path = Path(tok_dir) / "tokenizer.json"
    if not path.is_file():
        return
    tok = load_tokenizer(tok_dir)
    vocab = tok.get_vocab_size()
    cfg.setdefault("model", {})["vocab_size"] = vocab
    logger.info("Synced model.vocab_size=%d from tokenizer", vocab)

    # Keep attention PE buffers large enough for configured sequence length.
    seq_len = int(cfg.get("data", {}).get("seq_len", 256))
    model_max = int(cfg["model"].get("max_seq_len", seq_len))
    if model_max < seq_len:
        cfg["model"]["max_seq_len"] = seq_len
        logger.info("Raised model.max_seq_len to %d to match data.seq_len", seq_len)


def main() -> int:
    args = parse_args()
    cfg = load_config(*args.config)

    run_cfg = cfg.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    set_seed(seed)

    device = _resolve_device(args.device)
    info = device_info(device)
    logger.info("Device: %s", info)

    _sync_vocab_from_tokenizer(cfg)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    train_cfg = cfg.get("train", {})
    optimizer = build_optimizer(
        model,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )
    logger.info("Optimizer groups: %s", param_group_summary(optimizer))

    run_name = str(run_cfg.get("name", "tsl-run"))
    output_dir = str(run_cfg.get("output_dir", "runs"))
    writer = RunWriter(output_dir, run_name)
    writer.save_config(cfg)

    param_count = model.count_parameters()
    trainable = model.count_parameters(trainable_only=True)
    variants = {
        "norm": str(cfg["model"].get("norm")),
        "positional": str(cfg["model"].get("positional")),
        "feedforward": str(cfg["model"].get("feedforward")),
        "attention": str(cfg["model"].get("attention")),
    }
    writer.save_metadata(
        make_metadata(
            run_name=run_name,
            seed=seed,
            device=str(device),
            param_count=param_count,
            trainable_params=trainable,
            model_variants=variants,
            extra={"device_info": info},
        )
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        writer=writer,
    )
    summary = trainer.train()
    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
