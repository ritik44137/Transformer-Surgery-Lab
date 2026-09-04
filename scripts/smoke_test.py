#!/usr/bin/env python3
"""Top-level repo health check: config → model → toy train/eval → artifacts.

Runs quickly on CPU with synthetic batches (no dataset download required).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _log(msg: str) -> None:
    print(f"[smoke] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer Surgery Lab smoke test")
    parser.add_argument(
        "--config",
        default="configs/train/smoke.yaml",
        help="Train settings YAML (merged with a tiny in-memory model config)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from tsl.config import load_config
    from tsl.eval.perplexity import loss_to_perplexity
    from tsl.model.factory import build_model
    from tsl.tracking.schema import make_eval_metrics, make_summary, make_train_metrics
    from tsl.tracking.writer import RunWriter
    from tsl.train.losses import causal_lm_loss
    from tsl.train.optimizer import build_optimizer
    from tsl.utils.seed import set_seed

    set_seed(0)
    device = torch.device(args.device)

    _log(f"loading train settings from {args.config}")
    train_cfg = load_config(args.config, validate=False)
    train = train_cfg.get("train", train_cfg)

    model_cfg = {
        "model": {
            "vocab_size": 64,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 4,
            "d_ff": 64,
            "max_seq_len": 16,
            "dropout": 0.0,
            "norm": "layernorm",
            "positional": "sinusoidal",
            "feedforward": "relu",
            "attention": "mha",
        },
        "train": train,
        "run": {"name": "smoke", "seed": 0, "output_dir": "runs"},
        "data": {"seq_len": 8},
    }

    _log("building model")
    model = build_model(model_cfg).to(device)
    n_params = model.count_parameters()
    _log(f"params={n_params:,}")

    opt = build_optimizer(model, learning_rate=float(train.get("learning_rate", 3e-4)))
    max_steps = int(train.get("max_steps", 2))
    batch_size = int(train.get("batch_size", 2))
    seq_len = 8
    vocab = int(model_cfg["model"]["vocab_size"])

    _log(f"toy train loop ({max_steps} steps, batch={batch_size}, seq={seq_len})")
    model.train()
    last_loss = None
    for step in range(1, max_steps + 1):
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        _log(f"step={step} train_loss={last_loss:.4f}")

    _log("toy eval pass")
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        val_loss = float(causal_lm_loss(model(input_ids), labels).item())
    ppl = loss_to_perplexity(val_loss)
    _log(f"val_loss={val_loss:.4f} perplexity={ppl:.2f}")

    _log("writing smoke artifacts to a temp run dir")
    with tempfile.TemporaryDirectory(prefix="tsl_smoke_") as tmp:
        writer = RunWriter(tmp, "smoke")
        writer.save_metadata(
            {
                "run_name": "smoke",
                "seed": 0,
                "device": str(device),
                "param_count": n_params,
                "trainable_params": n_params,
                "model_variants": {
                    "norm": "layernorm",
                    "positional": "sinusoidal",
                    "feedforward": "relu",
                    "attention": "mha",
                },
            }
        )
        writer.log_train(
            make_train_metrics(
                step=max_steps,
                loss=last_loss or 0.0,
                lr=float(train.get("learning_rate", 3e-4)),
            )
        )
        writer.log_eval(
            make_eval_metrics(step=max_steps, val_loss=val_loss, perplexity=ppl)
        )
        writer.save_summary(
            make_summary(
                run_name="smoke",
                best_val_loss=val_loss,
                final_train_loss=last_loss,
                best_step=max_steps,
                total_steps=max_steps,
                param_count=n_params,
                model_variants={
                    "norm": "layernorm",
                    "positional": "sinusoidal",
                    "feedforward": "relu",
                    "attention": "mha",
                },
            )
        )
        assert (writer.run_dir / "summary.json").is_file()
        _log(f"artifacts ok under {writer.run_dir}")

    _log("PASS")


if __name__ == "__main__":
    main()
