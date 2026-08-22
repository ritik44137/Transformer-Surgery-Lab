#!/usr/bin/env python3
"""Post-training evaluation: val metrics + optional text samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.constants import SAMPLES_FILENAME
from tsl.data.datamodule import DataModule
from tsl.data.tokenizer import load_tokenizer
from tsl.eval.generation import generate_text
from tsl.eval.perplexity import loss_to_perplexity
from tsl.eval.run_load import load_run_model
from tsl.tracking.schema import make_eval_metrics
from tsl.train.loop import evaluate_loader
from tsl.utils.device import get_device
from tsl.utils.io import write_json
from tsl.utils.logging_utils import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a TSL run checkpoint.")
    p.add_argument("--run-dir", required=True, help="Path to runs/<run_name>")
    p.add_argument(
        "--checkpoint",
        default="best",
        help="best | latest | filename (default: best)",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--generate", action="store_true", help="Also write text samples")
    p.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used when --generate is set",
    )
    p.add_argument("--max-new-tokens", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prefer_cuda = args.device != "cpu"
    if args.device == "cuda":
        prefer_cuda = True
    device = get_device(prefer_cuda=prefer_cuda)
    if args.device == "cpu":
        device = get_device(prefer_cuda=False)

    run_dir = Path(args.run_dir)
    model, cfg, ckpt_path = load_run_model(
        run_dir, checkpoint=args.checkpoint, device=device, prefer_cuda=prefer_cuda
    )

    dm = DataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    metrics = evaluate_loader(
        model,
        val_loader,
        device=device,
        max_batches=args.max_batches,
    )
    ppl = loss_to_perplexity(metrics["val_loss"])
    metrics["perplexity"] = ppl

    out = make_eval_metrics(
        step=-1,
        val_loss=metrics["val_loss"],
        perplexity=ppl,
        extra={"checkpoint": str(ckpt_path), "source": "scripts/evaluate.py"},
    )
    report_path = run_dir / "eval_report.json"
    write_json(out, report_path)
    logger.info(
        "val_loss=%.4f  perplexity=%.2f  -> %s",
        metrics["val_loss"],
        ppl,
        report_path,
    )

    if args.generate:
        tok_dir = cfg.get("data", {}).get("tokenizer_dir")
        if not tok_dir or not Path(tok_dir).joinpath("tokenizer.json").is_file():
            logger.warning("Tokenizer not found; skipping generation")
        else:
            tokenizer = load_tokenizer(tok_dir)
            sample = generate_text(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
            )
            samples_path = run_dir / SAMPLES_FILENAME
            write_json({"samples": [sample]}, samples_path)
            logger.info("Sample: %s", sample["text"][:200])
            logger.info("Wrote samples to %s", samples_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
