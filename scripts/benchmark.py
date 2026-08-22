#!/usr/bin/env python3
"""Efficiency benchmarks: params, throughput, latency, memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl.constants import BENCHMARK_FILENAME
from tsl.eval.latency import measure_forward_latency, measure_generation_latency
from tsl.eval.memory import measure_peak_memory, reset_peak_memory
from tsl.eval.param_count import count_parameters
from tsl.eval.run_load import load_run_model
from tsl.eval.throughput import measure_throughput
from tsl.tracking.schema import make_benchmark
from tsl.utils.device import get_device
from tsl.utils.io import write_json
from tsl.utils.logging_utils import get_logger

logger = get_logger("benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a TSL run checkpoint.")
    p.add_argument("--run-dir", required=True, help="Path to runs/<run_name>")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--num-warmup", type=int, default=None)
    p.add_argument("--num-iters", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prefer_cuda = args.device != "cpu"
    device = get_device(prefer_cuda=prefer_cuda)
    if args.device == "cpu":
        device = get_device(prefer_cuda=False)

    run_dir = Path(args.run_dir)
    model, cfg, ckpt_path = load_run_model(
        run_dir, checkpoint=args.checkpoint, device=device, prefer_cuda=prefer_cuda
    )

    bench_cfg = cfg.get("benchmark", {})
    batch_size = int(args.batch_size or bench_cfg.get("batch_size", 8))
    seq_len = int(args.seq_len or bench_cfg.get("seq_len", cfg.get("data", {}).get("seq_len", 64)))
    # Clamp to model max_seq_len
    max_seq = int(cfg.get("model", {}).get("max_seq_len", seq_len))
    seq_len = min(seq_len, max_seq)
    num_warmup = int(args.num_warmup or bench_cfg.get("num_warmup", 3))
    num_iters = int(args.num_iters or bench_cfg.get("num_iters", 10))
    vocab_size = int(cfg["model"]["vocab_size"])

    params = count_parameters(model)
    logger.info("params: %s", params)

    reset_peak_memory(device)

    def _run_bench():
        thr = measure_throughput(
            model,
            mode="inference",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )
        fwd = measure_forward_latency(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )
        gen = measure_generation_latency(
            model,
            prompt_len=min(16, seq_len),
            max_new_tokens=16,
            vocab_size=vocab_size,
            device=device,
            num_warmup=max(1, num_warmup - 1),
            num_iters=max(3, num_iters // 2),
        )
        return thr, fwd, gen

    mem = measure_peak_memory(_run_bench, device=device)
    thr, fwd, gen = mem["result"]

    payload = make_benchmark(
        param_count=params["param_count"],
        tokens_per_sec=thr["tokens_per_sec"],
        forward_latency_ms=fwd["forward_latency_ms"],
        gen_latency_ms=gen["gen_latency_ms"],
        peak_memory_mb=mem.get("peak_memory_mb"),
        extra={
            "trainable_params": params["trainable_params"],
            "checkpoint": str(ckpt_path),
            "device": str(device),
            "throughput": thr,
            "forward": fwd,
            "generation": gen,
            "memory": {k: v for k, v in mem.items() if k != "result"},
        },
    )

    out_path = run_dir / BENCHMARK_FILENAME
    write_json(payload, out_path)
    logger.info(
        "tok/s=%.1f  fwd_ms=%.2f  gen_ms=%.2f  peak_mem_mb=%s  -> %s",
        payload["tokens_per_sec"],
        payload["forward_latency_ms"],
        payload["gen_latency_ms"],
        payload.get("peak_memory_mb"),
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
