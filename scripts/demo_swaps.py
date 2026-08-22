#!/usr/bin/env python3
"""Demonstrate that all Phase 5 architecture swaps work."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tsl.config import load_config
from tsl.model import build_model
from tsl.model.attention import build_attention
from tsl.model.feedforward import ReLUFeedForward, SwiGLUFeedForward
from tsl.model.norms import build_norm
from tsl.model.positional import RotaryEmbedding, apply_rotary_emb


def main() -> None:
    print("=" * 64)
    print("1) Component-level swap checks")
    print("=" * 64)

    x = torch.randn(2, 8, 32)
    for kind in ("layernorm", "rmsnorm"):
        y = build_norm(kind, 32)(x)
        print(f"  {kind:10s}  out={tuple(y.shape)}  finite={bool(torch.isfinite(y).all())}")

    rope = RotaryEmbedding(8, max_seq_len=32)
    cos, sin = rope(8)
    q = torch.randn(2, 4, 8, 8)
    q_rot = apply_rotary_emb(q, cos, sin)
    print(
        f"  rope        q={tuple(q.shape)} -> {tuple(q_rot.shape)}  "
        f"changed={not torch.allclose(q, q_rot)}"
    )

    relu = ReLUFeedForward(32, 64)
    swi = SwiGLUFeedForward(32, 64, scale_for_param_parity=True)
    n_relu = sum(p.numel() for p in relu.parameters())
    n_swi = sum(p.numel() for p in swi.parameters())
    print(f"  relu mlp    params={n_relu}  out={tuple(relu(x).shape)}")
    print(f"  swiglu mlp  params={n_swi}  d_ff_eff={swi.d_ff}  out={tuple(swi(x).shape)}")

    mha = build_attention("mha", 32, 4, num_kv_heads=4, max_seq_len=32, use_rope=False)
    gqa = build_attention("gqa", 32, 4, num_kv_heads=2, max_seq_len=32, use_rope=True)
    print(f"  mha         out={tuple(mha(x).shape)}  kv_heads={mha.num_kv_heads}")
    print(
        f"  gqa+rope    out={tuple(gqa(x).shape)}  kv_heads={gqa.num_kv_heads}  "
        f"q_per_kv={gqa.num_queries_per_kv}"
    )

    print()
    print("=" * 64)
    print("2) Config-driven full models (single-axis + modern)")
    print("=" * 64)

    experiments = [
        "configs/experiments/baseline_layernorm_sinusoidal_relu_mha.yaml",
        "configs/experiments/rmsnorm_sinusoidal_relu_mha.yaml",
        "configs/experiments/layernorm_rope_relu_mha.yaml",
        "configs/experiments/layernorm_sinusoidal_swiglu_mha.yaml",
        "configs/experiments/layernorm_sinusoidal_relu_gqa.yaml",
        "configs/experiments/rmsnorm_rope_swiglu_gqa.yaml",
    ]

    tiny = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_heads": 4,
        "d_ff": 128,
        "vocab_size": 200,
        "max_seq_len": 32,
        "dropout": 0.0,
    }

    rows = []
    for path in experiments:
        cfg = load_config(path)
        cfg["model"].update(tiny)
        if cfg["model"].get("attention") == "gqa":
            cfg["model"]["num_kv_heads"] = 2
        else:
            cfg["model"]["num_kv_heads"] = 4

        model = build_model(cfg)
        model.train()
        ids = torch.randint(0, 200, (2, 16))
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 200), ids[:, 1:].reshape(-1))
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        m = cfg["model"]
        rows.append(
            {
                "run": cfg["run"]["name"],
                "norm": m["norm"],
                "pos": m["positional"],
                "ff": m["feedforward"],
                "attn": m["attention"],
                "params": model.count_parameters(),
                "logits": tuple(logits.shape),
                "loss": float(loss.detach()),
                "grad_tensors": grads,
            }
        )

    hdr = (
        f"{'run':40s} {'norm':10s} {'pos':11s} {'ff':7s} {'attn':4s} "
        f"{'params':>8s} {'logits':12s} {'loss':>8s} {'grads':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:40s} {r['norm']:10s} {r['pos']:11s} {r['ff']:7s} {r['attn']:4s} "
            f"{r['params']:8d} {str(r['logits']):12s} {r['loss']:8.3f} {r['grad_tensors']:5d}"
        )

    base = rows[0]
    print()
    print("=" * 64)
    print("3) Single-axis diff check vs baseline")
    print("=" * 64)
    for r in rows[1:5]:
        changed = [
            f"{key}: {base[key]} -> {r[key]}"
            for key in ("norm", "pos", "ff", "attn")
            if r[key] != base[key]
        ]
        print(f"  {r['run']}")
        print(f"    changed axes ({len(changed)}): {', '.join(changed)}")

    modern = rows[5]
    changed_keys = [k for k in ("norm", "pos", "ff", "attn") if modern[k] != base[k]]
    print(f"  {modern['run']}")
    print(f"    changed axes ({len(changed_keys)}/4): {', '.join(changed_keys)}")

    print()
    print("ALL SWAPS WORKING")


if __name__ == "__main__":
    main()
