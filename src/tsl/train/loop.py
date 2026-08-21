"""Single train/eval step helpers."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tsl.train.losses import causal_lm_loss
from tsl.train.scheduler import set_optimizer_lr


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr: float,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """One optimizer step. Returns metrics including loss and tokens/sec."""
    model.train()
    set_optimizer_lr(optimizer, lr)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    elapsed = max(time.perf_counter() - t0, 1e-8)

    tokens = input_ids.numel()
    return {
        "loss": float(loss.detach().item()),
        "tokens_per_sec": tokens / elapsed,
        "lr": float(lr),
    }


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate loss on one batch (no grad)."""
    model.eval()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    return {"val_loss": float(loss.item())}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Mean validation loss (and perplexity) over a loader."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        metrics = eval_step(model, batch, device=device)
        total_loss += metrics["val_loss"]
        n += 1
    if n == 0:
        raise RuntimeError("Validation loader produced zero batches")
    mean_loss = total_loss / n
    # Cap for numerical safety when loss is huge early in training.
    ppl = float(torch.exp(torch.tensor(min(mean_loss, 20.0))).item())
    return {"val_loss": mean_loss, "perplexity": ppl}
