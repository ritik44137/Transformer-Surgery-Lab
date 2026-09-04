"""One-step training smoke: forward, loss, backward, optimizer step."""

from __future__ import annotations

import torch

from tsl.model.factory import build_model
from tsl.train.losses import causal_lm_loss
from tsl.train.optimizer import build_optimizer


def test_one_training_step_cpu() -> None:
    torch.manual_seed(0)
    model = build_model(
        {
            "model": {
                "vocab_size": 40,
                "hidden_size": 32,
                "num_layers": 1,
                "num_heads": 4,
                "num_kv_heads": 4,
                "d_ff": 64,
                "max_seq_len": 12,
                "dropout": 0.0,
            }
        }
    )
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)

    batch, seq, vocab = 2, 8, 40
    input_ids = torch.randint(0, vocab, (batch, seq))
    labels = torch.randint(0, vocab, (batch, seq))

    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    # Second forward should still produce finite logits after the update.
    with torch.no_grad():
        logits2 = model(input_ids)
    assert torch.isfinite(logits2).all()
