"""Small autoregressive text generation for qualitative samples."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Greedy / sampled generation from *prompt_ids* ``(batch, prompt_len)``.

    Returns the full sequence ``(batch, prompt_len + generated)``.
    Temperature ``<= 0`` forces greedy (argmax) decoding.
    """
    model.eval()
    if prompt_ids.ndim != 2:
        raise ValueError(f"prompt_ids must be (B, T), got {tuple(prompt_ids.shape)}")

    tokens = prompt_ids
    device = next(model.parameters()).device
    tokens = tokens.to(device)

    for _ in range(max_new_tokens):
        # Crop to model context if needed (use last max_seq if attribute exists).
        logits = model(tokens)  # (B, T, V)
        next_logits = logits[:, -1, :]

        if temperature is None or temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            logits_scaled = next_logits / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
                cutoff = values[:, -1].unsqueeze(-1)
                logits_scaled = logits_scaled.masked_fill(logits_scaled < cutoff, float("-inf"))
            probs = F.softmax(logits_scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_id], dim=1)
        if eos_id is not None and bool((next_id == eos_id).all()):
            break

    return tokens


def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> dict[str, str | list[int]]:
    """Encode *prompt*, generate, and decode to a sample dict."""
    from tsl.data.tokenizer import encode, decode

    ids = encode(tokenizer, prompt, add_special_tokens=True)
    # Drop trailing EOS from the prompt encoding so generation can continue.
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is not None and ids and ids[-1] == eos_id:
        ids = ids[:-1]
    prompt_t = torch.tensor([ids], dtype=torch.long)
    out = generate(
        model,
        prompt_t,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    out_ids = out[0].tolist()
    return {
        "prompt": prompt,
        "prompt_ids": ids,
        "output_ids": out_ids,
        "text": decode(tokenizer, out_ids, skip_special_tokens=True),
    }
