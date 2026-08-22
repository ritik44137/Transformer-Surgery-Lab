"""Config-driven model construction.

Variant selection is explicit if/elif — no registry magic — so the factory
stays easy to explain and debug in an interview.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from tsl.constants import (
    ATTN_MHA,
    FF_RELU,
    NORM_LAYERNORM,
    POS_ROPE,
    POS_SINUSOIDAL,
)
from tsl.model.attention import build_attention
from tsl.model.block import TransformerBlock
from tsl.model.embeddings import TokenEmbeddings
from tsl.model.feedforward import build_feedforward
from tsl.model.norms import build_norm
from tsl.model.positional import build_positional
from tsl.model.transformer import DecoderLM
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _model_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a full resolved config or a bare ``model`` mapping."""
    if "model" in cfg and isinstance(cfg["model"], Mapping):
        return dict(cfg["model"])
    return dict(cfg)


def build_model(cfg: Mapping[str, Any]) -> DecoderLM:
    """Build a :class:`DecoderLM` from a resolved config.

    Required model keys: ``vocab_size``, ``hidden_size``, ``num_layers``,
    ``num_heads``, ``d_ff``, ``max_seq_len``. Variant keys default to the
    classical baseline (LayerNorm / sinusoidal / ReLU / MHA).
    """
    m = _model_section(cfg)

    vocab_size = int(m["vocab_size"])
    hidden_size = int(m["hidden_size"])
    num_layers = int(m["num_layers"])
    num_heads = int(m["num_heads"])
    num_kv_heads = int(m.get("num_kv_heads", num_heads))
    d_ff = int(m["d_ff"])
    max_seq_len = int(m.get("max_seq_len", 256))
    dropout = float(m.get("dropout", 0.0))
    eps = float(m.get("norm_eps", 1e-5))
    tie_weights = bool(m.get("tie_weights", True))
    scale_swiglu = bool(m.get("swiglu_scale_for_param_parity", True))
    rope_base = float(m.get("rope_base", 10000.0))

    norm_kind = str(m.get("norm", NORM_LAYERNORM)).lower()
    pos_kind = str(m.get("positional", POS_SINUSOIDAL)).lower()
    ff_kind = str(m.get("feedforward", FF_RELU)).lower()
    attn_kind = str(m.get("attention", ATTN_MHA)).lower()
    use_rope = pos_kind == POS_ROPE

    positional = build_positional(pos_kind, hidden_size, max_seq_len=max_seq_len)
    embeddings = TokenEmbeddings(
        vocab_size,
        hidden_size,
        positional=positional,
        dropout=dropout,
    )

    blocks: list[TransformerBlock] = []
    for _ in range(num_layers):
        block = TransformerBlock(
            norm_attn=build_norm(norm_kind, hidden_size, eps=eps),
            attention=build_attention(
                attn_kind,
                hidden_size,
                num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_rope=use_rope,
                rope_base=rope_base,
            ),
            norm_ff=build_norm(norm_kind, hidden_size, eps=eps),
            feedforward=build_feedforward(
                ff_kind,
                hidden_size,
                d_ff=d_ff,
                dropout=dropout,
                scale_for_param_parity=scale_swiglu,
            ),
        )
        blocks.append(block)

    final_norm = build_norm(norm_kind, hidden_size, eps=eps)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    model = DecoderLM(
        embeddings=embeddings,
        blocks=nn.ModuleList(blocks),
        final_norm=final_norm,
        lm_head=lm_head,
        tie_weights=tie_weights,
    )

    n_params = model.count_parameters()
    logger.info(
        "Built DecoderLM: layers=%d hidden=%d heads=%d kv_heads=%d d_ff=%d vocab=%d "
        "params=%s (norm=%s pos=%s ff=%s attn=%s)",
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        d_ff,
        vocab_size,
        f"{n_params:,}",
        norm_kind,
        pos_kind,
        ff_kind,
        attn_kind,
    )
    return model
