"""Dataset preparation: load text, tokenize, split, and save artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tsl.constants import (
    DATA_META_FILENAME,
    TRAIN_TOKENS_FILENAME,
    VAL_TOKENS_FILENAME,
)
from tsl.data.tokenizer import (
    EOS_TOKEN,
    encode,
    get_or_train_tokenizer,
)
from tsl.utils.io import ensure_dir, write_json
from tsl.utils.logging_utils import get_logger
from tsl.utils.seed import set_seed

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Light whitespace normalization; preserves content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_texts_from_dir(raw_dir: str | Path, *, pattern: str = "*.txt") -> list[str]:
    """Load and normalize all matching text files under *raw_dir*."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    paths = sorted(raw_dir.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {raw_dir}")

    texts: list[str] = []
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        # Split on blank lines into story-like documents when possible.
        chunks = [normalize_text(c) for c in re.split(r"\n\s*\n", content)]
        texts.extend(c for c in chunks if c)
    logger.info("Loaded %d text documents from %s", len(texts), raw_dir)
    return texts


def load_texts_from_hf(
    dataset_name: str,
    *,
    split: str = "train",
    text_column: str = "text",
    max_texts: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Load texts from a HuggingFace datasets identifier."""
    from datasets import load_dataset

    logger.info("Loading HuggingFace dataset %s (split=%s)", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    if max_texts is not None and max_texts < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_texts))

    texts: list[str] = []
    for row in ds:
        value = row.get(text_column)
        if value is None:
            # TinyStories sometimes uses 'story'
            value = row.get("story")
        if not value:
            continue
        texts.append(normalize_text(str(value)))
    logger.info("Loaded %d texts from HuggingFace dataset", len(texts))
    return texts


def load_texts(cfg: dict[str, Any]) -> list[str]:
    """Resolve texts from config: prefer local raw files, else HuggingFace."""
    data = cfg.get("data", cfg)
    raw_dir = Path(data.get("raw_dir", "data/raw"))
    max_texts = data.get("max_texts")
    seed = int(data.get("seed", cfg.get("run", {}).get("seed", 42)))

    # Local override / smoke path
    if raw_dir.is_dir() and any(raw_dir.rglob("*.txt")):
        texts = load_texts_from_dir(raw_dir)
        if max_texts is not None:
            texts = texts[: int(max_texts)]
        return texts

    hf_name = data.get("hf_dataset")
    if not hf_name:
        raise FileNotFoundError(
            f"No local .txt files under {raw_dir} and no data.hf_dataset configured."
        )
    return load_texts_from_hf(
        hf_name,
        split=data.get("hf_split", "train"),
        text_column=data.get("text_column", "text"),
        max_texts=int(max_texts) if max_texts is not None else None,
        seed=seed,
    )


def texts_to_token_ids(
    texts: Sequence[str],
    tokenizer,
) -> np.ndarray:
    """Concatenate documents into one flat int32 token array (EOS-separated)."""
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    ids: list[int] = []
    for text in texts:
        piece = encode(tokenizer, text, add_special_tokens=True)
        ids.extend(piece)
        # TemplateProcessing already adds EOS; avoid double-EOS when missing.
        if eos_id is not None and (not piece or piece[-1] != eos_id):
            ids.append(eos_id)
    if not ids:
        raise ValueError("No tokens produced from input texts")
    return np.asarray(ids, dtype=np.int32)


def split_tokens(
    tokens: np.ndarray,
    *,
    train_split: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous train/val split on the flat token stream."""
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    # Contiguous split keeps document boundaries roughly intact vs shuffling IDs.
    n = len(tokens)
    cut = max(1, int(n * train_split))
    if cut >= n:
        cut = n - 1
    train, val = tokens[:cut], tokens[cut:]
    logger.info(
        "Split tokens: train=%d (%.1f%%) val=%d — seed=%s (contiguous)",
        len(train),
        100.0 * len(train) / n,
        len(val),
        seed,
    )
    return train, val


def save_processed(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    processed_dir: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write train/val token arrays and metadata under *processed_dir*."""
    processed_dir = ensure_dir(processed_dir)
    train_path = processed_dir / TRAIN_TOKENS_FILENAME
    val_path = processed_dir / VAL_TOKENS_FILENAME
    meta_path = processed_dir / DATA_META_FILENAME

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    payload = {
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "dtype": str(train_tokens.dtype),
        "train_file": TRAIN_TOKENS_FILENAME,
        "val_file": VAL_TOKENS_FILENAME,
    }
    if meta:
        payload.update(meta)
    write_json(payload, meta_path)

    logger.info("Wrote processed artifacts to %s", processed_dir)
    return {"train": train_path, "val": val_path, "meta": meta_path}


def prepare_dataset(cfg: dict[str, Any], *, force_retrain_tokenizer: bool = False) -> dict[str, Any]:
    """End-to-end preparation driven by a resolved config dict.

    Returns paths and summary metadata.
    """
    data = cfg.get("data", cfg)
    seed = int(data.get("seed", cfg.get("run", {}).get("seed", 42)))
    set_seed(seed)

    texts = load_texts(cfg)
    tokenizer_dir = data.get("tokenizer_dir", "data/tokenizer")
    vocab_size = int(data.get("vocab_size", 8000))

    # Train tokenizer on the same corpus we will encode (or load frozen artifact).
    tokenizer = get_or_train_tokenizer(
        texts,
        tokenizer_dir,
        vocab_size=vocab_size,
        force_retrain=force_retrain_tokenizer,
    )

    tokens = texts_to_token_ids(texts, tokenizer)
    train_tokens, val_tokens = split_tokens(
        tokens,
        train_split=float(data.get("train_split", 0.95)),
        seed=seed,
    )

    processed_dir = data.get("processed_dir", "data/processed")
    paths = save_processed(
        train_tokens,
        val_tokens,
        processed_dir,
        meta={
            "dataset": data.get("dataset", "unknown"),
            "num_documents": len(texts),
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_dir": str(tokenizer_dir),
            "seq_len": int(data.get("seq_len", 256)),
            "seed": seed,
            "train_split": float(data.get("train_split", 0.95)),
        },
    )
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "num_documents": len(texts),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_dir": str(tokenizer_dir),
        "processed_dir": str(processed_dir),
    }


def load_token_array(path: str | Path) -> np.ndarray:
    """Load a ``.npy`` token array."""
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token array, got shape {arr.shape} from {path}")
    return arr.astype(np.int32, copy=False)
