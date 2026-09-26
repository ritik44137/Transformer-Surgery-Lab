"""Byte-level BPE. Train once and reuse the same artifact for every run."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from tsl.constants import TOKENIZER_FILENAME
from tsl.utils.io import ensure_dir
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def tokenizer_path(tokenizer_dir: str | Path) -> Path:
    return Path(tokenizer_dir) / TOKENIZER_FILENAME


def train_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    logger.info("Trained tokenizer with vocab_size=%s", tokenizer.get_vocab_size())
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, tokenizer_dir: str | Path) -> Path:
    path = tokenizer_path(tokenizer_dir)
    ensure_dir(path.parent)
    tokenizer.save(str(path))
    logger.info("Saved tokenizer to %s", path)
    return path


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    path = tokenizer_path(tokenizer_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer artifact not found: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    # early smoke dumps were saved without a decoder
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevelDecoder()
    logger.info("Loaded tokenizer from %s (vocab=%s)", path, tokenizer.get_vocab_size())
    return tokenizer


def encode(
    tokenizer: Tokenizer,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    encoding = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return encoding.ids


def encode_batch(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    add_special_tokens: bool = True,
) -> list[list[int]]:
    encodings = tokenizer.encode_batch(list(texts), add_special_tokens=add_special_tokens)
    return [e.ids for e in encodings]


def decode(tokenizer: Tokenizer, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def get_or_train_tokenizer(
    texts: Iterable[str],
    tokenizer_dir: str | Path,
    *,
    vocab_size: int = 8000,
    force_retrain: bool = False,
) -> Tokenizer:
    path = tokenizer_path(tokenizer_dir)
    if path.is_file() and not force_retrain:
        return load_tokenizer(tokenizer_dir)
    tokenizer = train_tokenizer(texts, vocab_size=vocab_size)
    save_tokenizer(tokenizer, tokenizer_dir)
    return tokenizer
