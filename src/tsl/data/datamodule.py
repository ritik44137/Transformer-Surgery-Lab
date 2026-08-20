"""Config-driven dataloaders over processed token artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from tsl.constants import TRAIN_TOKENS_FILENAME, VAL_TOKENS_FILENAME
from tsl.data.dataset import CausalLMDataset, load_causal_dataset
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataModule:
    """Thin wrapper that builds train/val datasets and dataloaders from config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        data = cfg.get("data", {})
        train_cfg = cfg.get("train", {})

        self.processed_dir = Path(data.get("processed_dir", "data/processed"))
        self.seq_len = int(data.get("seq_len", 256))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.num_workers = int(data.get("num_workers", 0))
        self.pin_memory = bool(data.get("pin_memory", False))

        self.train_dataset: CausalLMDataset | None = None
        self.val_dataset: CausalLMDataset | None = None

    def setup(self) -> None:
        """Load processed token arrays into datasets."""
        train_path = self.processed_dir / TRAIN_TOKENS_FILENAME
        val_path = self.processed_dir / VAL_TOKENS_FILENAME
        if not train_path.is_file():
            raise FileNotFoundError(
                f"Missing {train_path}. Run scripts/prepare_data.py first."
            )
        if not val_path.is_file():
            raise FileNotFoundError(
                f"Missing {val_path}. Run scripts/prepare_data.py first."
            )

        self.train_dataset = load_causal_dataset(train_path, self.seq_len)
        self.val_dataset = load_causal_dataset(val_path, self.seq_len)
        logger.info(
            "Datasets ready: train_windows=%d val_windows=%d seq_len=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            self.seq_len,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Convenience: return ``(train_loader, val_loader)`` from config."""
    dm = DataModule(cfg)
    dm.setup()
    return dm.train_dataloader(), dm.val_dataloader()
