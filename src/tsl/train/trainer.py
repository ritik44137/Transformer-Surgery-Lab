"""High-level training loop coordinator."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tsl.tracking.schema import make_eval_metrics, make_summary, make_train_metrics
from tsl.tracking.writer import RunWriter
from tsl.train.checkpointing import checkpoint_paths, save_checkpoint
from tsl.train.loop import evaluate_loader, train_step
from tsl.train.scheduler import build_lr_scheduler
from tsl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Coordinates dataloaders, model, optimizer, logging, eval, and checkpoints."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        writer: RunWriter,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer

        train_cfg = cfg.get("train", {})
        self.max_steps = int(train_cfg.get("max_steps", 1000))
        self.log_interval = int(train_cfg.get("log_interval", 50))
        self.eval_interval = int(train_cfg.get("eval_interval", 200))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.learning_rate = float(train_cfg.get("learning_rate", 3e-4))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 100))
        schedule = str(train_cfg.get("scheduler", "cosine"))
        min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
        self.eval_max_batches = train_cfg.get("eval_max_batches")
        if self.eval_max_batches is not None:
            self.eval_max_batches = int(self.eval_max_batches)

        self.lr_at = build_lr_scheduler(
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            schedule=schedule,
            min_lr_ratio=min_lr_ratio,
        )

        model_cfg = cfg.get("model", {})
        self.variants = {
            "norm": str(model_cfg.get("norm", "layernorm")),
            "positional": str(model_cfg.get("positional", "sinusoidal")),
            "feedforward": str(model_cfg.get("feedforward", "relu")),
            "attention": str(model_cfg.get("attention", "mha")),
        }

        self.step = 0
        self.best_val_loss: float | None = None
        self.best_step: int | None = None
        self.last_train_loss: float | None = None
        self.paths = checkpoint_paths(writer.run_dir)

    def train(self) -> dict[str, Any]:
        """Run training until ``max_steps``. Returns the final summary dict."""
        self.model.train()
        train_iter = iter(self.train_loader)

        logger.info(
            "Training start: max_steps=%d log_every=%d eval_every=%d device=%s",
            self.max_steps,
            self.log_interval,
            self.eval_interval,
            self.device,
        )

        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            self.step += 1
            lr = self.lr_at(self.step)
            metrics = train_step(
                self.model,
                batch,
                optimizer=self.optimizer,
                device=self.device,
                lr=lr,
                grad_clip=self.grad_clip,
            )
            self.last_train_loss = metrics["loss"]

            if self.step % self.log_interval == 0 or self.step == 1:
                row = make_train_metrics(
                    step=self.step,
                    loss=metrics["loss"],
                    lr=metrics["lr"],
                    tokens_per_sec=metrics["tokens_per_sec"],
                )
                self.writer.log_train(row)
                logger.info(
                    "step %d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f",
                    self.step,
                    self.max_steps,
                    metrics["loss"],
                    metrics["lr"],
                    metrics["tokens_per_sec"],
                )

            if self.step % self.eval_interval == 0 or self.step == self.max_steps:
                self._evaluate_and_checkpoint()

        summary = self._write_summary()
        logger.info("Training complete.")
        return summary

    def _evaluate_and_checkpoint(self) -> None:
        eval_metrics = evaluate_loader(
            self.model,
            self.val_loader,
            device=self.device,
            max_batches=self.eval_max_batches,
        )
        row = make_eval_metrics(
            step=self.step,
            val_loss=eval_metrics["val_loss"],
            perplexity=eval_metrics["perplexity"],
        )
        self.writer.log_eval(row)
        logger.info(
            "eval step=%d  val_loss=%.4f  ppl=%.2f",
            self.step,
            eval_metrics["val_loss"],
            eval_metrics["perplexity"],
        )

        is_best = self.best_val_loss is None or eval_metrics["val_loss"] < self.best_val_loss
        if is_best:
            self.best_val_loss = eval_metrics["val_loss"]
            self.best_step = self.step

        save_checkpoint(
            self.paths["latest"],
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            best_val_loss=self.best_val_loss,
            config=dict(self.cfg),
        )
        if is_best:
            save_checkpoint(
                self.paths["best"],
                model=self.model,
                optimizer=self.optimizer,
                step=self.step,
                best_val_loss=self.best_val_loss,
                config=dict(self.cfg),
            )

    def _write_summary(self) -> dict[str, Any]:
        param_count = (
            self.model.count_parameters()
            if hasattr(self.model, "count_parameters")
            else sum(p.numel() for p in self.model.parameters())
        )
        summary = make_summary(
            run_name=self.writer.run_name,
            best_val_loss=self.best_val_loss,
            final_train_loss=self.last_train_loss,
            best_step=self.best_step,
            total_steps=self.step,
            param_count=param_count,
            model_variants=self.variants,
        )
        self.writer.save_summary(summary)
        return summary
