"""
Training Loop for the LSTM Forecaster.

Features:
- AdamW optimizer with decoupled weight decay.
- OneCycleLR scheduler for super-convergence.
- Gradient clipping to prevent exploding gradients.
- Early stopping with patience.
- Checkpointing with preprocessor state.
- Train/validation split with time-series-aware ordering.
- Apple Silicon MPS acceleration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from config.settings import config, MODEL_DIR
from src.model.lstm import LSTMForecaster, build_model
from src.model.preprocessor import DataPreprocessor


class Trainer:
    """End-to-end training pipeline for the LSTM forecaster."""

    def __init__(
        self,
        model: Optional[LSTMForecaster] = None,
        preprocessor: Optional[DataPreprocessor] = None,
    ) -> None:
        self.cfg = config.model
        self.device = self.cfg.device
        self.model = model or build_model(self.device)
        self.preprocessor = preprocessor or DataPreprocessor()

        # Loss function — Huber loss is more robust to outliers in financial data
        self.criterion = nn.HuberLoss(delta=1.0)

        # AdamW — decoupled weight decay for better generalisation
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0
        self.train_losses: list = []
        self.val_losses: list = []

    def train(
        self,
        train_df: "pandas.DataFrame",
        val_split: float = 0.15,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_df: Raw DataFrame with OHLCV data (must have 'close', 'volume', etc.)
            val_split: Fraction of data to use for validation (time-ordered split).

        Returns:
            Training metrics dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting LSTM Training Pipeline")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max epochs: {self.cfg.max_epochs}")
        logger.info(f"Batch size: {self.cfg.batch_size}")
        logger.info("=" * 60)

        # ── Preprocess ──
        features_df = self.preprocessor.fit_transform(train_df)

        # ── Create sequences ──
        X, y = self.preprocessor.create_sequences(features_df)

        # ── Time-series split (no shuffling — preserve temporal order) ──
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")

        # ── DataLoaders ──
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        # pin_memory is only beneficial for CUDA; it's unsupported on MPS
        use_pin_memory = (self.device == "cuda")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,  # OK to shuffle sequences (not timesteps within)
            drop_last=True,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        )

        # ── Scheduler ──
        scheduler = self._build_scheduler(train_loader)

        # ── Training loop ──
        start_time = time.time()

        for epoch in range(1, self.cfg.max_epochs + 1):
            self.epoch = epoch

            # Train epoch
            train_loss = self._train_epoch(train_loader, scheduler)
            self.train_losses.append(train_loss)

            # Validate epoch
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Logging
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d}/{self.cfg.max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                logger.success(
                    f"  ↳ New best! Val loss improved by {improvement:.6f}"
                )
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.cfg.patience:
                logger.warning(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={self.cfg.patience})"
                )
                break

            # Periodic checkpoint
            if epoch % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(
                    epoch, val_loss, suffix=f"_epoch{epoch}"
                )

        total_time = time.time() - start_time
        metrics = {
            "total_epochs": self.epoch,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "total_time_seconds": total_time,
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
        }

        logger.success(
            f"Training complete in {total_time:.1f}s | "
            f"Best val loss: {self.best_val_loss:.6f}"
        )

        return metrics

    def _train_epoch(
        self, loader: DataLoader, scheduler: Optional[object]
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        nan_streak = 0  # V6 FIX: Track consecutive NaN batches

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            # V6 FIX: NaN guard — skip corrupted batches, abort on persistence
            if torch.isnan(loss) or torch.isinf(loss):
                nan_streak += 1
                logger.warning(
                    f"NaN/Inf loss detected (streak={nan_streak}). "
                    f"Skipping batch to protect optimizer state."
                )
                if nan_streak >= 5:
                    logger.critical(
                        "5 consecutive NaN losses — aborting epoch. "
                        "Check data for NaN/Inf values or reduce learning rate."
                    )
                    break
                continue

            nan_streak = 0  # Reset on valid loss

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # V6 FIX: Check for NaN gradients after backward pass (MPS edge case)
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                logger.warning("NaN gradients detected — skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )

            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _build_scheduler(self, train_loader: DataLoader):
        """Build learning rate scheduler."""
        total_steps = len(train_loader) * self.cfg.max_epochs

        if self.cfg.scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.onecycle_max_lr,
                total_steps=total_steps,
                pct_start=self.cfg.onecycle_pct_start,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=10000.0,
            )
        elif self.cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.cfg.scheduler == "plateau":
            # Note: ReduceLROnPlateau needs to be stepped differently
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        return None

    def _save_checkpoint(
        self, epoch: int, val_loss: float, suffix: str = ""
    ) -> None:
        """Save model checkpoint with preprocessor state."""
        checkpoint_path = self.cfg.checkpoint_path
        if suffix:
            p = Path(checkpoint_path)
            checkpoint_path = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "preprocessor_state": self.preprocessor.get_state(),
            "model_config": {
                "input_features": self.model.input_features,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout_rate,
                "bidirectional": self.model.bidirectional,
                "use_attention": self.model.use_attention,
                "forecast_horizon": self.model.forecast_horizon,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None
    ) -> None:
        """Load a model checkpoint."""
        path = checkpoint_path or self.cfg.checkpoint_path

        if not Path(path).exists():
            logger.warning(f"No checkpoint found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        # Restore preprocessor state
        if "preprocessor_state" in checkpoint:
            self.preprocessor.load_state(checkpoint["preprocessor_state"])

        logger.success(
            f"Checkpoint loaded from {path} (epoch {self.epoch}, "
            f"val_loss={self.best_val_loss:.6f})"
        )
