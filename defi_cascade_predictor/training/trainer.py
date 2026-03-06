"""
Training pipeline for the TGN and baseline models.

Implements:
  - Temporal train/val/test splitting (no data leakage)
  - K-fold temporal cross-validation
  - Early stopping with patience
  - Learning rate scheduling
  - Gradient clipping
  - Checkpoint saving
"""

import os
import time
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

from .losses import CascadeLoss


class Trainer:
    """Trains the TGN model on temporal graph snapshot sequences."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = "cpu",
        output_dir: str = "outputs/checkpoints",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})

        # Loss function
        self.criterion = CascadeLoss(
            prediction_horizons=train_cfg.get(
                "prediction_horizons", [24, 72, 168, 720]
            ),
            focal_gamma=train_cfg.get("focal_loss_gamma", 2.0),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=sched_cfg.get("T_max", 200),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

        # Training params
        self.epochs = train_cfg.get("epochs", 200)
        self.patience = train_cfg.get("patience", 20)
        self.grad_clip = 1.0

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0

    def temporal_split(
        self,
        data: list[dict],
        labels: list[dict],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple:
        """Split temporal data chronologically (no data leakage).

        Returns:
            (train_data, train_labels, val_data, val_labels,
             test_data, test_labels)
        """
        n = len(data)
        test_start = int(n * (1 - test_ratio))
        val_start = int(n * (1 - test_ratio - val_ratio))

        return (
            data[:val_start], labels[:val_start],
            data[val_start:test_start], labels[val_start:test_start],
            data[test_start:], labels[test_start:],
        )

    def prepare_snapshot_dict(
        self,
        node_features: np.ndarray,
        edge_index_dict: dict[str, torch.Tensor],
        timestamp: float,
        edge_attr_dict: Optional[dict] = None,
    ) -> dict:
        """Convert raw data into the format expected by TGN.forward()."""
        return {
            "node_features": torch.tensor(
                node_features, dtype=torch.float32
            ).to(self.device),
            "edge_index_dict": {
                k: v.to(self.device) for k, v in edge_index_dict.items()
            },
            "timestamp": torch.tensor(
                [timestamp] * node_features.shape[0], dtype=torch.float32
            ).to(self.device),
            "edge_attr_dict": (
                {k: v.to(self.device) for k, v in edge_attr_dict.items()}
                if edge_attr_dict
                else None
            ),
        }

    def prepare_target_dict(self, label_row: dict) -> dict[str, torch.Tensor]:
        """Convert label row into target tensor dict."""
        targets = {}
        for key, val in label_row.items():
            if key.startswith("cascade_") or key in ("severity", "risk_score"):
                tgt_key = "severity" if key in ("severity", "risk_score") else key
                targets[tgt_key] = torch.tensor(
                    float(val), dtype=torch.float32
                ).to(self.device)
        return targets

    def train_epoch(
        self,
        train_data: list[dict],
        train_labels: list[dict],
        tbptt_window: int = 10,
    ) -> float:
        """Train for one epoch over the temporal sequence.

        Uses windowed TBPTT: accumulates loss over tbptt_window steps,
        backprops once, then detaches memory at the window boundary.
        Memory is NOT reset at epoch start — it persists across epochs.
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        window_loss = torch.tensor(0.0, device=self.device)
        window_count = 0

        for i in range(len(train_data)):
            snapshot = train_data[i]
            targets = train_labels[i]

            # Forward pass
            predictions = self.model(
                node_features=snapshot["node_features"],
                edge_index_dict=snapshot["edge_index_dict"],
                timestamps=snapshot["timestamp"],
                edge_attr_dict=snapshot.get("edge_attr_dict"),
            )

            # Compute loss
            losses = self.criterion(predictions, targets)
            loss = losses["total_loss"]
            window_loss = window_loss + loss
            window_count += 1
            epoch_loss += loss.item()
            n_batches += 1

            # Windowed TBPTT: backprop at window boundaries
            if window_count >= tbptt_window or i == len(train_data) - 1:
                avg_loss = window_loss / window_count
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()
                if hasattr(self.model, "memory"):
                    self.model.memory.detach_memory()
                window_loss = torch.tensor(0.0, device=self.device)
                window_count = 0

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(
        self,
        val_data: list[dict],
        val_labels: list[dict],
        warmup_data: list[dict] = None,
    ) -> tuple[float, dict]:
        """Validate on held-out temporal data.

        Args:
            val_data: Validation snapshots.
            val_labels: Validation labels.
            warmup_data: If provided, run through these first to build memory state.
        """
        self.model.eval()
        val_loss = 0.0
        all_predictions = {
            f"cascade_{h}h": [] for h in self.config.get(
                "training", {}
            ).get("prediction_horizons", [24, 72, 168, 720])
        }
        all_targets = {k: [] for k in all_predictions}
        n = 0

        # Warm up memory on training data
        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

        if warmup_data:
            for snapshot in warmup_data:
                self.model(
                    node_features=snapshot["node_features"],
                    edge_index_dict=snapshot["edge_index_dict"],
                    timestamps=snapshot["timestamp"],
                    edge_attr_dict=snapshot.get("edge_attr_dict"),
                )
                if hasattr(self.model, "memory"):
                    self.model.memory.detach_memory()

        for i in range(len(val_data)):
            snapshot = val_data[i]
            targets = val_labels[i]

            predictions = self.model(
                node_features=snapshot["node_features"],
                edge_index_dict=snapshot["edge_index_dict"],
                timestamps=snapshot["timestamp"],
                edge_attr_dict=snapshot.get("edge_attr_dict"),
            )

            losses = self.criterion(predictions, targets)
            val_loss += losses["total_loss"].item()

            for key in all_predictions:
                if key in predictions:
                    pred_val = torch.sigmoid(predictions[key]).cpu().item()
                    all_predictions[key].append(pred_val)
                if key in targets:
                    all_targets[key].append(targets[key].cpu().item())

            n += 1

            if hasattr(self.model, "memory"):
                self.model.memory.detach_memory()

        avg_loss = val_loss / max(n, 1)
        return avg_loss, all_predictions, all_targets

    def train(
        self,
        train_data: list[dict],
        train_labels: list[dict],
        val_data: list[dict],
        val_labels: list[dict],
    ) -> dict:
        """Full training loop with early stopping.

        Returns:
            Dict with training history and best metrics.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"patience={self.patience}, device={self.device}"
        )
        logger.info(
            f"Train: {len(train_data)} snapshots, Val: {len(val_data)} snapshots"
        )

        start_time = time.time()

        # Reset memory only once at training start
        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

        for epoch in range(self.epochs):
            # Train (memory persists across epochs)
            train_loss = self.train_epoch(train_data, train_labels)
            self.train_losses.append(train_loss)

            # Validate (with warmup on training data)
            val_loss, val_preds, val_tgts = self.validate(
                val_data, val_labels, warmup_data=train_data
            )
            self.val_losses.append(val_loss)

            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0

                # Save checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.best_model_state,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.output_dir / "best_model.pt",
                )
            else:
                self.epochs_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"No Improve: {self.epochs_no_improve}/{self.patience}"
                )

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        elapsed = time.time() - start_time
        logger.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Best val loss: {self.best_val_loss:.6f}"
        )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": len(self.train_losses) - self.epochs_no_improve,
            "total_epochs": len(self.train_losses),
            "training_time": elapsed,
        }

    def temporal_cross_validate(
        self,
        all_data: list[dict],
        all_labels: list[dict],
        n_folds: int = 5,
    ) -> list[dict]:
        """Expanding window temporal cross-validation.

        Uses expanding training windows to respect temporal ordering:
          Fold 1: Train [0:n1], Val [n1:n2]
          Fold 2: Train [0:n2], Val [n2:n3]
          ...

        Returns:
            List of per-fold results.
        """
        n = len(all_data)
        fold_size = n // (n_folds + 1)
        results = []

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            val_end = min(train_end + fold_size, n)

            if val_end <= train_end:
                break

            train_d = all_data[:train_end]
            train_l = all_labels[:train_end]
            val_d = all_data[train_end:val_end]
            val_l = all_labels[train_end:val_end]

            logger.info(
                f"Fold {fold + 1}/{n_folds}: "
                f"Train [{0}:{train_end}], Val [{train_end}:{val_end}]"
            )

            # Reset model and optimizer for each fold
            self.model.apply(self._reset_weights)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("training", {}).get("learning_rate", 3e-4),
                weight_decay=self.config.get("training", {}).get(
                    "weight_decay", 1e-4
                ),
            )
            self.best_val_loss = float("inf")
            self.epochs_no_improve = 0
            self.train_losses = []
            self.val_losses = []

            fold_result = self.train(train_d, train_l, val_d, val_l)
            fold_result["fold"] = fold + 1

            # Evaluate on validation set
            val_loss, val_preds, val_tgts = self.validate(val_d, val_l)
            fold_result["val_predictions"] = val_preds
            fold_result["val_targets"] = val_tgts

            results.append(fold_result)

        return results

    @staticmethod
    def _reset_weights(m):
        """Reset model weights for cross-validation folds."""
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
