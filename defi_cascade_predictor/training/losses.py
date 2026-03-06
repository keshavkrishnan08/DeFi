"""
Custom loss functions for cascade prediction.

Implements:
  - FocalLoss: addresses extreme class imbalance (cascades are rare)
  - CascadeLoss: multi-task loss combining classification + severity + propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance in cascade prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Uses ONLY focal modulation (gamma) and alpha balancing — no additional
    pos_weight multiplier, which would create triple-redundant weighting
    and unstable gradients.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        pos_weight: float = 1.0,  # kept for API compat; only used in BCE
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for POSITIVE class (0.75 = 3:1 pos bias)
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid).
            targets: Binary labels (0 or 1).
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # BCE loss per element
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Focal modulation: down-weight easy examples
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing: higher alpha = more weight on positives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CascadeLoss(nn.Module):
    """Multi-task loss for cascade prediction combining:
      1. Multi-horizon classification (focal loss)
      2. Severity estimation (MSE)
      3. Propagation path prediction (BCE)

    Total loss = sum of weighted component losses.
    """

    def __init__(
        self,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
        severity_weight: float = 0.3,
        propagation_weight: float = 0.2,
        horizon_weights: dict[int, float] = None,
    ):
        super().__init__()
        self.prediction_horizons = prediction_horizons
        self.severity_weight = severity_weight
        self.propagation_weight = propagation_weight

        # Different weights for different horizons
        if horizon_weights is None:
            self.horizon_weights = {h: 1.0 for h in prediction_horizons}
            # Short-term predictions weighted higher
            if 24 in self.horizon_weights:
                self.horizon_weights[24] = 1.5
            if 72 in self.horizon_weights:
                self.horizon_weights[72] = 1.2
        else:
            self.horizon_weights = horizon_weights

        # Focal loss for each horizon
        self.focal_losses = {
            h: FocalLoss(gamma=focal_gamma, alpha=0.75)
            for h in prediction_horizons
        }

        # Severity loss
        self.severity_loss = nn.MSELoss()

        # Propagation loss
        self.propagation_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model outputs dict with keys:
                cascade_{h}h, severity, propagation.
            targets: Ground truth dict with matching keys.

        Returns:
            Dict with total_loss and component losses.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # 1. Multi-horizon cascade classification losses
        for h in self.prediction_horizons:
            key = f"cascade_{h}h"
            if key in predictions and key in targets:
                pred = predictions[key]
                tgt = targets[key]
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if tgt.dim() == 0:
                    tgt = tgt.unsqueeze(0)
                loss = self.focal_losses[h](pred, tgt)
                weighted = self.horizon_weights.get(h, 1.0) * loss
                losses[f"loss_{key}"] = loss
                total = total + weighted

        # 2. Severity estimation loss
        if "severity" in predictions and "severity" in targets:
            sev_loss = self.severity_loss(
                predictions["severity"], targets["severity"]
            )
            losses["loss_severity"] = sev_loss
            total = total + self.severity_weight * sev_loss

        # 3. Propagation path loss
        if "propagation" in predictions and "propagation" in targets:
            prop_loss = self.propagation_loss(
                predictions["propagation"], targets["propagation"]
            )
            losses["loss_propagation"] = prop_loss
            total = total + self.propagation_weight * prop_loss

        losses["total_loss"] = total
        return losses


class MonotonicityRegularization(nn.Module):
    """Enforce P(longer horizon) >= P(shorter horizon).

    Penalizes cases where a shorter-horizon cascade probability exceeds
    a longer-horizon probability, since a cascade within 7 days implies
    a cascade within 30 days.

    Uses a hinge loss on the raw logits: ReLU(shorter_logit - longer_logit).
    """

    def __init__(self, prediction_horizons: list[int], weight: float = 0.5):
        super().__init__()
        self.horizons = sorted(prediction_horizons)
        self.weight = weight

    def forward(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(predictions.values())).device
        loss = torch.tensor(0.0, device=device)
        sorted_keys = [f"cascade_{h}h" for h in self.horizons]
        available = [k for k in sorted_keys if k in predictions]
        for i in range(len(available) - 1):
            shorter = predictions[available[i]]
            longer = predictions[available[i + 1]]
            violations = torch.relu(shorter - longer)
            loss = loss + violations.mean()
        return self.weight * loss
