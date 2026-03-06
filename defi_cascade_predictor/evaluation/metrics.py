"""
Evaluation metrics for cascade prediction.

Computes classification metrics, calibration, and cascade-specific
metrics like lead time and early warning score.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    classification_report,
)
from loguru import logger


class MetricsCalculator:
    """Computes comprehensive evaluation metrics for cascade prediction."""

    def __init__(
        self,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        threshold: float = 0.5,
    ):
        self.prediction_horizons = prediction_horizons
        self.threshold = threshold

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = None,
    ) -> dict[str, float]:
        """Compute all metrics for a single horizon.

        Args:
            y_true: Binary labels [n_samples].
            y_prob: Predicted probabilities [n_samples].
            threshold: Classification threshold.

        Returns:
            Dict of metric_name -> value.
        """
        if threshold is None:
            threshold = self.threshold

        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        # Ensure valid
        mask = np.isfinite(y_prob) & np.isfinite(y_true)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("Cannot compute metrics: no positive or no negative samples")
            return self._empty_metrics()

        y_pred = (y_prob >= threshold).astype(int)

        metrics = {}

        # Discrimination metrics
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)

        # Classification metrics
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

        # Calibration
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        metrics["optimal_threshold"] = float(thresholds[optimal_idx])
        metrics["youden_j"] = float(j_scores[optimal_idx])

        # Positive rate
        metrics["positive_rate"] = y_true.mean()
        metrics["predicted_positive_rate"] = y_pred.mean()
        metrics["n_samples"] = len(y_true)

        return metrics

    def compute_multi_horizon_metrics(
        self,
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Compute metrics for all prediction horizons.

        Args:
            predictions: Dict of horizon_key -> probabilities.
            targets: Dict of horizon_key -> binary labels.

        Returns:
            Nested dict: horizon_key -> metric_name -> value.
        """
        all_metrics = {}
        for h in self.prediction_horizons:
            key = f"cascade_{h}h"
            if key in predictions and key in targets:
                y_prob = np.array(predictions[key])
                y_true = np.array(targets[key])
                metrics = self.compute_all_metrics(y_true, y_prob)
                all_metrics[key] = metrics
                logger.info(
                    f"  {key}: AUROC={metrics['auroc']:.4f}, "
                    f"AUPRC={metrics['auprc']:.4f}, "
                    f"F1={metrics['f1']:.4f}"
                )
        return all_metrics

    def compute_lead_time(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        timestamps: np.ndarray,
        cascade_start_times: list,
        threshold: float = None,
    ) -> dict[str, float]:
        """Compute lead time: how far in advance the model detects cascades.

        Args:
            y_true: Binary labels.
            y_prob: Predicted probabilities.
            timestamps: Datetime array aligned with predictions.
            cascade_start_times: List of cascade start timestamps.
            threshold: Detection threshold.

        Returns:
            Dict with mean/median/min lead time in hours.
        """
        if threshold is None:
            threshold = self.threshold

        y_pred = (np.array(y_prob) >= threshold).astype(int)
        lead_times = []

        for cascade_start in cascade_start_times:
            # Find the first prediction before cascade that raised alarm
            pre_cascade = [
                i for i, t in enumerate(timestamps)
                if t < cascade_start and y_pred[i] == 1
            ]
            if pre_cascade:
                first_alarm_idx = min(pre_cascade, key=lambda i: abs(
                    timestamps[i] - cascade_start
                ))
                # For the closest alarm, compute lead time
                last_alarm = max(
                    i for i in pre_cascade if timestamps[i] < cascade_start
                )
                lead_time_hours = (
                    cascade_start - timestamps[last_alarm]
                ).total_seconds() / 3600
                lead_times.append(lead_time_hours)

        if not lead_times:
            return {
                "mean_lead_time_hours": 0,
                "median_lead_time_hours": 0,
                "min_lead_time_hours": 0,
                "max_lead_time_hours": 0,
                "detection_rate": 0,
            }

        return {
            "mean_lead_time_hours": np.mean(lead_times),
            "median_lead_time_hours": np.median(lead_times),
            "min_lead_time_hours": np.min(lead_times),
            "max_lead_time_hours": np.max(lead_times),
            "detection_rate": len(lead_times) / len(cascade_start_times),
        }

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
    ) -> float:
        """Find optimal classification threshold.

        Args:
            y_true: Binary labels.
            y_prob: Predicted probabilities.
            metric: Metric to optimize ("f1", "youden", "precision_recall").

        Returns:
            Optimal threshold value.
        """
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_score = -1
        best_threshold = 0.5

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "youden":
                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                score = tpr - fpr
            elif metric == "precision_recall":
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * p * r / (p + r) if (p + r) > 0 else 0
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = t

        return best_threshold

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty metrics dict when computation is impossible."""
        return {
            "auroc": 0.5, "auprc": 0.0, "f1": 0.0, "precision": 0.0,
            "recall": 0.0, "mcc": 0.0, "brier_score": 0.25,
            "true_positives": 0, "false_positives": 0,
            "true_negatives": 0, "false_negatives": 0,
            "specificity": 0.0, "npv": 0.0, "optimal_threshold": 0.5,
            "youden_j": 0.0, "positive_rate": 0.0,
            "predicted_positive_rate": 0.0, "n_samples": 0,
        }
