"""
XGBoost baseline: gradient boosted trees on tabular features.
Measures value of deep learning and graph structure over traditional ML.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


class XGBoostCascadePredictor:
    """XGBoost baseline using hand-crafted tabular features.

    Features include: aggregated protocol features, network centrality
    metrics, and rolling statistics — without any graph neural network
    or temporal deep learning components.
    """

    def __init__(
        self,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        **xgb_params,
    ):
        self.prediction_horizons = prediction_horizons
        self.xgb_params = {
            "n_estimators": xgb_params.get("n_estimators", 500),
            "max_depth": xgb_params.get("max_depth", 8),
            "learning_rate": xgb_params.get("learning_rate", 0.05),
            "subsample": xgb_params.get("subsample", 0.8),
            "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
            "min_child_weight": xgb_params.get("min_child_weight", 5),
            "reg_alpha": xgb_params.get("reg_alpha", 0.1),
            "reg_lambda": xgb_params.get("reg_lambda", 1.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": 1,
        }
        self.models = {}
        self.severity_model = None
        self.feature_names = None

    def prepare_features(
        self,
        node_features_sequence: list[np.ndarray],
        window: int = 30,
    ) -> np.ndarray:
        """Convert temporal graph features into tabular features.

        Aggregates node features across all protocols and computes
        rolling statistics over the temporal window.

        Args:
            node_features_sequence: List of [num_nodes, feature_dim] arrays.
            window: Number of past timesteps to include.

        Returns:
            Tabular feature matrix [num_samples, num_features].
        """
        all_features = []
        seq_len = len(node_features_sequence)

        for t in range(window, seq_len):
            features = []

            # Current timestep features (flattened across all nodes)
            current = node_features_sequence[t].flatten()
            features.extend(current)

            # Aggregated statistics across nodes
            current_2d = node_features_sequence[t]
            features.extend(current_2d.mean(axis=0))  # mean across protocols
            features.extend(current_2d.std(axis=0))   # std across protocols
            features.extend(current_2d.min(axis=0))   # min across protocols
            features.extend(current_2d.max(axis=0))   # max across protocols

            # Temporal statistics over window
            window_data = np.array(
                node_features_sequence[max(0, t - window):t]
            )  # [window, nodes, features]

            # Mean change over window
            if window_data.shape[0] > 1:
                changes = np.diff(window_data, axis=0)
                features.extend(changes.mean(axis=(0, 1)))  # avg change
                features.extend(changes.std(axis=(0, 1)))   # change volatility

                # Trend: first vs last in window
                trend = window_data[-1].mean(axis=0) - window_data[0].mean(axis=0)
                features.extend(trend)
            else:
                n_feat = current_2d.shape[1]
                features.extend(np.zeros(n_feat * 3))

            all_features.append(features)

        return np.array(all_features, dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        y_dict: dict[str, np.ndarray],
        eval_set: Optional[tuple] = None,
    ):
        """Train XGBoost models for each prediction horizon.

        Args:
            X: Feature matrix [num_samples, num_features].
            y_dict: Dict mapping "cascade_{h}h" -> binary labels.
            eval_set: Optional (X_val, y_val_dict) for early stopping.
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. pip install xgboost")
            return

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        for horizon_key, y in y_dict.items():
            logger.info(
                f"Training XGBoost for {horizon_key} "
                f"(pos_rate: {y.mean():.4f})"
            )

            # Handle class imbalance
            pos_count = y.sum()
            neg_count = len(y) - pos_count
            scale_pos = neg_count / max(pos_count, 1)

            params = {**self.xgb_params, "scale_pos_weight": scale_pos}
            model = xgb.XGBClassifier(**params)

            fit_kwargs = {}
            if eval_set is not None:
                X_val, y_val_dict = eval_set
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
                fit_kwargs["eval_set"] = [(X_val, y_val_dict[horizon_key])]
                fit_kwargs["verbose"] = False

            if "verbose" not in fit_kwargs:
                fit_kwargs["verbose"] = False
            model.fit(X, y, **fit_kwargs)
            self.models[horizon_key] = model
            logger.info(f"  Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Generate cascade probability predictions.

        Args:
            X: Feature matrix [num_samples, num_features].

        Returns:
            Dict mapping horizon key -> probability array.
        """
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        predictions = {}
        for horizon_key, model in self.models.items():
            predictions[horizon_key] = model.predict_proba(X)[:, 1]
        return predictions

    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """Get feature importance for each horizon model."""
        importances = {}
        for horizon_key, model in self.models.items():
            importances[horizon_key] = model.feature_importances_
        return importances
