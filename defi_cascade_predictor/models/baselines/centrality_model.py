"""
Network Centrality baseline.
Predicts cascade risk using graph-theoretic centrality metrics
combined with protocol-level risk indicators.
"""

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional
from loguru import logger


class CentralityModel:
    """Centrality-based cascade predictor.

    Uses network centrality metrics (betweenness, eigenvector, PageRank)
    combined with basic protocol features as input to logistic regression.

    Ablation: measures the value of deep graph learning over simple
    network science metrics.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        prediction_horizons: list[int] = [24, 72, 168, 720],
    ):
        self.adj = adjacency_matrix
        self.prediction_horizons = prediction_horizons
        self.models = {}
        self.scaler = StandardScaler()
        self._compute_centrality_metrics()

    def _compute_centrality_metrics(self):
        """Compute static graph centrality metrics."""
        G = nx.from_numpy_array(self.adj, create_using=nx.DiGraph)
        n = len(self.adj)

        self.centrality = {
            "degree": np.array(
                [nx.degree_centrality(G).get(i, 0) for i in range(n)]
            ),
            "betweenness": np.array(
                [nx.betweenness_centrality(G).get(i, 0) for i in range(n)]
            ),
            "closeness": np.array(
                [nx.closeness_centrality(G).get(i, 0) for i in range(n)]
            ),
            "pagerank": np.array(
                [nx.pagerank(G, weight="weight").get(i, 0) for i in range(n)]
            ),
        }

        try:
            eigen = nx.eigenvector_centrality_numpy(G, weight="weight")
            self.centrality["eigenvector"] = np.array(
                [eigen.get(i, 0) for i in range(n)]
            )
        except Exception:
            self.centrality["eigenvector"] = np.zeros(n)

        try:
            self.centrality["clustering"] = np.array(
                [
                    nx.clustering(G.to_undirected(), weight="weight").get(i, 0)
                    for i in range(n)
                ]
            )
        except Exception:
            self.centrality["clustering"] = np.zeros(n)

        logger.info("Centrality metrics computed")

    def prepare_features(
        self,
        node_features_sequence: list[np.ndarray],
        window: int = 30,
    ) -> np.ndarray:
        """Combine centrality with basic protocol features.

        Args:
            node_features_sequence: List of [num_nodes, feature_dim] arrays.
            window: Lookback window.

        Returns:
            Feature matrix [num_samples, num_features].
        """
        all_features = []
        n = len(self.adj)

        for t in range(window, len(node_features_sequence)):
            features = []

            # Current protocol features (aggregated)
            current = node_features_sequence[t]
            features.extend(current.mean(axis=0))  # mean across protocols
            features.extend(current.std(axis=0))   # dispersion

            # Centrality metrics
            for metric_name, values in self.centrality.items():
                features.append(values.mean())
                features.append(values.std())
                features.append(values.max())

            # Centrality-weighted protocol risk
            current_risk = current.mean(axis=1)  # per-protocol risk score
            for metric_name, values in self.centrality.items():
                # Weighted risk: centrality * current risk
                features.append((values * current_risk).sum())

            # Temporal features
            if t >= window:
                past = np.array(node_features_sequence[t - window:t])
                features.extend(past.mean(axis=(0, 1)))
                changes = np.diff(past, axis=0)
                if changes.shape[0] > 0:
                    features.extend(changes.mean(axis=(0, 1)))
                else:
                    features.extend(np.zeros(current.shape[1]))
            else:
                features.extend(np.zeros(current.shape[1] * 2))

            all_features.append(features)

        return np.array(all_features, dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        y_dict: dict[str, np.ndarray],
    ):
        """Train logistic regression for each horizon.

        Args:
            X: Feature matrix.
            y_dict: Labels per horizon.
        """
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.fit_transform(X)

        for horizon_key, y in y_dict.items():
            logger.info(f"Training centrality model for {horizon_key}")
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_scaled, y)
            self.models[horizon_key] = model

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict cascade probabilities."""
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)
        predictions = {}
        for horizon_key, model in self.models.items():
            predictions[horizon_key] = model.predict_proba(X_scaled)[:, 1]
        return predictions
