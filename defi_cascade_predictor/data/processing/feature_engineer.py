"""
Feature engineering for DeFi composability graph nodes and edges.

Computes protocol-level features from raw data and organizes them into
feature groups for ablation studies.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Engineers node and edge features for the composability graph.

    Feature groups (for ablation studies):
      - tvl_features: TVL level, change, drawdown, rank
      - price_features: token prices, returns, volatility
      - liquidity_features: utilization, borrow rates, supply rates
      - network_features: graph centrality, degree, clustering
      - macro_features: fed funds rate, VIX, DXY, yield curve
      - temporal_features: day-of-week, rolling stats, trend
    """

    FEATURE_GROUPS = {
        "tvl_features": [
            "tvl_usd", "tvl_change_1d", "tvl_change_7d", "tvl_change_30d",
            "tvl_drawdown", "tvl_rank", "tvl_zscore", "tvl_ma_ratio",
        ],
        "price_features": [
            "token_price", "price_return_1d", "price_return_7d",
            "price_return_30d", "volatility_7d", "volatility_30d",
            "volume_usd", "volume_ratio", "drawdown",
        ],
        "liquidity_features": [
            "utilization_rate", "borrow_rate", "supply_rate",
            "total_supply", "total_borrow", "borrow_supply_ratio",
            "rate_spread", "rate_change_7d",
        ],
        "network_features": [
            "degree_centrality", "betweenness_centrality",
            "eigenvector_centrality", "clustering_coeff",
            "pagerank", "num_shared_collaterals",
        ],
        "macro_features": [
            "fed_funds_rate", "treasury_10y", "vix", "dollar_index",
            "yield_curve_slope", "real_rate", "vix_change_7d",
            "dxy_return_7d", "sp500_return_7d",
        ],
        "temporal_features": [
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
            "days_since_last_cascade", "cascade_frequency_90d",
        ],
    }

    def __init__(self, protocol_names: list[str]):
        self.protocol_names = protocol_names
        self.protocol_to_idx = {n: i for i, n in enumerate(protocol_names)}
        self.feature_names: list[str] = []
        self._build_feature_list()

    def _build_feature_list(self):
        """Build ordered list of all feature names."""
        self.feature_names = []
        for group_features in self.FEATURE_GROUPS.values():
            self.feature_names.extend(group_features)
        logger.info(f"Total features per node: {len(self.feature_names)}")

    def get_feature_dim(self) -> int:
        """Return total feature dimension."""
        return len(self.feature_names)

    def get_feature_group_indices(self, group_name: str) -> list[int]:
        """Get feature indices for a specific group (for ablation)."""
        group_features = self.FEATURE_GROUPS.get(group_name, [])
        return [
            self.feature_names.index(f)
            for f in group_features
            if f in self.feature_names
        ]

    def compute_tvl_features(
        self,
        tvl_df: pd.DataFrame,
        date: datetime,
    ) -> dict[str, np.ndarray]:
        """Compute TVL-based features for all protocols at a given date.

        Args:
            tvl_df: DataFrame with [protocol, date, tvl_usd].
            date: Target date.

        Returns:
            Dict mapping protocol_name -> feature array.
        """
        features = {}
        date = pd.Timestamp(date)

        for protocol in self.protocol_names:
            proto_df = tvl_df[
                (tvl_df["protocol"] == protocol)
                & (tvl_df["chain"] == "aggregate")
                & (tvl_df["date"] <= date)
            ].sort_values("date")

            if proto_df.empty:
                features[protocol] = np.zeros(
                    len(self.FEATURE_GROUPS["tvl_features"])
                )
                continue

            current_tvl = proto_df["tvl_usd"].iloc[-1]
            tvl_series = proto_df["tvl_usd"]

            feat = {
                "tvl_usd": np.log1p(current_tvl),  # log-scale
                "tvl_change_1d": self._safe_pct_change(tvl_series, 1),
                "tvl_change_7d": self._safe_pct_change(tvl_series, 7),
                "tvl_change_30d": self._safe_pct_change(tvl_series, 30),
                "tvl_drawdown": self._compute_drawdown(tvl_series),
                "tvl_rank": 0.0,  # computed after all protocols
                "tvl_zscore": self._compute_zscore(tvl_series, 90),
                "tvl_ma_ratio": self._compute_ma_ratio(tvl_series, 30),
            }
            features[protocol] = np.array(
                [feat.get(f, 0.0) for f in self.FEATURE_GROUPS["tvl_features"]]
            )

        # Compute TVL rank
        tvl_values = {
            p: features[p][0] for p in self.protocol_names  # tvl_usd is index 0
        }
        sorted_protos = sorted(tvl_values, key=tvl_values.get, reverse=True)
        for rank, proto in enumerate(sorted_protos):
            features[proto][5] = rank / max(len(sorted_protos) - 1, 1)

        return features

    def compute_price_features(
        self,
        price_df: pd.DataFrame,
        date: datetime,
        protocol_token_map: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Compute price-based features for each protocol's governance token."""
        features = {}
        date = pd.Timestamp(date)
        n_feat = len(self.FEATURE_GROUPS["price_features"])

        for protocol in self.protocol_names:
            token_id = protocol_token_map.get(protocol)
            if token_id is None:
                features[protocol] = np.zeros(n_feat)
                continue

            token_df = price_df[
                (price_df["token"] == token_id) & (price_df["date"] <= date)
            ].sort_values("date")

            if token_df.empty:
                features[protocol] = np.zeros(n_feat)
                continue

            price = token_df["price_usd"].iloc[-1]
            prices = token_df["price_usd"]
            volumes = token_df["volume_usd"] if "volume_usd" in token_df.columns else pd.Series([0.0])
            returns = np.log(prices / prices.shift(1)).dropna()

            feat = {
                "token_price": np.log1p(price),
                "price_return_1d": self._safe_pct_change(prices, 1),
                "price_return_7d": self._safe_pct_change(prices, 7),
                "price_return_30d": self._safe_pct_change(prices, 30),
                "volatility_7d": returns.tail(7).std() if len(returns) >= 7 else 0,
                "volatility_30d": returns.tail(30).std() if len(returns) >= 30 else 0,
                "volume_usd": np.log1p(volumes.iloc[-1]) if len(volumes) > 0 else 0,
                "volume_ratio": self._compute_ma_ratio(volumes, 7),
                "drawdown": self._compute_drawdown(prices),
            }
            features[protocol] = np.array(
                [feat.get(f, 0.0) for f in self.FEATURE_GROUPS["price_features"]]
            )

        return features

    def compute_liquidity_features(
        self,
        lending_data: pd.DataFrame,
        date: datetime,
    ) -> dict[str, np.ndarray]:
        """Compute lending/liquidity features for lending protocols."""
        features = {}
        n_feat = len(self.FEATURE_GROUPS["liquidity_features"])

        for protocol in self.protocol_names:
            proto_data = lending_data[lending_data["protocol"].str.contains(
                protocol.split("-")[0], case=False, na=False
            )]

            if proto_data.empty:
                features[protocol] = np.zeros(n_feat)
                continue

            # Aggregate across all assets
            avg_util = proto_data["utilization"].mean()
            avg_borrow = proto_data["borrow_rate_variable"].mean()
            avg_supply = proto_data["supply_rate"].mean()
            total_supply = proto_data["total_supply"].sum()
            total_borrow = proto_data["total_borrow"].sum()

            feat = {
                "utilization_rate": avg_util,
                "borrow_rate": avg_borrow,
                "supply_rate": avg_supply,
                "total_supply": np.log1p(total_supply),
                "total_borrow": np.log1p(total_borrow),
                "borrow_supply_ratio": (
                    total_borrow / total_supply if total_supply > 0 else 0
                ),
                "rate_spread": avg_borrow - avg_supply,
                "rate_change_7d": 0.0,  # would need historical rates
            }
            features[protocol] = np.array(
                [feat.get(f, 0.0) for f in self.FEATURE_GROUPS["liquidity_features"]]
            )

        return features

    def compute_network_features(
        self, adjacency_matrix: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute graph-theoretic features from the adjacency matrix."""
        import networkx as nx
        from .graph_constructor import ComposabilityGraphConstructor

        n_feat = len(self.FEATURE_GROUPS["network_features"])
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

        degree_cent = nx.degree_centrality(G)
        try:
            between_cent = nx.betweenness_centrality(G, weight="weight")
        except Exception:
            between_cent = {i: 0.0 for i in range(len(self.protocol_names))}
        try:
            eigen_cent = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eigen_cent = {i: 0.0 for i in range(len(self.protocol_names))}
        try:
            clustering = nx.clustering(G.to_undirected(), weight="weight")
        except Exception:
            clustering = {i: 0.0 for i in range(len(self.protocol_names))}
        try:
            pagerank = nx.pagerank(G, weight="weight")
        except Exception:
            pagerank = {i: 1.0 / len(self.protocol_names)
                        for i in range(len(self.protocol_names))}

        features = {}
        for i, protocol in enumerate(self.protocol_names):
            n_collaterals = sum(
                1 for token, protos in
                ComposabilityGraphConstructor.SHARED_COLLATERAL_MAP.items()
                if protocol in protos
            )

            feat = {
                "degree_centrality": degree_cent.get(i, 0),
                "betweenness_centrality": between_cent.get(i, 0),
                "eigenvector_centrality": eigen_cent.get(i, 0),
                "clustering_coeff": clustering.get(i, 0),
                "pagerank": pagerank.get(i, 0),
                "num_shared_collaterals": n_collaterals / 10.0,
            }
            features[protocol] = np.array(
                [feat.get(f, 0.0) for f in self.FEATURE_GROUPS["network_features"]]
            )

        return features

    def compute_macro_features_for_date(
        self, macro_df: pd.DataFrame, date: datetime
    ) -> np.ndarray:
        """Extract macro features for a specific date (shared across all nodes)."""
        date = pd.Timestamp(date)
        n_feat = len(self.FEATURE_GROUPS["macro_features"])

        if macro_df.empty:
            return np.zeros(n_feat)

        # Find closest date
        if hasattr(macro_df.index, 'get_indexer'):
            idx = macro_df.index.get_indexer([date], method="ffill")[0]
            if idx < 0:
                return np.zeros(n_feat)
            row = macro_df.iloc[idx]
        else:
            filtered = macro_df[macro_df.index <= date]
            if filtered.empty:
                return np.zeros(n_feat)
            row = filtered.iloc[-1]

        feature_cols = self.FEATURE_GROUPS["macro_features"]
        values = []
        for col in feature_cols:
            if col in row.index:
                val = row[col]
                values.append(0.0 if pd.isna(val) else float(val))
            else:
                values.append(0.0)

        return np.array(values)

    def compute_temporal_features(
        self,
        date: datetime,
        cascade_labels: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Compute temporal encoding features for a date."""
        date = pd.Timestamp(date)
        dow = date.dayofweek
        month = date.month

        feat = {
            "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
            "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "days_since_last_cascade": 365.0,  # default
            "cascade_frequency_90d": 0.0,
        }

        if cascade_labels is not None and not cascade_labels.empty:
            past = cascade_labels[
                (cascade_labels["date"] < date)
                & (cascade_labels["cascade_active"] == 1)
            ]
            if not past.empty:
                last_cascade = past["date"].max()
                feat["days_since_last_cascade"] = (date - last_cascade).days
                recent = past[past["date"] >= date - pd.Timedelta(days=90)]
                feat["cascade_frequency_90d"] = len(recent)

        # Normalize
        feat["days_since_last_cascade"] = min(
            feat["days_since_last_cascade"] / 365.0, 1.0
        )
        feat["cascade_frequency_90d"] = min(
            feat["cascade_frequency_90d"] / 30.0, 1.0
        )

        return np.array(
            [feat.get(f, 0.0) for f in self.FEATURE_GROUPS["temporal_features"]]
        )

    def build_node_features(
        self,
        date: datetime,
        tvl_features: dict[str, np.ndarray],
        price_features: dict[str, np.ndarray],
        liquidity_features: dict[str, np.ndarray],
        network_features: dict[str, np.ndarray],
        macro_features: np.ndarray,
        temporal_features: np.ndarray,
        exclude_groups: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Combine all feature groups into final node feature matrix.

        Args:
            All feature dicts from compute_* methods.
            exclude_groups: Feature groups to zero out (for ablation).

        Returns:
            Array of shape [num_protocols, total_feature_dim].
        """
        exclude_groups = exclude_groups or []
        node_features = []

        for protocol in self.protocol_names:
            parts = []

            # TVL features
            tvl = tvl_features.get(protocol, np.zeros(
                len(self.FEATURE_GROUPS["tvl_features"])
            ))
            if "tvl_features" in exclude_groups:
                tvl = np.zeros_like(tvl)
            parts.append(tvl)

            # Price features
            price = price_features.get(protocol, np.zeros(
                len(self.FEATURE_GROUPS["price_features"])
            ))
            if "price_features" in exclude_groups:
                price = np.zeros_like(price)
            parts.append(price)

            # Liquidity features
            liq = liquidity_features.get(protocol, np.zeros(
                len(self.FEATURE_GROUPS["liquidity_features"])
            ))
            if "liquidity_features" in exclude_groups:
                liq = np.zeros_like(liq)
            parts.append(liq)

            # Network features
            net = network_features.get(protocol, np.zeros(
                len(self.FEATURE_GROUPS["network_features"])
            ))
            if "network_features" in exclude_groups:
                net = np.zeros_like(net)
            parts.append(net)

            # Macro features (shared across all nodes)
            macro = macro_features.copy()
            if "macro_features" in exclude_groups:
                macro = np.zeros_like(macro)
            parts.append(macro)

            # Temporal features (shared across all nodes)
            temporal = temporal_features.copy()
            if "temporal_features" in exclude_groups:
                temporal = np.zeros_like(temporal)
            parts.append(temporal)

            node_features.append(np.concatenate(parts))

        result = np.array(node_features, dtype=np.float32)
        # Replace NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        return result

    # --- Utility methods ---

    @staticmethod
    def _safe_pct_change(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        old = series.iloc[-(periods + 1)]
        new = series.iloc[-1]
        if old == 0:
            return 0.0
        return (new - old) / abs(old)

    @staticmethod
    def _compute_drawdown(series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0
        rolling_max = series.cummax()
        dd = series / rolling_max - 1
        return dd.iloc[-1]

    @staticmethod
    def _compute_zscore(series: pd.Series, window: int) -> float:
        if len(series) < window:
            return 0.0
        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0.0
        return (series.iloc[-1] - mean) / std

    @staticmethod
    def _compute_ma_ratio(series: pd.Series, window: int) -> float:
        if len(series) < window:
            return 1.0
        ma = series.tail(window).mean()
        if ma == 0:
            return 1.0
        return series.iloc[-1] / ma
