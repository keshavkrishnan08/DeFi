"""
DeFi Composability Graph Constructor.

Builds a dynamic heterogeneous graph encoding cross-protocol dependencies:
  - Nodes: protocols, liquidity pools, tokens
  - Edges: shared collateral, liquidity flows, oracle dependencies,
           governance overlap, price correlation, liquidation pathways

This is the core data structure for the Temporal Graph Network.
"""

from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, TemporalData
from loguru import logger


class ComposabilityGraphConstructor:
    """Constructs the DeFi composability graph from collected protocol data.

    The graph captures cross-protocol risk dependencies that enable
    liquidation cascades to propagate through the DeFi ecosystem.
    """

    # Known cross-protocol relationships
    SHARED_COLLATERAL_MAP = {
        # token -> list of protocols that accept it as collateral
        "WETH": ["aave-v3", "aave-v2", "compound-v3", "compound-v2", "makerdao", "balancer"],
        "WBTC": ["aave-v3", "aave-v2", "compound-v2", "makerdao", "balancer"],
        "USDC": ["aave-v3", "aave-v2", "compound-v3", "compound-v2", "balancer"],
        "USDT": ["aave-v3", "aave-v2", "compound-v2"],
        "DAI": ["aave-v3", "aave-v2", "compound-v2", "balancer"],
        "stETH": ["aave-v3", "aave-v2"],
        "wstETH": ["aave-v3", "makerdao"],
        "LINK": ["aave-v3", "aave-v2", "compound-v2"],
        "UNI": ["aave-v3", "aave-v2", "compound-v2"],
        "CRV": ["aave-v3", "aave-v2"],
    }

    ORACLE_DEPENDENCIES = {
        # oracle_source -> protocols using it
        "chainlink_eth_usd": [
            "aave-v3", "aave-v2", "compound-v3", "compound-v2",
            "makerdao", "uniswap-v3",
        ],
        "chainlink_btc_usd": [
            "aave-v3", "aave-v2", "compound-v2", "makerdao",
        ],
        "chainlink_link_usd": ["aave-v3", "aave-v2", "compound-v2"],
        "curve_steth_pool": ["lido", "aave-v3"],
        "uniswap_twap": ["uniswap-v3", "uniswap-v2"],
    }

    LIQUIDITY_PATHWAYS = {
        # (source, target): description
        ("lido", "aave-v3"): "stETH deposited as collateral on Aave",
        ("lido", "curve-dex"): "stETH/ETH pool on Curve",
        ("makerdao", "uniswap-v3"): "DAI traded on Uniswap",
        ("aave-v3", "uniswap-v3"): "Liquidations route through Uniswap",
        ("compound-v3", "uniswap-v3"): "Liquidations route through Uniswap",
        ("curve-dex", "convex-finance"): "Curve LP tokens staked on Convex",
        ("convex-finance", "yearn-finance"): "Convex strategies in Yearn vaults",
        ("balancer", "aave-v3"): "Balancer pools provide liquidity for Aave",
        ("balancer", "curve-dex"): "Balancer and Curve share stablecoin liquidity",
        ("aave-v3", "morpho"): "Morpho optimizes Aave rates",
    }

    GOVERNANCE_TOKEN_OVERLAP = {
        # token -> protocols whose governance token holders overlap significantly
        "CRV": ["curve-dex", "convex-finance", "yearn-finance"],
        "CVX": ["convex-finance", "curve-dex"],
        "AAVE": ["aave-v3", "aave-v2"],
        "COMP": ["compound-v3", "compound-v2"],
        "UNI": ["uniswap-v3", "uniswap-v2"],
        "LDO": ["lido", "curve-dex"],
        "BAL": ["balancer"],
    }

    def __init__(self, protocols: list[dict], edge_types: list[str]):
        """
        Args:
            protocols: List of protocol configs from config.yaml.
            edge_types: List of edge type names to include.
        """
        self.protocols = {p["name"]: p for p in protocols}
        self.protocol_names = [p["name"] for p in protocols]
        self.edge_types = edge_types
        self.protocol_to_idx = {
            name: i for i, name in enumerate(self.protocol_names)
        }
        logger.info(
            f"GraphConstructor initialized: {len(self.protocol_names)} protocols, "
            f"{len(self.edge_types)} edge types"
        )

    def build_static_edges(self) -> dict[str, list[tuple[int, int]]]:
        """Build static edge lists from known cross-protocol relationships.

        Returns:
            Dict mapping edge_type -> list of (src_idx, dst_idx) tuples.
        """
        edges = {etype: [] for etype in self.edge_types}

        # 1. Shared collateral edges
        if "shared_collateral" in self.edge_types:
            for token, protos in self.SHARED_COLLATERAL_MAP.items():
                valid = [p for p in protos if p in self.protocol_to_idx]
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        src = self.protocol_to_idx[valid[i]]
                        dst = self.protocol_to_idx[valid[j]]
                        edges["shared_collateral"].append((src, dst))
                        edges["shared_collateral"].append((dst, src))

        # 2. Oracle dependency edges
        if "oracle_dependency" in self.edge_types:
            for oracle, protos in self.ORACLE_DEPENDENCIES.items():
                valid = [p for p in protos if p in self.protocol_to_idx]
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        src = self.protocol_to_idx[valid[i]]
                        dst = self.protocol_to_idx[valid[j]]
                        edges["oracle_dependency"].append((src, dst))
                        edges["oracle_dependency"].append((dst, src))

        # 3. Liquidity flow edges (directed)
        if "liquidity_flow" in self.edge_types:
            for (src_name, dst_name) in self.LIQUIDITY_PATHWAYS:
                if (
                    src_name in self.protocol_to_idx
                    and dst_name in self.protocol_to_idx
                ):
                    src = self.protocol_to_idx[src_name]
                    dst = self.protocol_to_idx[dst_name]
                    edges["liquidity_flow"].append((src, dst))
                    edges["liquidity_flow"].append((dst, src))

        # 4. Liquidation pathway edges
        if "liquidation_pathway" in self.edge_types:
            # Lending protocols -> DEX (liquidation routes)
            lending = [
                p for p in self.protocol_names
                if self.protocols[p]["type"] in ("lending", "cdp")
            ]
            dexes = [
                p for p in self.protocol_names
                if self.protocols[p]["type"] == "dex"
            ]
            for lend in lending:
                for dex in dexes:
                    src = self.protocol_to_idx[lend]
                    dst = self.protocol_to_idx[dex]
                    edges["liquidation_pathway"].append((src, dst))

        # 5. Governance overlap edges
        if "governance_overlap" in self.edge_types:
            for token, protos in self.GOVERNANCE_TOKEN_OVERLAP.items():
                valid = [p for p in protos if p in self.protocol_to_idx]
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        src = self.protocol_to_idx[valid[i]]
                        dst = self.protocol_to_idx[valid[j]]
                        edges["governance_overlap"].append((src, dst))
                        edges["governance_overlap"].append((dst, src))

        # Deduplicate
        for etype in edges:
            edges[etype] = list(set(edges[etype]))

        edge_counts = {k: len(v) for k, v in edges.items() if v}
        logger.info(f"Static edges built: {edge_counts}")
        return edges

    def compute_price_correlation_edges(
        self,
        price_features: pd.DataFrame,
        threshold: float = 0.7,
        window: int = 30,
        date: Optional[datetime] = None,
    ) -> list[tuple[int, int, float]]:
        """Compute dynamic price correlation edges between protocol tokens.

        Args:
            price_features: Token price DataFrame with log returns.
            threshold: Correlation threshold for creating an edge.
            window: Rolling window in days.
            date: Date for which to compute correlations.

        Returns:
            List of (src_idx, dst_idx, correlation_weight) tuples.
        """
        from .feature_engineer import FeatureEngineer

        edges = []
        token_map = {
            "aave": "aave-v3",
            "compound-governance-token": "compound-v3",
            "maker": "makerdao",
            "uniswap": "uniswap-v3",
            "curve-dao-token": "curve-dex",
            "lido-dao": "lido",
            "rocket-pool": "rocket-pool",
            "convex-finance": "convex-finance",
            "yearn-finance": "yearn-finance",
        }

        if "log_return" not in price_features.columns:
            return edges

        # Pivot to get returns by token
        if date is not None:
            mask = price_features["date"] <= date
            df = price_features[mask].tail(window * 20)
        else:
            df = price_features

        pivot = df.pivot_table(
            index="date", columns="token", values="log_return"
        ).dropna(axis=1, how="all")

        if pivot.shape[1] < 2:
            return edges

        corr_matrix = pivot.corr()

        for t1 in corr_matrix.columns:
            for t2 in corr_matrix.columns:
                if t1 >= t2:
                    continue
                corr_val = corr_matrix.loc[t1, t2]
                if abs(corr_val) >= threshold:
                    p1 = token_map.get(t1)
                    p2 = token_map.get(t2)
                    if (
                        p1 in self.protocol_to_idx
                        and p2 in self.protocol_to_idx
                    ):
                        src = self.protocol_to_idx[p1]
                        dst = self.protocol_to_idx[p2]
                        edges.append((src, dst, abs(corr_val)))
                        edges.append((dst, src, abs(corr_val)))

        return edges

    def build_snapshot(
        self,
        date: datetime,
        node_features: np.ndarray,
        static_edges: dict[str, list[tuple[int, int]]],
        dynamic_corr_edges: Optional[list[tuple[int, int, float]]] = None,
    ) -> HeteroData:
        """Build a single temporal graph snapshot.

        Args:
            date: Timestamp for this snapshot.
            node_features: Array of shape [num_protocols, feature_dim].
            static_edges: Pre-computed static edge dict.
            dynamic_corr_edges: Optional dynamic correlation edges.

        Returns:
            PyG HeteroData object for this snapshot.
        """
        data = HeteroData()
        num_nodes = len(self.protocol_names)

        # Node features
        data["protocol"].x = torch.tensor(node_features, dtype=torch.float32)
        data["protocol"].num_nodes = num_nodes

        # Static edges
        for etype, edge_list in static_edges.items():
            if edge_list:
                src = [e[0] for e in edge_list]
                dst = [e[1] for e in edge_list]
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                data["protocol", etype, "protocol"].edge_index = edge_index

        # Dynamic correlation edges
        if dynamic_corr_edges and "price_correlation" in self.edge_types:
            src = [e[0] for e in dynamic_corr_edges]
            dst = [e[1] for e in dynamic_corr_edges]
            weights = [e[2] for e in dynamic_corr_edges]
            if src:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                data[
                    "protocol", "price_correlation", "protocol"
                ].edge_index = edge_index
                data[
                    "protocol", "price_correlation", "protocol"
                ].edge_attr = edge_attr

        # Metadata
        data.timestamp = date

        return data

    def build_temporal_sequence(
        self,
        node_features_series: dict[datetime, np.ndarray],
        price_features: Optional[pd.DataFrame] = None,
        corr_threshold: float = 0.7,
    ) -> list[HeteroData]:
        """Build a sequence of temporal graph snapshots.

        Args:
            node_features_series: Dict mapping date -> node feature array.
            price_features: Optional price data for dynamic edges.
            corr_threshold: Threshold for correlation edges.

        Returns:
            List of HeteroData snapshots sorted by time.
        """
        static_edges = self.build_static_edges()

        snapshots = []
        sorted_dates = sorted(node_features_series.keys())

        for date in sorted_dates:
            features = node_features_series[date]

            # Compute dynamic correlation edges if price data available
            dyn_edges = None
            if price_features is not None:
                dyn_edges = self.compute_price_correlation_edges(
                    price_features,
                    threshold=corr_threshold,
                    date=date,
                )

            snapshot = self.build_snapshot(
                date=date,
                node_features=features,
                static_edges=static_edges,
                dynamic_corr_edges=dyn_edges,
            )
            snapshots.append(snapshot)

        logger.info(f"Built temporal sequence of {len(snapshots)} graph snapshots")
        return snapshots

    def build_homogeneous_edge_index(
        self,
        static_edges: Optional[dict] = None,
    ) -> torch.Tensor:
        """Build a single combined edge index for homogeneous GNN baselines.

        Merges all edge types into one edge index.
        """
        if static_edges is None:
            static_edges = self.build_static_edges()

        all_edges = set()
        for edge_list in static_edges.values():
            for src, dst in edge_list:
                all_edges.add((src, dst))

        if not all_edges:
            return torch.zeros((2, 0), dtype=torch.long)

        src = [e[0] for e in all_edges]
        dst = [e[1] for e in all_edges]
        return torch.tensor([src, dst], dtype=torch.long)

    def get_adjacency_matrix(
        self, static_edges: Optional[dict] = None
    ) -> np.ndarray:
        """Build a weighted adjacency matrix for network analysis baselines."""
        n = len(self.protocol_names)
        adj = np.zeros((n, n))

        if static_edges is None:
            static_edges = self.build_static_edges()

        # Weight by edge type importance
        edge_weights = {
            "shared_collateral": 1.0,
            "liquidity_flow": 0.8,
            "oracle_dependency": 0.9,
            "governance_overlap": 0.5,
            "price_correlation": 0.7,
            "liquidation_pathway": 1.0,
        }

        for etype, edge_list in static_edges.items():
            w = edge_weights.get(etype, 0.5)
            for src, dst in edge_list:
                adj[src, dst] = max(adj[src, dst], w)

        return adj
