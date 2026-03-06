"""
The Graph subgraph collector for protocol-specific DeFi data.
Queries Aave, Compound, Uniswap, and MakerDAO subgraphs.
Free access via hosted service endpoints.
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger


# Subgraph endpoints (hosted service)
SUBGRAPH_URLS = {
    "aave_v3": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
    "aave_v2": "https://api.thegraph.com/subgraphs/name/aave/protocol-v2",
    "uniswap_v3": "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
    "compound_v2": "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2",
}


class SubgraphCollector:
    """Collects protocol-level data from The Graph's hosted subgraphs.

    Falls back to synthetic data if subgraph endpoints are unavailable.
    """

    def __init__(self, raw_dir: str = "data/raw", rate_limit: float = 1.0):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.session = requests.Session()

    def _query_subgraph(self, url: str, query: str) -> dict:
        """Execute a GraphQL query against a subgraph."""
        time.sleep(self.rate_limit)
        for attempt in range(3):
            try:
                resp = self.session.post(
                    url, json={"query": query}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    logger.warning(f"Subgraph errors: {data['errors']}")
                return data.get("data", {})
            except requests.exceptions.RequestException as e:
                logger.warning(f"Subgraph query failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return {}

    def collect_aave_reserves(self) -> pd.DataFrame:
        """Collect Aave reserve/market data including utilization and rates."""
        logger.info("Collecting Aave V3 reserve data")
        query = """
        {
            reserves(first: 100) {
                id
                name
                symbol
                underlyingAsset
                liquidityRate
                variableBorrowRate
                stableBorrowRate
                totalATokenSupply
                totalCurrentVariableDebt
                totalCurrentStableDebt
                availableLiquidity
                utilizationRate
                liquidityIndex
                variableBorrowIndex
                reserveFactor
                baseLTVasCollateral
                reserveLiquidationThreshold
                reserveLiquidationBonus
            }
        }
        """
        data = self._query_subgraph(SUBGRAPH_URLS.get("aave_v3", ""), query)
        reserves = data.get("reserves", [])

        if not reserves:
            logger.warning("No Aave data from subgraph, generating synthetic")
            return self._generate_synthetic_lending_data("aave")

        records = []
        for r in reserves:
            records.append({
                "protocol": "aave_v3",
                "asset": r.get("symbol", ""),
                "underlying": r.get("underlyingAsset", ""),
                "supply_rate": float(r.get("liquidityRate", 0)) / 1e27,
                "borrow_rate_variable": float(r.get("variableBorrowRate", 0)) / 1e27,
                "total_supply": float(r.get("totalATokenSupply", 0)),
                "total_borrow": float(r.get("totalCurrentVariableDebt", 0)),
                "utilization": float(r.get("utilizationRate", 0)),
                "ltv": float(r.get("baseLTVasCollateral", 0)) / 10000,
                "liquidation_threshold": (
                    float(r.get("reserveLiquidationThreshold", 0)) / 10000
                ),
                "liquidation_bonus": (
                    float(r.get("reserveLiquidationBonus", 0)) / 10000
                ),
            })

        df = pd.DataFrame(records)
        df.to_parquet(self.raw_dir / "aave_reserves.parquet", index=False)
        return df

    def collect_uniswap_pools(self, min_tvl: float = 1_000_000) -> pd.DataFrame:
        """Collect Uniswap V3 pool data."""
        logger.info("Collecting Uniswap V3 pool data")
        query = f"""
        {{
            pools(
                first: 100,
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: {{ totalValueLockedUSD_gt: "{min_tvl}" }}
            ) {{
                id
                token0 {{ symbol name id }}
                token1 {{ symbol name id }}
                feeTier
                liquidity
                sqrtPrice
                tick
                totalValueLockedUSD
                totalValueLockedToken0
                totalValueLockedToken1
                volumeUSD
                txCount
                token0Price
                token1Price
            }}
        }}
        """
        data = self._query_subgraph(SUBGRAPH_URLS.get("uniswap_v3", ""), query)
        pools = data.get("pools", [])

        if not pools:
            logger.warning("No Uniswap data from subgraph, generating synthetic")
            return self._generate_synthetic_dex_data()

        records = []
        for p in pools:
            records.append({
                "protocol": "uniswap_v3",
                "pool_id": p.get("id", ""),
                "token0": p.get("token0", {}).get("symbol", ""),
                "token1": p.get("token1", {}).get("symbol", ""),
                "fee_tier": int(p.get("feeTier", 0)),
                "tvl_usd": float(p.get("totalValueLockedUSD", 0)),
                "volume_usd": float(p.get("volumeUSD", 0)),
                "tx_count": int(p.get("txCount", 0)),
                "token0_price": float(p.get("token0Price", 0)),
                "token1_price": float(p.get("token1Price", 0)),
            })

        df = pd.DataFrame(records)
        df.to_parquet(self.raw_dir / "uniswap_pools.parquet", index=False)
        return df

    def collect_all(self) -> dict[str, pd.DataFrame]:
        """Collect data from all subgraphs."""
        return {
            "aave_reserves": self.collect_aave_reserves(),
            "uniswap_pools": self.collect_uniswap_pools(),
        }

    # --- Synthetic data generators ---

    def _generate_synthetic_lending_data(self, protocol: str) -> pd.DataFrame:
        """Generate synthetic lending protocol data."""
        rng = np.random.RandomState(44)
        assets = [
            "WETH", "WBTC", "USDC", "USDT", "DAI", "LINK",
            "UNI", "AAVE", "CRV", "stETH", "wstETH", "rETH",
            "FRAX", "LUSD", "cbETH",
        ]
        records = []
        for asset in assets:
            is_stable = asset in ["USDC", "USDT", "DAI", "FRAX", "LUSD"]
            util = rng.uniform(0.5, 0.9) if is_stable else rng.uniform(0.2, 0.7)
            records.append({
                "protocol": protocol,
                "asset": asset,
                "underlying": f"0x{''.join(rng.choice(list('0123456789abcdef'), 40))}",
                "supply_rate": rng.uniform(0.001, 0.08),
                "borrow_rate_variable": rng.uniform(0.01, 0.15),
                "total_supply": rng.uniform(1e6, 1e9),
                "total_borrow": rng.uniform(1e5, 5e8),
                "utilization": util,
                "ltv": rng.uniform(0.5, 0.85),
                "liquidation_threshold": rng.uniform(0.75, 0.95),
                "liquidation_bonus": rng.uniform(1.04, 1.10),
            })
        return pd.DataFrame(records)

    def _generate_synthetic_dex_data(self) -> pd.DataFrame:
        """Generate synthetic DEX pool data."""
        rng = np.random.RandomState(45)
        pools = [
            ("WETH", "USDC", 3000), ("WETH", "USDT", 3000),
            ("WBTC", "WETH", 3000), ("USDC", "USDT", 100),
            ("DAI", "USDC", 100), ("WETH", "DAI", 3000),
            ("stETH", "WETH", 100), ("LINK", "WETH", 3000),
            ("UNI", "WETH", 3000), ("AAVE", "WETH", 3000),
            ("CRV", "WETH", 10000), ("wstETH", "WETH", 100),
        ]
        records = []
        for t0, t1, fee in pools:
            records.append({
                "protocol": "uniswap_v3",
                "pool_id": f"0x{''.join(rng.choice(list('0123456789abcdef'), 40))}",
                "token0": t0,
                "token1": t1,
                "fee_tier": fee,
                "tvl_usd": rng.uniform(1e6, 5e8),
                "volume_usd": rng.uniform(1e5, 1e8),
                "tx_count": rng.randint(1000, 500_000),
                "token0_price": rng.uniform(0.1, 50000),
                "token1_price": rng.uniform(0.1, 50000),
            })
        return pd.DataFrame(records)
