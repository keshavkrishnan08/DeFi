"""
CoinGecko API collector for token prices, market data, and correlations.
Free demo tier: 10-30 calls/min, no API key required for basic endpoints.
"""

import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger


BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoCollector:
    """Collects token price and market data from CoinGecko's free API."""

    def __init__(self, raw_dir: str = "data/raw", rate_limit: float = 2.5):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit  # seconds between requests
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Rate-limited GET request."""
        for attempt in range(3):
            try:
                time.sleep(self.rate_limit)
                resp = self.session.get(
                    f"{BASE_URL}/{endpoint}", params=params, timeout=30
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        raise ConnectionError(f"Failed to fetch {endpoint} after 3 attempts")

    # Mapping of DeFi protocol names to their CoinGecko token IDs
    PROTOCOL_TOKEN_MAP = {
        "aave-v3": "aave",
        "aave-v2": "aave",
        "compound-v3": "compound-governance-token",
        "compound-v2": "compound-governance-token",
        "makerdao": "maker",
        "uniswap-v3": "uniswap",
        "uniswap-v2": "uniswap",
        "curve-dex": "curve-dao-token",
        "lido": "lido-dao",
        "rocket-pool": "rocket-pool",
        "convex-finance": "convex-finance",
        "yearn-finance": "yearn-finance",
        "frax": "frax-share",
        "instadapp": "instadapp",
        "morpho": "morpho",
    }

    # Key tokens to track for the DeFi ecosystem
    KEY_TOKENS = [
        "ethereum", "bitcoin", "tether", "usd-coin", "dai",
        "wrapped-bitcoin", "staked-ether", "frax", "rocket-pool-eth",
        "chainlink", "aave", "compound-governance-token", "maker",
        "uniswap", "curve-dao-token", "lido-dao", "convex-finance",
    ]

    def collect_token_price_history(
        self,
        token_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> pd.DataFrame:
        """Collect historical OHLC price data for a token."""
        logger.info(f"Collecting price history for {token_id} ({days} days)")
        data = self._get(
            f"coins/{token_id}/market_chart",
            params={"vs_currency": vs_currency, "days": days, "interval": "daily"},
        )

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        market_caps = data.get("market_caps", [])

        records = []
        for i in range(len(prices)):
            record = {
                "token": token_id,
                "date": datetime.fromtimestamp(prices[i][0] / 1000),
                "price_usd": prices[i][1],
            }
            if i < len(volumes):
                record["volume_usd"] = volumes[i][1]
            if i < len(market_caps):
                record["market_cap_usd"] = market_caps[i][1]
            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.drop_duplicates(subset=["token", "date"])
        return df

    def collect_all_token_prices(
        self, days: int = 1460,  # ~4 years
    ) -> pd.DataFrame:
        """Collect price history for all key DeFi tokens."""
        all_dfs = []
        for token_id in self.KEY_TOKENS:
            try:
                df = self.collect_token_price_history(token_id, days=days)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to collect prices for {token_id}: {e}")
        result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        if not result.empty:
            result.to_parquet(self.raw_dir / "token_prices.parquet", index=False)
        return result

    def collect_global_market_data(self) -> dict:
        """Collect global crypto market statistics."""
        logger.info("Collecting global market data")
        data = self._get("global")
        return data.get("data", {})

    def compute_price_correlation_matrix(
        self,
        price_df: pd.DataFrame,
        window: int = 30,
    ) -> pd.DataFrame:
        """Compute rolling pairwise price correlation matrix."""
        pivot = price_df.pivot_table(
            index="date", columns="token", values="price_usd"
        )
        # Compute log returns
        returns = np.log(pivot / pivot.shift(1)).dropna()
        # Rolling correlation
        corr = returns.rolling(window=window).corr()
        return corr

    def compute_return_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute return-based features for each token."""
        features = []
        for token_id, group in price_df.groupby("token"):
            group = group.sort_values("date").copy()
            group["log_return"] = np.log(
                group["price_usd"] / group["price_usd"].shift(1)
            )
            group["volatility_7d"] = group["log_return"].rolling(7).std()
            group["volatility_30d"] = group["log_return"].rolling(30).std()
            group["return_7d"] = group["price_usd"].pct_change(7)
            group["return_30d"] = group["price_usd"].pct_change(30)
            group["volume_ma_7d"] = group["volume_usd"].rolling(7).mean()
            group["volume_ratio"] = (
                group["volume_usd"] / group["volume_ma_7d"]
            )
            group["drawdown"] = (
                group["price_usd"] / group["price_usd"].cummax() - 1
            )
            features.append(group)

        result = pd.concat(features, ignore_index=True)
        return result
