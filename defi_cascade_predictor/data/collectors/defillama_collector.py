"""
DefiLlama API collector for protocol TVL, yields, and stablecoin data.
Free API, no authentication required.
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger


BASE_URL = "https://api.llama.fi"
YIELDS_URL = "https://yields.llama.fi"
STABLECOINS_URL = "https://stablecoins.llama.fi"


class DefiLlamaCollector:
    """Collects DeFi protocol data from DefiLlama's free public API."""

    def __init__(self, raw_dir: str = "data/raw", rate_limit: float = 0.5):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Rate-limited GET request with retry logic."""
        for attempt in range(3):
            try:
                time.sleep(self.rate_limit)
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        raise ConnectionError(f"Failed to fetch {url} after 3 attempts")

    def collect_protocol_tvl(self, protocol_slug: str) -> pd.DataFrame:
        """Collect historical TVL for a protocol."""
        logger.info(f"Collecting TVL for {protocol_slug}")
        data = self._get(f"{BASE_URL}/protocol/{protocol_slug}")

        # Extract chain-level TVL breakdown
        records = []
        chain_tvls = data.get("chainTvls", {})
        for chain, chain_data in chain_tvls.items():
            if "tvl" in chain_data:
                for point in chain_data["tvl"]:
                    records.append({
                        "protocol": protocol_slug,
                        "chain": chain,
                        "date": datetime.fromtimestamp(point["date"]),
                        "tvl_usd": point["totalLiquidityUSD"],
                    })

        # Extract aggregate TVL
        aggregate_tvl = data.get("tvl", [])
        for point in aggregate_tvl:
            records.append({
                "protocol": protocol_slug,
                "chain": "aggregate",
                "date": datetime.fromtimestamp(point["date"]),
                "tvl_usd": point["totalLiquidityUSD"],
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.drop_duplicates(subset=["protocol", "chain", "date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def collect_all_protocols_tvl(self, protocol_slugs: list[str]) -> pd.DataFrame:
        """Collect TVL for all specified protocols."""
        all_dfs = []
        for slug in protocol_slugs:
            try:
                df = self.collect_protocol_tvl(slug)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to collect TVL for {slug}: {e}")
        result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        if not result.empty:
            result.to_parquet(self.raw_dir / "protocol_tvl.parquet", index=False)
        return result

    def collect_yields(self) -> pd.DataFrame:
        """Collect current yield/APY data across all DeFi pools."""
        logger.info("Collecting yield data from DefiLlama")
        data = self._get(f"{YIELDS_URL}/pools")
        df = pd.DataFrame(data.get("data", []))

        if not df.empty:
            # Keep relevant columns
            cols = [
                "pool", "chain", "project", "symbol", "tvlUsd",
                "apyBase", "apyReward", "apy", "apyMean30d",
                "volumeUsd1d", "volumeUsd7d", "ilRisk",
                "stablecoin", "exposure",
            ]
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols]
            df.to_parquet(self.raw_dir / "defi_yields.parquet", index=False)

        return df

    def collect_stablecoin_data(self) -> pd.DataFrame:
        """Collect stablecoin market cap and peg data."""
        logger.info("Collecting stablecoin data")
        data = self._get(f"{STABLECOINS_URL}/stablecoins?includePrices=true")

        records = []
        for coin in data.get("peggedAssets", []):
            records.append({
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "peg_type": coin.get("pegType"),
                "peg_mechanism": coin.get("pegMechanism"),
                "circulating": coin.get("circulating", {}).get("peggedUSD", 0),
                "price": coin.get("price"),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_parquet(self.raw_dir / "stablecoins.parquet", index=False)
        return df

    def collect_stablecoin_history(self, stablecoin_id: int) -> pd.DataFrame:
        """Collect historical peg data for a stablecoin."""
        logger.info(f"Collecting stablecoin history for ID {stablecoin_id}")
        data = self._get(
            f"{STABLECOINS_URL}/stablecoincharts/all",
            params={"stablecoin": stablecoin_id},
        )

        records = []
        for point in data:
            records.append({
                "date": datetime.fromtimestamp(point["date"]),
                "circulating": point.get("totalCirculating", {}).get("peggedUSD", 0),
                "minted": point.get("totalMintedToCirculation", {}).get("peggedUSD", 0),
                "bridged": point.get("totalBridgedToChain", {}).get("peggedUSD", 0),
            })

        return pd.DataFrame(records)

    def collect_protocol_fees(self, protocol_slug: str) -> pd.DataFrame:
        """Collect protocol fee/revenue data."""
        logger.info(f"Collecting fees for {protocol_slug}")
        try:
            data = self._get(
                f"{BASE_URL}/summary/fees/{protocol_slug}",
                params={"dataType": "dailyFees"},
            )
            records = []
            for point in data.get("totalDataChart", []):
                records.append({
                    "protocol": protocol_slug,
                    "date": datetime.fromtimestamp(point[0]),
                    "daily_fees_usd": point[1],
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"Could not collect fees for {protocol_slug}: {e}")
            return pd.DataFrame()

    def collect_hack_data(self) -> pd.DataFrame:
        """Collect DeFi hack/exploit database."""
        logger.info("Collecting hack data")
        data = self._get(f"{BASE_URL}/hacks")

        records = []
        for hack in data:
            records.append({
                "name": hack.get("name"),
                "date": hack.get("date"),
                "amount_usd": hack.get("amount", 0),
                "chain": ",".join(hack.get("chains", [])),
                "classification": hack.get("classification"),
                "technique": hack.get("technique"),
                "target_type": hack.get("target_type"),
                "bridge_hack": hack.get("bridge_hack", False),
                "returned_funds": hack.get("returnedFund", 0),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(self.raw_dir / "defi_hacks.parquet", index=False)
        return df

    def collect_chain_tvl_history(self) -> pd.DataFrame:
        """Collect aggregate TVL history across all chains."""
        logger.info("Collecting chain TVL history")
        data = self._get(f"{BASE_URL}/v2/historicalChainTvl")
        records = []
        for point in data:
            records.append({
                "date": datetime.fromtimestamp(point["date"]),
                "tvl_usd": point.get("tvl", 0),
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df.to_parquet(self.raw_dir / "chain_tvl_history.parquet", index=False)
        return df
