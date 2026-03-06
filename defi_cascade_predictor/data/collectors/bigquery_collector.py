"""
Google BigQuery collector for Ethereum on-chain data.
Requires google-cloud-bigquery and authentication.
Falls back to synthetic data generation if credentials unavailable.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger


class BigQueryCollector:
    """Collects Ethereum on-chain data from Google BigQuery public datasets.

    Uses the free public dataset: bigquery-public-data.crypto_ethereum
    Free tier: 1TB/month of query processing.
    Falls back to synthetic data if credentials are not available.
    """

    ETHEREUM_DATASET = "bigquery-public-data.crypto_ethereum"

    # SQL queries for DeFi-relevant on-chain metrics
    QUERIES = {
        "daily_gas_metrics": """
            SELECT
                DATE(block_timestamp) as date,
                AVG(gas_price) / 1e9 as avg_gas_gwei,
                MAX(gas_price) / 1e9 as max_gas_gwei,
                SUM(receipt_gas_used) as total_gas_used,
                COUNT(*) as tx_count,
                COUNT(DISTINCT from_address) as unique_senders,
                SUM(CAST(value AS FLOAT64)) / 1e18 as total_eth_transferred
            FROM `{dataset}.transactions`
            WHERE DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY date
            ORDER BY date
        """,
        "daily_contract_activity": """
            SELECT
                DATE(block_timestamp) as date,
                COUNT(*) as contract_calls,
                COUNT(DISTINCT to_address) as unique_contracts,
                SUM(receipt_gas_used) as contract_gas_used
            FROM `{dataset}.transactions`
            WHERE to_address IS NOT NULL
                AND input != '0x'
                AND DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY date
            ORDER BY date
        """,
        "daily_token_transfers": """
            SELECT
                DATE(block_timestamp) as date,
                token_address,
                COUNT(*) as transfer_count,
                COUNT(DISTINCT from_address) as unique_senders,
                COUNT(DISTINCT to_address) as unique_receivers,
                SUM(SAFE_CAST(value AS FLOAT64)) as total_value_raw
            FROM `{dataset}.token_transfers`
            WHERE DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY date, token_address
            HAVING transfer_count > 100
            ORDER BY date, transfer_count DESC
        """,
        "large_liquidation_events": """
            SELECT
                DATE(block_timestamp) as date,
                block_number,
                transaction_hash,
                from_address,
                to_address,
                receipt_gas_used,
                CAST(value AS FLOAT64) / 1e18 as eth_value
            FROM `{dataset}.transactions`
            WHERE receipt_status = 1
                AND CAST(value AS FLOAT64) / 1e18 > 100
                AND DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY eth_value DESC
            LIMIT 10000
        """,
    }

    def __init__(self, raw_dir: str = "data/raw", project_id: Optional[str] = None):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        self.client = None

        try:
            from google.cloud import bigquery
            self.client = bigquery.Client(project=self.project_id)
            logger.info("BigQuery client initialized")
        except Exception as e:
            logger.warning(f"BigQuery unavailable: {e}. Will use synthetic data.")

    def run_query(self, query: str) -> pd.DataFrame:
        """Execute a BigQuery SQL query and return as DataFrame."""
        if self.client is None:
            raise RuntimeError("BigQuery client not available")
        logger.info(f"Running BigQuery query ({len(query)} chars)")
        return self.client.query(query).to_dataframe()

    def collect_daily_gas_metrics(
        self, start_date: str = "2021-01-01", end_date: str = "2025-12-31"
    ) -> pd.DataFrame:
        """Collect daily gas price and transaction metrics."""
        if self.client is not None:
            try:
                query = self.QUERIES["daily_gas_metrics"].format(
                    dataset=self.ETHEREUM_DATASET,
                    start_date=start_date,
                    end_date=end_date,
                )
                df = self.run_query(query)
                df.to_parquet(self.raw_dir / "daily_gas_metrics.parquet", index=False)
                return df
            except Exception as e:
                logger.warning(f"BigQuery failed: {e}. Generating synthetic data.")

        return self._generate_synthetic_gas_metrics(start_date, end_date)

    def collect_daily_contract_activity(
        self, start_date: str = "2021-01-01", end_date: str = "2025-12-31"
    ) -> pd.DataFrame:
        """Collect daily smart contract interaction metrics."""
        if self.client is not None:
            try:
                query = self.QUERIES["daily_contract_activity"].format(
                    dataset=self.ETHEREUM_DATASET,
                    start_date=start_date,
                    end_date=end_date,
                )
                df = self.run_query(query)
                df.to_parquet(
                    self.raw_dir / "daily_contract_activity.parquet", index=False
                )
                return df
            except Exception as e:
                logger.warning(f"BigQuery failed: {e}. Generating synthetic data.")

        return self._generate_synthetic_contract_activity(start_date, end_date)

    def collect_all(
        self, start_date: str = "2021-01-01", end_date: str = "2025-12-31"
    ) -> dict[str, pd.DataFrame]:
        """Collect all on-chain metrics."""
        results = {
            "gas_metrics": self.collect_daily_gas_metrics(start_date, end_date),
            "contract_activity": self.collect_daily_contract_activity(
                start_date, end_date
            ),
        }
        return results

    # --- Synthetic data generators for development ---

    def _generate_synthetic_gas_metrics(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate realistic synthetic gas metrics."""
        logger.info("Generating synthetic gas metrics")
        rng = np.random.RandomState(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)
        t = np.linspace(0, 1, n)

        # Base gas price with spikes at cascade events
        base_gas = 30 + 20 * np.sin(4 * np.pi * t) + rng.exponential(5, n)
        for spike_pos in [0.28, 0.35, 0.45, 0.62]:
            idx = int(spike_pos * n)
            if idx < n:
                spike_width = rng.randint(3, 8)
                for j in range(max(0, idx - spike_width), min(n, idx + spike_width)):
                    base_gas[j] += rng.uniform(50, 300)

        # Transaction count correlates with gas
        tx_count = (
            1_200_000 + 300_000 * np.sin(2 * np.pi * t)
            + rng.normal(0, 50_000, n)
        )
        tx_count = np.clip(tx_count, 800_000, 2_000_000)

        df = pd.DataFrame({
            "date": dates,
            "avg_gas_gwei": np.clip(base_gas, 5, 500),
            "max_gas_gwei": np.clip(base_gas * rng.uniform(2, 5, n), 10, 5000),
            "total_gas_used": tx_count * rng.uniform(60_000, 120_000, n),
            "tx_count": tx_count.astype(int),
            "unique_senders": (tx_count * rng.uniform(0.3, 0.5, n)).astype(int),
            "total_eth_transferred": rng.exponential(50_000, n),
        })
        df.to_parquet(self.raw_dir / "daily_gas_metrics.parquet", index=False)
        return df

    def _generate_synthetic_contract_activity(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate realistic synthetic contract activity metrics."""
        logger.info("Generating synthetic contract activity")
        rng = np.random.RandomState(43)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)
        t = np.linspace(0, 1, n)

        contract_calls = (
            800_000 + 200_000 * t + rng.normal(0, 30_000, n)
        )

        df = pd.DataFrame({
            "date": dates,
            "contract_calls": np.clip(contract_calls, 500_000, 1_500_000).astype(int),
            "unique_contracts": rng.randint(20_000, 80_000, n),
            "contract_gas_used": contract_calls * rng.uniform(80_000, 150_000, n),
        })
        df.to_parquet(
            self.raw_dir / "daily_contract_activity.parquet", index=False
        )
        return df
