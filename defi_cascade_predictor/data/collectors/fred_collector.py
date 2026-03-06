"""
FRED (Federal Reserve Economic Data) collector for macroeconomic indicators.
Free API with key registration at https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger


class FREDCollector:
    """Collects macroeconomic data from FRED API.

    If fredapi is not installed or no API key is set, generates synthetic
    macro data based on realistic distributions for development/testing.
    """

    # Series IDs for macro indicators relevant to DeFi/crypto correlation
    SERIES = {
        "fed_funds_rate": "DFF",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DTB3",
        "cpi_yoy": "CPIAUCSL",
        "m2_money_supply": "M2SL",
        "vix": "VIXCLS",
        "dollar_index": "DTWEXBGS",
        "sp500": "SP500",
        "unemployment": "UNRATE",
        "real_gdp": "GDPC1",
        "consumer_sentiment": "UMCSENT",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        raw_dir: str = "data/raw",
    ):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.fred = None

        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
                logger.info("FRED API initialized successfully")
            except ImportError:
                logger.warning("fredapi not installed. Using synthetic data.")
        else:
            logger.warning("No FRED API key set. Using synthetic macro data.")

    def collect_series(
        self,
        series_id: str,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.Series:
        """Collect a single FRED series."""
        if self.fred is not None:
            try:
                return self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        return pd.Series(dtype=float)

    def collect_all_macro_data(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Collect all macro indicators and combine into a single DataFrame."""
        logger.info("Collecting macroeconomic data from FRED")

        if self.fred is not None:
            series_data = {}
            for name, series_id in self.SERIES.items():
                try:
                    s = self.collect_series(series_id, start_date, end_date)
                    if not s.empty:
                        series_data[name] = s
                except Exception as e:
                    logger.warning(f"Skipping {name} ({series_id}): {e}")

            if series_data:
                df = pd.DataFrame(series_data)
                df.index.name = "date"
                df = df.ffill().bfill()  # forward/backward fill for different frequencies
                df.to_parquet(self.raw_dir / "macro_data.parquet")
                return df

        # Generate synthetic macro data for development
        logger.info("Generating synthetic macroeconomic data")
        return self._generate_synthetic_macro(start_date, end_date)

    def _generate_synthetic_macro(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate realistic synthetic macro data for development/testing."""
        rng = np.random.RandomState(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)

        # Base trends with realistic ranges
        t = np.linspace(0, 1, n)

        # Fed funds rate: 0% -> 5.5% -> 4.5% trajectory (2021-2025)
        ffr = np.where(
            t < 0.3, 0.08 + rng.normal(0, 0.01, n),
            np.where(
                t < 0.7,
                0.08 + (5.5 - 0.08) * ((t - 0.3) / 0.4),
                5.5 - 1.0 * (t - 0.7) / 0.3,
            ),
        ) + rng.normal(0, 0.02, n)
        ffr = np.clip(ffr, 0, 6)

        # 10Y Treasury
        t10y = 1.5 + 2.5 * np.sin(2 * np.pi * t) + rng.normal(0, 0.1, n)
        t10y = np.clip(t10y, 0.5, 5.0)

        # VIX
        vix_base = 20 + 5 * np.sin(4 * np.pi * t) + rng.normal(0, 3, n)
        # Add spikes at cascade events
        for spike_pos in [0.28, 0.35, 0.62]:  # approximate positions of events
            idx = int(spike_pos * n)
            spike = 20 * np.exp(-0.5 * ((np.arange(n) - idx) / 5) ** 2)
            vix_base += spike
        vix = np.clip(vix_base, 10, 80)

        # Dollar index
        dxy = 95 + 15 * t + rng.normal(0, 0.5, n)
        dxy = np.clip(dxy, 88, 115)

        # S&P 500
        sp500 = 4000 + 800 * t + rng.normal(0, 30, n)
        sp500 = np.cumsum(np.diff(sp500, prepend=sp500[0]) * 0.01) + 4000

        df = pd.DataFrame(
            {
                "fed_funds_rate": ffr,
                "treasury_10y": t10y,
                "treasury_2y": t10y + 0.5 * rng.normal(0, 0.3, n),
                "treasury_3m": np.clip(ffr - 0.1 + rng.normal(0, 0.05, n), 0, 6),
                "cpi_yoy": 2 + 4 * np.sin(np.pi * t) + rng.normal(0, 0.2, n),
                "m2_money_supply": 20000 + 2000 * t + rng.normal(0, 50, n),
                "vix": vix,
                "dollar_index": dxy,
                "sp500": sp500,
                "unemployment": 5 - 2 * t + rng.normal(0, 0.1, n),
                "consumer_sentiment": 70 + 10 * np.sin(2 * np.pi * t)
                + rng.normal(0, 2, n),
            },
            index=dates,
        )
        df.index.name = "date"
        df = df.ffill().bfill()
        df.to_parquet(self.raw_dir / "macro_data.parquet")
        return df

    def compute_macro_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived macro features relevant to DeFi risk."""
        df = macro_df.copy()

        # Yield curve slope (10Y - 2Y spread)
        if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
            df["yield_curve_slope"] = df["treasury_10y"] - df["treasury_2y"]

        # Real rate (fed funds - CPI)
        if "fed_funds_rate" in df.columns and "cpi_yoy" in df.columns:
            df["real_rate"] = df["fed_funds_rate"] - df["cpi_yoy"]

        # VIX changes
        if "vix" in df.columns:
            df["vix_change_1d"] = df["vix"].pct_change(1)
            df["vix_change_7d"] = df["vix"].pct_change(7)
            df["vix_ma_30d"] = df["vix"].rolling(30).mean()
            df["vix_above_ma"] = (df["vix"] > df["vix_ma_30d"]).astype(float)

        # Dollar index momentum
        if "dollar_index" in df.columns:
            df["dxy_return_7d"] = df["dollar_index"].pct_change(7)
            df["dxy_return_30d"] = df["dollar_index"].pct_change(30)

        # M2 growth rate
        if "m2_money_supply" in df.columns:
            df["m2_growth_30d"] = df["m2_money_supply"].pct_change(30)

        # Rate of change of fed funds (hawkish/dovish signal)
        if "fed_funds_rate" in df.columns:
            df["ffr_change_30d"] = df["fed_funds_rate"].diff(30)

        return df
