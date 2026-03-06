"""
Cascade event labeler: creates ground truth labels for liquidation cascade
prediction from known historical events and TVL-based anomaly detection.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger


class CascadeLabeler:
    """Labels time periods as cascade/non-cascade based on known events
    and statistical anomaly detection on TVL drawdowns."""

    def __init__(self, cascade_events: list[dict]):
        """
        Args:
            cascade_events: List of dicts with keys:
                name, start, peak, end, severity, tvl_loss_pct
        """
        self.events = []
        for event in cascade_events:
            self.events.append({
                "name": event["name"],
                "start": pd.Timestamp(event["start"]),
                "peak": pd.Timestamp(event["peak"]),
                "end": pd.Timestamp(event["end"]),
                "severity": event["severity"],
                "tvl_loss_pct": event["tvl_loss_pct"],
            })
        logger.info(f"CascadeLabeler initialized with {len(self.events)} known events")

    def label_known_events(
        self,
        dates: pd.DatetimeIndex,
        prediction_horizons: list[int] = [24, 72, 168, 720],
    ) -> pd.DataFrame:
        """Create binary labels for each date and prediction horizon.

        For each horizon h, label[t] = 1 if a cascade occurs within the next h hours.

        Args:
            dates: DatetimeIndex of all timestamps to label.
            prediction_horizons: List of horizons in hours [1, 6, 24, 168].

        Returns:
            DataFrame with columns: date, cascade_{h}h for each horizon,
            cascade_severity, cascade_name.
        """
        labels = pd.DataFrame({"date": dates})

        for h in prediction_horizons:
            col = f"cascade_{h}h"
            labels[col] = 0

        labels["cascade_severity"] = "none"
        labels["cascade_name"] = ""
        labels["cascade_active"] = 0
        labels["tvl_loss_pct"] = 0.0

        for event in self.events:
            # Mark cascade-active period
            active_mask = (labels["date"] >= event["start"]) & (
                labels["date"] <= event["end"]
            )
            labels.loc[active_mask, "cascade_active"] = 1
            labels.loc[active_mask, "cascade_severity"] = event["severity"]
            labels.loc[active_mask, "cascade_name"] = event["name"]
            labels.loc[active_mask, "tvl_loss_pct"] = event["tvl_loss_pct"]

            # For prediction labels: mark the PRE-cascade window
            for h in prediction_horizons:
                horizon_td = timedelta(hours=h)
                pre_mask = (labels["date"] >= event["start"] - horizon_td) & (
                    labels["date"] < event["start"]
                )
                labels.loc[pre_mask, f"cascade_{h}h"] = 1

            # Also mark the active period itself
            for h in prediction_horizons:
                labels.loc[active_mask, f"cascade_{h}h"] = 1

        return labels

    def detect_tvl_anomalies(
        self,
        tvl_df: pd.DataFrame,
        z_threshold: float = -2.5,
        drawdown_threshold: float = -0.10,
        min_duration_days: int = 2,
    ) -> list[dict]:
        """Detect potential cascade events from TVL drawdown anomalies.

        Args:
            tvl_df: DataFrame with columns [date, tvl_usd].
            z_threshold: Z-score threshold for TVL change (negative).
            drawdown_threshold: Minimum drawdown to qualify.
            min_duration_days: Minimum number of consecutive anomaly days.

        Returns:
            List of detected anomaly events.
        """
        df = tvl_df.copy().sort_values("date")
        df["tvl_return"] = df["tvl_usd"].pct_change()
        df["tvl_return_7d"] = df["tvl_usd"].pct_change(7)
        df["rolling_high"] = df["tvl_usd"].rolling(30).max()
        df["drawdown"] = df["tvl_usd"] / df["rolling_high"] - 1

        # Z-score of daily returns
        mean_ret = df["tvl_return"].rolling(90).mean()
        std_ret = df["tvl_return"].rolling(90).std()
        df["z_score"] = (df["tvl_return"] - mean_ret) / std_ret

        # Identify anomalous days
        df["is_anomaly"] = (
            (df["z_score"] < z_threshold)
            | (df["drawdown"] < drawdown_threshold)
        ).astype(int)

        # Group consecutive anomalous days into events
        df["anomaly_group"] = (
            df["is_anomaly"].diff().ne(0).cumsum() * df["is_anomaly"]
        )

        events = []
        for group_id in df[df["anomaly_group"] > 0]["anomaly_group"].unique():
            group = df[df["anomaly_group"] == group_id]
            duration = (group["date"].max() - group["date"].min()).days + 1
            if duration >= min_duration_days:
                events.append({
                    "name": f"detected_anomaly_{group_id}",
                    "start": group["date"].min(),
                    "peak": group.loc[group["drawdown"].idxmin(), "date"],
                    "end": group["date"].max(),
                    "severity": self._classify_severity(
                        group["drawdown"].min()
                    ),
                    "tvl_loss_pct": abs(group["drawdown"].min()),
                    "duration_days": duration,
                    "max_z_score": group["z_score"].min(),
                })

        logger.info(f"Detected {len(events)} potential cascade events from TVL data")
        return events

    def _classify_severity(self, max_drawdown: float) -> str:
        """Classify event severity based on maximum drawdown."""
        if max_drawdown < -0.30:
            return "catastrophic"
        elif max_drawdown < -0.15:
            return "severe"
        elif max_drawdown < -0.08:
            return "moderate"
        else:
            return "minor"

    def create_multi_horizon_labels(
        self,
        tvl_df: pd.DataFrame,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        combine_known_and_detected: bool = True,
        z_threshold: float = -2.5,
        drawdown_threshold: float = -0.10,
        min_duration_days: int = 2,
    ) -> pd.DataFrame:
        """Create final multi-horizon labels combining known events and
        detected anomalies.

        Args:
            tvl_df: DataFrame with [date, tvl_usd] for anomaly detection.
            prediction_horizons: Forecast horizons in hours.
            combine_known_and_detected: Whether to also detect from TVL.
            z_threshold: Z-score threshold for anomaly detection.
            drawdown_threshold: Drawdown threshold for anomaly detection.
            min_duration_days: Minimum consecutive anomaly days.

        Returns:
            Complete label DataFrame.
        """
        dates = pd.DatetimeIndex(tvl_df["date"].unique()).sort_values()
        labels = self.label_known_events(dates, prediction_horizons)

        if combine_known_and_detected:
            detected = self.detect_tvl_anomalies(
                tvl_df,
                z_threshold=z_threshold,
                drawdown_threshold=drawdown_threshold,
                min_duration_days=min_duration_days,
            )
            for event in detected:
                # Only add if not already covered by a known event
                overlap = False
                for known in self.events:
                    if (
                        event["start"] <= known["end"]
                        and event["end"] >= known["start"]
                    ):
                        overlap = True
                        break
                if not overlap:
                    self.events.append(event)
                    logger.info(f"Added detected event: {event['name']}")

            # Re-label with new events
            labels = self.label_known_events(dates, prediction_horizons)

        # Add continuous risk score (higher near cascade events)
        labels["risk_score"] = 0.0
        for event in self.events:
            severity_map = {
                "catastrophic": 1.0, "severe": 0.75,
                "moderate": 0.5, "minor": 0.25,
            }
            base_score = severity_map.get(event["severity"], 0.25)
            for idx, row in labels.iterrows():
                dist = abs((row["date"] - event["start"]).days)
                if dist <= 30:
                    contribution = base_score * np.exp(-0.1 * dist)
                    labels.loc[idx, "risk_score"] = max(
                        labels.loc[idx, "risk_score"], contribution
                    )

        return labels

    def get_event_windows(
        self, pre_days: int = 14, post_days: int = 7
    ) -> list[dict]:
        """Get expanded event windows for case study analysis."""
        windows = []
        for event in self.events:
            windows.append({
                **event,
                "window_start": event["start"] - timedelta(days=pre_days),
                "window_end": event["end"] + timedelta(days=post_days),
            })
        return windows
