#Description:Builds behavioral and transaction-level fraud features using only information available at decision time.
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    #Sort per customer before any rolling feature is calculated.
    df = transactions.sort_values(["customer_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    df["is_international"] = (df["country"] != "US").astype(int)
    #Compare each purchase with the customer-specific expected spend level.
    df["amount_to_baseline"] = (
        df["amount"] / df["customer_baseline_amount"].clip(lower=1.0)
    ).clip(upper=100)
    df["log_amount"] = np.log1p(df["amount"])
    df["log_distance"] = np.log1p(df["distance_from_home_km"])

    grouped = df.groupby("customer_id", group_keys=False)

    #Use past-only rolling behavior to avoid target leakage.
    df["prior_txn_count"] = grouped.cumcount()
    df["prior_avg_amount"] = grouped["amount"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["prior_std_amount"] = grouped["amount"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    df["amount_zscore"] = (
        (df["amount"] - df["prior_avg_amount"])
        / df["prior_std_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 20)

    previous_time = grouped["timestamp"].shift(1)
    df["minutes_since_prior_txn"] = (
        (df["timestamp"] - previous_time).dt.total_seconds() / 60
    ).fillna(1440).clip(0, 10080)

    #Velocity proxy: very short gaps often indicate bursty transaction activity.
    df["rapid_repeat"] = (df["minutes_since_prior_txn"] <= 15).astype(int)
    df["high_amount_deviation"] = (df["amount_to_baseline"] >= 3).astype(int)

    return df.sort_values("timestamp").reset_index(drop=True)
