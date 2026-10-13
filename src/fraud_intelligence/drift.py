#Description:Computes population stability index to flag feature-distribution drift between reference and recent data.
from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference, current, bins: int = 10) -> float:
    #Use reference quantile bins so PSI reflects how the recent population shifted from training behavior.
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)
    if reference.empty or current.empty:
        return float("nan")

    quantiles = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    ref_counts = pd.cut(reference, quantiles, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(current, quantiles, include_lowest=True).value_counts(sort=False)

    ref_pct = (ref_counts / ref_counts.sum()).clip(lower=1e-6)
    cur_pct = (cur_counts / cur_counts.sum()).clip(lower=1e-6)
    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    #Map PSI values into simple operational monitoring statuses.
    rows = []
    for feature in features:
        psi = population_stability_index(reference_df[feature], current_df[feature])
        if psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "monitor"
        else:
            status = "investigate"
        rows.append({"feature": feature, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
