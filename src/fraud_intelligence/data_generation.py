#Description:Generates realistic synthetic card transactions with behavioral fraud signals and no paid data dependency.
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FRAUD_BASE_RATE, N_TRANSACTIONS, RANDOM_SEED

MERCHANT_CATEGORIES = [
    "grocery", "restaurants", "fuel", "retail", "electronics",
    "travel", "entertainment", "digital_goods", "healthcare", "utilities",
]
CARD_TYPES = ["credit", "debit"]
CHANNELS = ["chip", "contactless", "online", "mobile_wallet"]
COUNTRIES = ["US", "CA", "GB", "MX", "DE"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_transactions(n_rows: int = N_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    #Use a fixed seed by default so portfolio results are reproducible across runs.
    rng = np.random.default_rng(seed)

    customer_count = max(2500, n_rows // 10)
    customer_ids = rng.integers(100000, 100000 + customer_count, size=n_rows)
    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="8min")

    #Assign each customer a persistent spending baseline before individual transactions are generated.
    customer_baseline = rng.lognormal(mean=3.4, sigma=0.55, size=customer_count)
    baseline_index = customer_ids - 100000
    normal_spend = customer_baseline[baseline_index]

    merchant_category = rng.choice(
        MERCHANT_CATEGORIES,
        size=n_rows,
        p=[0.18, 0.14, 0.10, 0.15, 0.08, 0.06, 0.08, 0.06, 0.08, 0.07],
    )
    channel = rng.choice(CHANNELS, size=n_rows, p=[0.34, 0.26, 0.28, 0.12])
    country = rng.choice(COUNTRIES, size=n_rows, p=[0.90, 0.03, 0.025, 0.025, 0.02])
    card_type = rng.choice(CARD_TYPES, size=n_rows, p=[0.68, 0.32])

    amount = rng.lognormal(mean=np.log(np.maximum(normal_spend, 5)), sigma=0.70)
    hour = timestamps.hour.to_numpy()
    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    new_device = rng.binomial(1, 0.055, size=n_rows)
    international = (country != "US").astype(int)
    account_age_days = rng.integers(30, 3650, size=n_rows)
    distance_from_home_km = np.maximum(0, rng.gamma(shape=1.6, scale=18, size=n_rows))

    #Inject risk patterns rather than assigning labels randomly.
    category_risk = pd.Series(merchant_category).map({
        "electronics": 0.85,
        "travel": 0.65,
        "digital_goods": 1.05,
        "retail": 0.30,
    }).fillna(0).to_numpy()
    channel_risk = pd.Series(channel).map({"online": 0.80, "mobile_wallet": 0.25}).fillna(0).to_numpy()
    spend_ratio = amount / np.maximum(normal_spend, 1)

    raw_score = (
        -5.35
        + 0.95 * new_device
        + 0.85 * international
        + 0.55 * is_night
        + 0.52 * np.log1p(distance_from_home_km / 30)
        + 0.72 * (spend_ratio > 3.0)
        + 0.58 * (spend_ratio > 6.0)
        + category_risk
        + channel_risk
        + 0.45 * (account_age_days < 120)
        + rng.normal(0, 0.55, size=n_rows)
    )

    #Convert latent risk to probability, then calibrate the overall synthetic fraud prevalence.
    probabilities = _sigmoid(raw_score)
    #Scale probabilities toward the configured base rate while preserving relative risk.
    probabilities *= FRAUD_BASE_RATE / max(probabilities.mean(), 1e-9)
    probabilities = np.clip(probabilities, 0, 0.92)
    is_fraud = rng.binomial(1, probabilities)

    #Fraudulent activity tends to be larger and farther from the customer's normal behavior.
    amount = np.where(is_fraud == 1, amount * rng.uniform(1.4, 3.5, size=n_rows), amount)
    distance_from_home_km = np.where(
        is_fraud == 1,
        distance_from_home_km + rng.gamma(2.0, 65.0, size=n_rows),
        distance_from_home_km,
    )

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(1, n_rows + 1)],
        "timestamp": timestamps,
        "customer_id": customer_ids.astype(str),
        "amount": amount.round(2),
        "merchant_category": merchant_category,
        "channel": channel,
        "country": country,
        "card_type": card_type,
        "new_device": new_device,
        "account_age_days": account_age_days,
        "distance_from_home_km": distance_from_home_km.round(2),
        "customer_baseline_amount": normal_spend.round(2),
        "is_fraud": is_fraud,
    })

    return df
