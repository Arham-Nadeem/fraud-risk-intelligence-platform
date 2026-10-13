#Description:Lightweight unit tests for synthetic generation, feature engineering, and threshold selection.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_intelligence.data_generation import generate_transactions
from fraud_intelligence.evaluation import choose_threshold, threshold_table
from fraud_intelligence.features import build_features


def test_generation_schema_and_fraud_rate():
    #Confirm synthetic data remains usable and fraud prevalence stays realistically imbalanced.
    df = generate_transactions(n_rows=5000, seed=7)
    expected = {"transaction_id", "timestamp", "amount", "merchant_category", "is_fraud"}
    assert expected.issubset(df.columns)
    assert len(df) == 5000
    assert 0 < df["is_fraud"].mean() < 0.20


def test_feature_engineering_adds_behavioral_features():
    #Protect the behavioral feature contract used by training and scoring.
    df = build_features(generate_transactions(n_rows=1500, seed=8))
    assert "amount_zscore" in df.columns
    assert "minutes_since_prior_txn" in df.columns
    assert "rapid_repeat" in df.columns
    assert df["amount_to_baseline"].notna().all()


def test_threshold_optimizer_returns_valid_choice():
    #Verify cost-based threshold selection always returns an actionable decision boundary.
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    amounts = [20, 40, 100, 500]
    table = threshold_table(y, scores, amounts, thresholds=[0.3, 0.5, 0.7])
    chosen = choose_threshold(table)
    assert 0 <= chosen["threshold"] <= 1
    assert chosen["total_business_cost"] >= 0
