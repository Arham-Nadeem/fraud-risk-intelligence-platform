#Description:Runs the complete synthetic-data, feature-engineering, modeling, threshold, drift, and reporting workflow.
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_intelligence.config import ARTIFACTS, DATA_PROCESSED, DATA_RAW, DB_PATH, FIGURES, TABLES, TRAIN_SHARE, VALID_SHARE
from fraud_intelligence.data_generation import generate_transactions
from fraud_intelligence.database import load_to_sqlite
from fraud_intelligence.drift import drift_report
from fraud_intelligence.evaluation import choose_threshold, model_metrics, threshold_table
from fraud_intelligence.features import build_features
from fraud_intelligence.modeling import MODEL_FEATURES, feature_importance, save_model, train_models
from fraud_intelligence.reporting import plot_feature_importance, plot_model_curves, plot_risk_by_category, plot_threshold_tradeoff


def ensure_directories() -> None:
    #Create every generated-output directory before the pipeline begins writing artifacts.
    for path in [DATA_RAW, DATA_PROCESSED, ARTIFACTS, FIGURES, TABLES]:
        path.mkdir(parents=True, exist_ok=True)


def chronological_split(df: pd.DataFrame):
    #Split chronologically to better approximate training on the past and scoring future activity.
    n = len(df)
    train_end = int(n * TRAIN_SHARE)
    valid_end = train_end + int(n * VALID_SHARE)
    return df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(), df.iloc[valid_end:].copy()


def main() -> None:
    ensure_directories()

    print("[1/8] Generating synthetic transaction data...")
    raw = generate_transactions()
    raw.to_csv(DATA_RAW / "transactions.csv", index=False)

    print("[2/8] Engineering behavioral fraud features...")
    featured = build_features(raw)
    featured.to_csv(DATA_PROCESSED / "transactions_featured.csv", index=False)
    load_to_sqlite(featured, DB_PATH)

    train_df, valid_df, test_df = chronological_split(featured)

    print("[3/8] Training baseline and gradient-boosted models...")
    best_bundle, model_comparison = train_models(train_df, valid_df)
    model_comparison.to_csv(TABLES / "model_comparison.csv", index=False)

    print(f"[4/8] Selected model: {best_bundle.name}")
    valid_scores = best_bundle.pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
    thresholds = threshold_table(valid_df["is_fraud"], valid_scores, valid_df["amount"])
    chosen = choose_threshold(thresholds)
    chosen_threshold = float(chosen["threshold"])
    thresholds.to_csv(TABLES / "threshold_tradeoff.csv", index=False)

    print(f"[5/8] Business-optimized threshold: {chosen_threshold:.3f}")
    test_scores = best_bundle.pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    metrics = model_metrics(test_df["is_fraud"], test_scores, chosen_threshold, test_df["amount"])
    metrics["model"] = best_bundle.name
    metrics["threshold"] = chosen_threshold

    #Persist the complete labeled test sample so the Streamlit app can demonstrate analyst review workflows.
    test_scored = test_df.copy()
    test_scored["risk_score"] = test_scores
    test_scored["review_decision"] = (test_scores >= chosen_threshold).astype(int)
    test_scored.to_csv(DATA_PROCESSED / "test_transactions_scored.csv", index=False)

    print("[6/8] Computing explainability and drift monitoring outputs...")
    importance = feature_importance(best_bundle)
    importance.to_csv(TABLES / "feature_importance.csv", index=False)
    drift = drift_report(
        train_df,
        test_df,
        ["amount", "distance_from_home_km", "amount_to_baseline", "hour", "minutes_since_prior_txn"],
    )
    drift.to_csv(TABLES / "drift_report.csv", index=False)

    print("[7/8] Generating visual reports...")
    plot_model_curves(test_df["is_fraud"], test_scores, FIGURES / "precision_recall_curve.png")
    plot_threshold_tradeoff(thresholds, chosen_threshold, FIGURES / "threshold_tradeoff.png")
    plot_feature_importance(importance, FIGURES / "feature_importance.png")
    plot_risk_by_category(test_scored, FIGURES / "fraud_rate_by_category.png")

    print("[8/8] Saving model and summary artifacts...")
    save_model(best_bundle, ARTIFACTS / "fraud_model.joblib")
    with open(ARTIFACTS / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame([metrics]).to_csv(TABLES / "test_metrics.csv", index=False)

    print("\nPipeline complete.")
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    print(f"\nSQLite database: {DB_PATH}")
    print(f"Reports: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
