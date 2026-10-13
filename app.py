#Description:Interactive Streamlit dashboard for fraud monitoring, review-queue analysis, and transaction investigation.
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "processed" / "test_transactions_scored.csv"
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"
IMPORTANCE_PATH = ROOT / "reports" / "tables" / "feature_importance.csv"
THRESHOLD_PATH = ROOT / "reports" / "tables" / "threshold_tradeoff.csv"
DRIFT_PATH = ROOT / "reports" / "tables" / "drift_report.csv"
MODEL_COMPARISON_PATH = ROOT / "reports" / "tables" / "model_comparison.csv"
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(page_title="Fraud Risk Intelligence", page_icon="🛡️", layout="wide")
st.title("Fraud Risk Intelligence Platform")
st.caption("Local, no-API fraud detection, transaction investigation, and model-monitoring demo")

required_paths = [DATA_PATH, MODEL_PATH, METADATA_PATH, IMPORTANCE_PATH, THRESHOLD_PATH, DRIFT_PATH]
if not all(path.exists() for path in required_paths):
    st.warning("Run `py run_pipeline.py` first to generate the model, scored data, and reporting artifacts.")
    st.stop()

#Load persisted artifacts once so every dashboard tab uses the same model run.
scored = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
importance = pd.read_csv(IMPORTANCE_PATH)
thresholds = pd.read_csv(THRESHOLD_PATH)
drift = pd.read_csv(DRIFT_PATH)
model_comparison = pd.read_csv(MODEL_COMPARISON_PATH) if MODEL_COMPARISON_PATH.exists() else pd.DataFrame()

#Precompute category statistics for both monitoring and transaction-level reason codes.
category_summary = (
    scored.groupby("merchant_category")
    .agg(
        transactions=("transaction_id", "count"),
        observed_fraud_rate=("is_fraud", "mean"),
        average_risk_score=("risk_score", "mean"),
    )
    .sort_values("observed_fraud_rate", ascending=False)
)
category_risk_cutoff = float(category_summary["observed_fraud_rate"].quantile(0.75))
distance_cutoff = float(scored["distance_from_home_km"].quantile(0.95))


def transaction_reason_codes(row: pd.Series) -> list[dict]:
    #Translate engineered features into investigator-friendly behavioral signals.
    reasons: list[dict] = []

    if float(row["amount_to_baseline"]) >= 3:
        reasons.append({
            "signal": "Spend anomaly",
            "detail": f"Amount is {float(row['amount_to_baseline']):.1f}× the customer's baseline",
            "severity": "High" if float(row["amount_to_baseline"]) >= 6 else "Medium",
        })
    if float(row["distance_from_home_km"]) >= distance_cutoff:
        reasons.append({
            "signal": "Location anomaly",
            "detail": f"Transaction occurred {float(row['distance_from_home_km']):.0f} km from home (top 5% of test activity)",
            "severity": "High",
        })
    if int(row.get("new_device", 0)) == 1:
        reasons.append({"signal": "New device", "detail": "Transaction used a newly observed device", "severity": "Medium"})
    if int(row.get("is_international", 0)) == 1:
        reasons.append({"signal": "International activity", "detail": f"Transaction country is {row['country']}", "severity": "Medium"})
    if int(row.get("is_night", 0)) == 1:
        reasons.append({"signal": "Unusual hour", "detail": f"Transaction occurred around {int(row['hour']):02d}:00", "severity": "Low"})
    if int(row.get("rapid_repeat", 0)) == 1:
        reasons.append({"signal": "Velocity spike", "detail": "Transaction followed the prior transaction within 15 minutes", "severity": "Medium"})
    if float(row.get("amount_zscore", 0)) >= 3:
        reasons.append({"signal": "Customer-history deviation", "detail": f"Amount z-score is {float(row['amount_zscore']):.1f}", "severity": "High"})

    category_rate = float(category_summary.loc[row["merchant_category"], "observed_fraud_rate"])
    if category_rate >= category_risk_cutoff:
        reasons.append({
            "signal": "Elevated merchant-category risk",
            "detail": f"{row['merchant_category']} has a {category_rate:.2%} observed fraud rate in the test sample",
            "severity": "Low",
        })

    if not reasons:
        reasons.append({"signal": "No dominant behavioral anomaly", "detail": "Risk is driven by the model's combined feature pattern", "severity": "Low"})
    return reasons[:6]


def format_review_outcome(row: pd.Series) -> str:
    #Expose labeled test outcomes only because this repository uses synthetic ground truth.
    if int(row["review_decision"]) == 1 and int(row["is_fraud"]) == 1:
        return "True Positive"
    if int(row["review_decision"]) == 1 and int(row["is_fraud"]) == 0:
        return "False Positive"
    if int(row["review_decision"]) == 0 and int(row["is_fraud"]) == 1:
        return "False Negative"
    return "True Negative"


#Executive metrics summarize model quality and operational workload at the selected threshold.
metric_cols = st.columns(6)
metric_cols[0].metric("Transactions", f"{len(scored):,}")
metric_cols[1].metric("Observed fraud rate", f"{scored['is_fraud'].mean():.2%}")
metric_cols[2].metric("Model PR-AUC", f"{metadata['pr_auc']:.3f}")
metric_cols[3].metric("Recall", f"{metadata['recall']:.1%}")
metric_cols[4].metric("Precision", f"{metadata['precision']:.1%}")
metric_cols[5].metric("Decision threshold", f"{metadata['threshold']:.2f}")

queue_tab, investigation_tab, performance_tab, monitoring_tab = st.tabs([
    "Review Queue", "Transaction Investigator", "Model Performance", "Monitoring"
])

with queue_tab:
    st.subheader("Risk review queue")
    st.caption("Lower the queue threshold to inspect borderline cases and false-positive tradeoffs.")

    #Default slightly below the production threshold so the queue contains a realistic mix of outcomes.
    default_queue_threshold = max(0.50, float(metadata["threshold"]) - 0.15)
    minimum_risk = st.slider("Minimum risk score", 0.0, 1.0, default_queue_threshold, 0.01)
    filtered = scored[scored["risk_score"] >= minimum_risk].sort_values("risk_score", ascending=False).copy()
    filtered["risk_score_pct"] = filtered["risk_score"].map(lambda value: f"{value:.2%}")
    filtered["decision"] = filtered["review_decision"].map({1: "Review", 0: "Clear"})
    filtered["demo_outcome"] = filtered.apply(format_review_outcome, axis=1)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Queued transactions", f"{len(filtered):,}")
    q2.metric("Queue rate", f"{len(filtered) / len(scored):.1%}")
    q3.metric("Observed fraud in queue", f"{filtered['is_fraud'].mean():.1%}" if len(filtered) else "0.0%")
    q4.metric("False positives in queue", f"{((filtered['review_decision'] == 1) & (filtered['is_fraud'] == 0)).sum():,}")

    display_columns = [
        "transaction_id", "timestamp", "amount", "merchant_category", "channel", "country",
        "new_device", "amount_to_baseline", "distance_from_home_km", "risk_score_pct", "decision", "demo_outcome",
    ]
    st.dataframe(filtered[display_columns].head(200), use_container_width=True, hide_index=True)

with investigation_tab:
    st.subheader("Investigate a transaction")
    candidate_ids = scored.sort_values("risk_score", ascending=False)["transaction_id"].tolist()
    transaction_id = st.selectbox("Transaction", candidate_ids)
    row = scored.loc[scored["transaction_id"] == transaction_id].iloc[0]

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Risk score", f"{float(row['risk_score']):.2%}")
    i2.metric("Decision", "Review" if int(row["review_decision"]) else "Clear")
    i3.metric("Amount", f"${float(row['amount']):,.2f}")
    i4.metric("Demo outcome", format_review_outcome(row))

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Transaction details")
        details = pd.DataFrame([
            ("Merchant category", row["merchant_category"]),
            ("Channel", row["channel"]),
            ("Country", row["country"]),
            ("New device", "Yes" if int(row["new_device"]) else "No"),
            ("Amount vs baseline", f"{float(row['amount_to_baseline']):.2f}×"),
            ("Distance from home", f"{float(row['distance_from_home_km']):,.1f} km"),
            ("Minutes since prior transaction", f"{float(row['minutes_since_prior_txn']):,.0f}"),
        ], columns=["Field", "Value"])
        st.dataframe(details, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Transaction-specific risk signals")
        st.dataframe(pd.DataFrame(transaction_reason_codes(row)), use_container_width=True, hide_index=True)

    st.markdown("#### Global model drivers")
    st.caption("These are model-wide feature importance values; the signals above explain the selected transaction behaviorally.")
    st.dataframe(importance.head(10), use_container_width=True, hide_index=True)

with performance_tab:
    st.subheader("Model performance and threshold economics")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")
    p2.metric("PR-AUC", f"{metadata['pr_auc']:.3f}")
    p3.metric("Precision", f"{metadata['precision']:.1%}")
    p4.metric("Recall", f"{metadata['recall']:.1%}")
    p5.metric("Review rate", f"{metadata.get('review_rate', 0):.1%}")

    confusion = pd.DataFrame(
        [
            [metadata["true_negatives"], metadata["false_positives"]],
            [metadata["false_negatives"], metadata["true_positives"]],
        ],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted clear", "Predicted review"],
    )
    st.markdown("#### Confusion matrix")
    st.dataframe(confusion, use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Precision-recall curve")
        st.image(str(FIGURES / "precision_recall_curve.png"), use_container_width=True)
    with chart_right:
        st.markdown("#### Threshold trade-off")
        st.image(str(FIGURES / "threshold_tradeoff.png"), use_container_width=True)

    st.markdown("#### Model comparison")
    if not model_comparison.empty:
        comparison_display = model_comparison.copy()
        comparison_display["validation_pr_auc"] = comparison_display["validation_pr_auc"].map(lambda value: f"{value:.3f}")
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)

    chosen_row = thresholds.iloc[(thresholds["threshold"] - float(metadata["threshold"])).abs().argsort()[:1]]
    if not chosen_row.empty:
        selected = chosen_row.iloc[0]
        b1, b2, b3 = st.columns(3)
        b1.metric("Fraud value captured", f"{float(selected['fraud_value_capture_rate']):.1%}")
        b2.metric("False-positive rate", f"{float(selected['false_positive_rate']):.2%}")
        b3.metric("Estimated business cost", f"${float(selected['total_business_cost']):,.0f}")

with monitoring_tab:
    st.subheader("Risk segmentation and drift monitoring")
    category_display = category_summary.reset_index().copy()
    category_display["observed_fraud_rate"] = category_display["observed_fraud_rate"].map(lambda value: f"{value:.2%}")
    category_display["average_risk_score"] = category_display["average_risk_score"].map(lambda value: f"{value:.2%}")
    st.markdown("#### Risk by merchant category")
    st.dataframe(category_display, use_container_width=True, hide_index=True)

    st.markdown("#### Population Stability Index")
    st.caption("PSI flags feature-distribution shifts between the training reference window and the test window.")
    drift_display = drift.copy()
    drift_display["psi"] = drift_display["psi"].map(lambda value: f"{value:.4f}")
    st.dataframe(drift_display, use_container_width=True, hide_index=True)

    m_left, m_right = st.columns(2)
    with m_left:
        st.image(str(FIGURES / "fraud_rate_by_category.png"), use_container_width=True)
    with m_right:
        st.image(str(FIGURES / "feature_importance.png"), use_container_width=True)
