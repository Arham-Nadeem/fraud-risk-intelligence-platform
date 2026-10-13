#Description:Creates recruiter-friendly model, threshold, risk, and feature-importance visualizations.
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_model_curves(y_true, scores, figure_path) -> None:
    #PR curves are more informative than accuracy for an imbalanced fraud dataset.
    fig, ax = plt.subplots(figsize=(8, 5))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(table: pd.DataFrame, chosen_threshold: float, figure_path) -> None:
    #Visualize the operational tradeoff created by changing the review threshold.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table["threshold"], table["recall"], label="Fraud recall")
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["fraud_value_capture_rate"], label="Fraud value captured")
    ax.axvline(chosen_threshold, linestyle="--", label=f"Chosen threshold {chosen_threshold:.2f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Risk Threshold Trade-Off")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, figure_path) -> None:
    if importance.empty:
        return
    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top Fraud-Risk Features")
    ax.set_xlabel("Model importance")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_risk_by_category(scored: pd.DataFrame, figure_path) -> None:
    summary = (
        scored.groupby("merchant_category")
        .agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"), avg_risk=("risk_score", "mean"))
        .reset_index()
        .sort_values("fraud_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["merchant_category"], summary["fraud_rate"] * 100)
    ax.set_ylabel("Observed fraud rate (%)")
    ax.set_xlabel("Merchant category")
    ax.set_title("Fraud Rate by Merchant Category")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
