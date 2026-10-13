#Description:Evaluates fraud detection with imbalance-aware metrics and cost-sensitive threshold optimization.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import FALSE_POSITIVE_CUSTOMER_COST, MANUAL_REVIEW_COST, MISSED_FRAUD_MULTIPLIER


def threshold_table(y_true, scores, amounts, thresholds=None) -> pd.DataFrame:
    #Sweep thresholds instead of assuming the default 0.50 decision boundary.
    if thresholds is None:
        thresholds = np.linspace(0.03, 0.90, 88)

    rows = []
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)
    total_fraud_value = amounts[y_true == 1].sum()

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        captured_value = amounts[(y_true == 1) & (pred == 1)].sum()
        missed_value = amounts[(y_true == 1) & (pred == 0)].sum()
        review_count = int(pred.sum())

        #Combine fraud loss, investigation workload, and customer-friction cost into one decision objective.
        total_cost = (
            MISSED_FRAUD_MULTIPLIER * missed_value
            + MANUAL_REVIEW_COST * review_count
            + FALSE_POSITIVE_CUSTOMER_COST * fp
        )

        rows.append({
            "threshold": round(float(threshold), 4),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "false_positive_rate": fp / max(fp + tn, 1),
            "review_rate": review_count / len(y_true),
            "fraud_value_capture_rate": captured_value / max(total_fraud_value, 1),
            "captured_fraud_value": captured_value,
            "missed_fraud_value": missed_value,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_business_cost": total_cost,
        })

    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame) -> pd.Series:
    #Minimize estimated business cost and favor stronger fraud-value capture on ties.
    return table.sort_values(
        ["total_business_cost", "fraud_value_capture_rate"],
        ascending=[True, False],
    ).iloc[0]


def model_metrics(y_true, scores, threshold: float, amounts=None) -> dict:
    #Report ranking metrics and operational classification outcomes at the chosen threshold.
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "review_rate": float(pred.mean()),
        "false_positive_rate": fp / max(fp + tn, 1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    if amounts is not None:
        #Attach dollar-based capture metrics when transaction values are available.
        amounts = np.asarray(amounts)
        fraud_value = amounts[y_true == 1].sum()
        captured_value = amounts[(y_true == 1) & (pred == 1)].sum()
        metrics["fraud_value_capture_rate"] = captured_value / max(fraud_value, 1)
        metrics["captured_fraud_value"] = float(captured_value)
        metrics["total_fraud_value"] = float(fraud_value)

    return metrics
