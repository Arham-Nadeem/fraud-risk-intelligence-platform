#Description:Trains interpretable and nonlinear fraud models, then selects the stronger validation performer.
from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "amount", "new_device", "account_age_days", "distance_from_home_km",
    "hour", "day_of_week", "is_weekend", "is_night", "is_international",
    "amount_to_baseline", "log_amount", "log_distance", "prior_txn_count",
    "prior_avg_amount", "prior_std_amount", "amount_zscore",
    "minutes_since_prior_txn", "rapid_repeat", "high_amount_deviation",
]
CATEGORICAL_FEATURES = ["merchant_category", "channel", "country", "card_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ModelBundle:
    name: str
    pipeline: Pipeline
    validation_pr_auc: float


def _preprocessor() -> ColumnTransformer:
    #Scale numeric features and one-hot encode categories inside a leakage-safe sklearn pipeline.
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def train_models(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[ModelBundle, pd.DataFrame]:
    x_train = train_df[MODEL_FEATURES]
    y_train = train_df["is_fraud"]
    x_valid = valid_df[MODEL_FEATURES]
    y_valid = valid_df["is_fraud"]

    #Weight the boosted model against the severe class imbalance in fraud labels.
    fraud_count = max(int(y_train.sum()), 1)
    nonfraud_count = max(int((1 - y_train).sum()), 1)
    class_ratio = nonfraud_count / fraud_count

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=1.5,
            scale_pos_weight=class_ratio,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        ),
    }

    rows = []
    bundles = []
    #Compare models on validation PR-AUC because fraud is rare and ranking positives matters more than accuracy.
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", _preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        scores = pipeline.predict_proba(x_valid)[:, 1]
        pr_auc = average_precision_score(y_valid, scores)
        rows.append({"model": name, "validation_pr_auc": pr_auc})
        bundles.append(ModelBundle(name, pipeline, pr_auc))

    comparison = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)
    best = max(bundles, key=lambda bundle: bundle.validation_pr_auc)
    return best, comparison


def save_model(bundle: ModelBundle, path) -> None:
    joblib.dump(bundle.pipeline, path)


def feature_importance(bundle: ModelBundle, top_n: int = 20) -> pd.DataFrame:
    #Support both tree importances and absolute logistic coefficients through one reporting interface.
    preprocess = bundle.pipeline.named_steps["preprocess"]
    model = bundle.pipeline.named_steps["model"]
    names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
