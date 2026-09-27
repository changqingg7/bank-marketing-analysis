"""Feature engineering and dataset splitting for pre-call prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target_subscription"
LEAKAGE_COLUMNS = ["y", "target_subscription", "duration"]


def build_precall_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features known before the current marketing call begins."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    features = df.drop(columns=LEAKAGE_COLUMNS).copy()
    features["age_group"] = pd.cut(
        features["age"], bins=[0, 25, 35, 45, 55, 65, float("inf")], right=False
    ).astype(str)
    features["previously_contacted"] = (features["pdays"] != 999).astype("int8")
    features["pdays_since_last_contact"] = features["pdays"].mask(
        features["pdays"] == 999, float("nan")
    )
    features["has_housing_loan"] = (features["housing"] == "yes").astype("int8")
    features["has_personal_loan"] = (features["loan"] == "yes").astype("int8")
    features = features.drop(columns=["pdays", "housing", "loan"])

    return features, df[TARGET_COLUMN].astype("int8")


def stratified_train_test_split(
    features: pd.DataFrame, target: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data while preserving the positive-class rate in both subsets."""
    return train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
