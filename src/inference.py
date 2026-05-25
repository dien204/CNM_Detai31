import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import calculate_trust_score, get_risk_level

TARGET = "isFraud"
ID_COL = "TransactionID"

COMMON_USER_ID_COLUMNS = [
    "UserID",
    "user_id",
    "CustomerID",
    "customer_id",
    "AccountID",
    "account_id",
]

SURROGATE_USER_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

DEFAULT_MANUAL_FIELDS = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "C1",
    "C2",
    "D1",
    "V1",
]


def load_json_if_exists(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _safe_numeric_series(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _reference_default(reference_df: Optional[pd.DataFrame], col: str, fallback: float = 0.0) -> float:
    if reference_df is None or col not in reference_df.columns:
        return fallback
    numeric = pd.to_numeric(reference_df[col], errors="coerce")
    value = numeric.median()
    if pd.isna(value) or np.isinf(value):
        return fallback
    return float(value)


def encode_with_known_classes(series: pd.Series, classes: Iterable[str]) -> pd.Series:
    mapping = {str(value): idx for idx, value in enumerate(classes)}
    encoded = series.astype(str).map(mapping)
    return encoded.fillna(-1).astype(float)


def prepare_features_for_inference(
    input_df: pd.DataFrame,
    feature_columns: List[str],
    reference_df: Optional[pd.DataFrame] = None,
    fill_values: Optional[Dict[str, float]] = None,
    label_encoders: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Return a numeric DataFrame ordered exactly like the training feature list.

    The function accepts both processed CSV files and partially raw CSV files.
    Missing columns are filled by training medians when available, then by
    medians from the demo/reference data, then by 0. Unknown categorical values
    are mapped to -1 so the dashboard can continue running during demo.
    """
    if input_df is None or input_df.empty:
        raise ValueError("Input data is empty.")

    df = input_df.copy()
    fill_values = fill_values or {}
    label_encoders = label_encoders or {}

    prepared_columns = {}

    for col in feature_columns:
        default_value = fill_values.get(col, _reference_default(reference_df, col, 0.0))

        if col not in df.columns:
            prepared_columns[col] = pd.Series(default_value, index=df.index)
            continue

        if col in label_encoders and df[col].dtype == object:
            prepared_columns[col] = encode_with_known_classes(df[col], label_encoders[col])
        else:
            prepared_columns[col] = _safe_numeric_series(df[col], default_value)

    prepared = pd.DataFrame(prepared_columns, index=df.index)
    return prepared[feature_columns]


def score_transactions(
    model,
    input_df: pd.DataFrame,
    feature_columns: List[str],
    reference_df: Optional[pd.DataFrame] = None,
    fill_values: Optional[Dict[str, float]] = None,
    label_encoders: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = prepare_features_for_inference(
        input_df,
        feature_columns,
        reference_df=reference_df,
        fill_values=fill_values,
        label_encoders=label_encoders,
    )
    fraud_probability = model.predict_proba(X)[:, 1]

    result = input_df.copy()
    result["Fraud_Probability"] = fraud_probability
    result["Trust_Score"] = [calculate_trust_score(p) for p in fraud_probability]
    result["Risk_Level"] = [get_risk_level(score) for score in result["Trust_Score"]]
    result["Risk_Rank"] = pd.cut(
        result["Trust_Score"],
        bins=[-np.inf, 49.999, 79.999, np.inf],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    ).astype(str)
    return result, X


def choose_user_identifier(df: pd.DataFrame) -> str:
    for col in COMMON_USER_ID_COLUMNS:
        if col in df.columns:
            return col

    available = [col for col in SURROGATE_USER_COLUMNS if col in df.columns]
    if not available:
        return "SyntheticUserID"

    return "User_Key"


def add_user_key(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    result = df.copy()
    user_col = choose_user_identifier(result)

    if user_col in result.columns and user_col != "User_Key":
        result[user_col] = result[user_col].astype(str)
        return result, user_col

    available = [col for col in SURROGATE_USER_COLUMNS if col in result.columns]
    if available:
        key_frame = result[available].astype(str).fillna("NA")
        result["User_Key"] = key_frame.agg("|".join, axis=1)
        return result, "User_Key"

    result["SyntheticUserID"] = [f"U{i:06d}" for i in range(len(result))]
    return result, "SyntheticUserID"


def aggregate_user_scores(scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df, user_col = add_user_key(scored_df)

    aggregation = {
        "Trust_Score": ["mean", "min", "max"],
        "Fraud_Probability": ["mean", "max"],
    }

    if "TransactionAmt" in df.columns:
        aggregation["TransactionAmt"] = ["sum", "mean"]

    user_df = df.groupby(user_col).agg(aggregation)
    user_df.columns = ["_".join(col).strip("_") for col in user_df.columns]
    user_df = user_df.reset_index()

    counts = df.groupby(user_col).size().reset_index(name="Transaction_Count")
    high_risk = (
        df.assign(_high_risk=df["Trust_Score"] < 50)
        .groupby(user_col)["_high_risk"]
        .sum()
        .reset_index(name="High_Risk_Transactions")
    )

    user_df = user_df.merge(counts, on=user_col, how="left").merge(high_risk, on=user_col, how="left")
    user_df["User_Trust_Score"] = user_df["Trust_Score_mean"].round(2)
    user_df["User_Risk_Level"] = user_df["User_Trust_Score"].apply(get_risk_level)
    user_df["User_Fraud_Probability"] = user_df["Fraud_Probability_mean"].round(4)

    ordered_cols = [
        user_col,
        "User_Trust_Score",
        "User_Risk_Level",
        "User_Fraud_Probability",
        "Fraud_Probability_max",
        "Transaction_Count",
        "High_Risk_Transactions",
        "Trust_Score_min",
        "Trust_Score_max",
    ]
    optional = [col for col in ["TransactionAmt_sum", "TransactionAmt_mean"] if col in user_df.columns]
    user_df = user_df[[col for col in ordered_cols + optional if col in user_df.columns]]
    user_df = user_df.sort_values(["User_Trust_Score", "High_Risk_Transactions"], ascending=[True, False])

    return user_df, user_col


def build_manual_sample(reference_df: pd.DataFrame, overrides: Dict[str, float], feature_columns: List[str]) -> pd.DataFrame:
    if reference_df is not None and not reference_df.empty:
        base = reference_df[feature_columns].median(numeric_only=True).to_dict()
    else:
        base = {col: 0.0 for col in feature_columns}

    row = {col: float(base.get(col, 0.0)) for col in feature_columns}
    row.update({key: value for key, value in overrides.items() if key in feature_columns})
    row[ID_COL] = 99999999
    return pd.DataFrame([row])


def summarize_batch(scored_df: pd.DataFrame) -> Dict[str, float]:
    if scored_df.empty:
        return {}

    total = len(scored_df)
    low_trust = int((scored_df["Trust_Score"] < 50).sum())
    medium_trust = int(((scored_df["Trust_Score"] >= 50) & (scored_df["Trust_Score"] < 80)).sum())
    high_trust = int((scored_df["Trust_Score"] >= 80).sum())

    return {
        "total_transactions": total,
        "avg_trust_score": float(scored_df["Trust_Score"].mean()),
        "avg_fraud_probability": float(scored_df["Fraud_Probability"].mean()),
        "low_trust_count": low_trust,
        "medium_trust_count": medium_trust,
        "high_trust_count": high_trust,
        "low_trust_rate": low_trust / total if total else 0.0,
    }
