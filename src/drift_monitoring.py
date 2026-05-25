from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.database import get_connection, init_db


MONITORED_FEATURES = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "Fraud_Probability",
    "Trust_Score",
]


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _level(relative_change_pct: float, current_missing_rate: float) -> str:
    if current_missing_rate >= 0.60:
        return "High"
    if abs(relative_change_pct) >= 50:
        return "High"
    if abs(relative_change_pct) >= 25 or current_missing_rate >= 0.30:
        return "Medium"
    return "Low"


def _standardize_current_df(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    rename_map = {
        "amount": "TransactionAmt",
        "transaction_time": "TransactionDT",
        "fraud_probability": "Fraud_Probability",
        "trust_score": "Trust_Score",
        "source_transaction_id": "TransactionID",
    }
    for src, dst in rename_map.items():
        if src in current.columns and dst not in current.columns:
            current[dst] = current[src]
    return current


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    monitored_features: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compare current data with reference/training-like data using simple statistics.

    This is intentionally lightweight so it works in a local student project
    without installing a full drift-monitoring stack.
    """
    if reference_df is None or reference_df.empty or current_df is None or current_df.empty:
        return pd.DataFrame()

    ref = _standardize_current_df(reference_df)
    cur = _standardize_current_df(current_df)
    features = list(monitored_features or MONITORED_FEATURES)
    rows = []
    for feature in features:
        if feature not in cur.columns:
            continue
        cur_s = _as_numeric(cur[feature])
        ref_s = _as_numeric(ref[feature]) if feature in ref.columns else pd.Series(dtype=float)
        cur_valid = cur_s.dropna()
        ref_valid = ref_s.dropna()
        if cur_valid.empty:
            rows.append(
                {
                    "feature": feature,
                    "reference_mean": np.nan,
                    "current_mean": np.nan,
                    "relative_change_pct": np.nan,
                    "current_missing_rate": 1.0,
                    "drift_level": "High",
                    "recommendation": "Cột hiện tại không có giá trị hợp lệ, cần kiểm tra mapping hoặc nguồn dữ liệu.",
                }
            )
            continue
        ref_mean = float(ref_valid.mean()) if not ref_valid.empty else np.nan
        cur_mean = float(cur_valid.mean())
        if np.isnan(ref_mean) or abs(ref_mean) < 1e-9:
            relative_change = 0.0
        else:
            relative_change = ((cur_mean - ref_mean) / abs(ref_mean)) * 100.0
        missing_rate = float(cur_s.isna().mean())
        level = _level(relative_change, missing_rate)
        if level == "High":
            rec = "Drift cao, nên kiểm tra dữ liệu mới và cân nhắc đánh giá lại model."
        elif level == "Medium":
            rec = "Có thay đổi đáng chú ý, nên theo dõi thêm trong các batch tiếp theo."
        else:
            rec = "Phân phối hiện tại tương đối ổn định so với dữ liệu tham chiếu."
        rows.append(
            {
                "feature": feature,
                "reference_mean": round(ref_mean, 4) if not np.isnan(ref_mean) else np.nan,
                "current_mean": round(cur_mean, 4),
                "reference_std": round(float(ref_valid.std()), 4) if len(ref_valid) > 1 else np.nan,
                "current_std": round(float(cur_valid.std()), 4) if len(cur_valid) > 1 else np.nan,
                "relative_change_pct": round(float(relative_change), 2),
                "current_missing_rate": round(missing_rate, 4),
                "drift_level": level,
                "recommendation": rec,
            }
        )
    if not rows:
        return pd.DataFrame()
    order = {"High": 0, "Medium": 1, "Low": 2}
    result = pd.DataFrame(rows)
    result["_order"] = result["drift_level"].map(order).fillna(9)
    return result.sort_values(["_order", "feature"]).drop(columns=["_order"]).reset_index(drop=True)


def read_recent_transactions_as_current(db_path: str, limit: int = 5000) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT source_transaction_id, transaction_time, amount, product_code, device_hash, ip_address,
                   fraud_probability, trust_score, risk_level
            FROM transactions
            ORDER BY transaction_time DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
    finally:
        conn.close()
    return _standardize_current_df(df)


def drift_from_database(reference_df: pd.DataFrame, db_path: str, limit: int = 5000) -> pd.DataFrame:
    current = read_recent_transactions_as_current(db_path, limit=limit)
    return compute_drift_report(reference_df, current)


def drift_metrics(drift_df: pd.DataFrame) -> Dict[str, int]:
    if drift_df is None or drift_df.empty or "drift_level" not in drift_df.columns:
        return {"high": 0, "medium": 0, "low": 0, "total": 0}
    counts = drift_df["drift_level"].value_counts().to_dict()
    return {
        "high": int(counts.get("High", 0)),
        "medium": int(counts.get("Medium", 0)),
        "low": int(counts.get("Low", 0)),
        "total": int(len(drift_df)),
    }
