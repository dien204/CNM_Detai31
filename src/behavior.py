from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.database import get_connection, init_db
from src.utils import get_risk_level
from src.explainability import enhanced_explain_prediction


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _read_all(db_path: str):
    init_db(db_path)
    conn = get_connection(db_path)
    users = pd.read_sql_query("SELECT * FROM users", conn)
    logins = pd.read_sql_query("SELECT * FROM login_events", conn)
    transactions = pd.read_sql_query("SELECT * FROM transactions", conn)
    devices = pd.read_sql_query("SELECT * FROM devices", conn)
    addresses = pd.read_sql_query("SELECT * FROM addresses", conn)
    feedback = pd.read_sql_query("SELECT * FROM user_feedback", conn)
    conn.close()
    return users, logins, transactions, devices, addresses, feedback


def compute_user_behavior_scores(db_path: str, window_days: int = 30) -> pd.DataFrame:
    users, logins, transactions, devices, addresses, feedback = _read_all(db_path)
    if users.empty:
        return pd.DataFrame()

    now = pd.Timestamp.utcnow().tz_localize(None)
    since = now - pd.Timedelta(days=window_days)

    for df, col in [(logins, "login_time"), (transactions, "transaction_time"), (users, "created_at"), (feedback, "created_at")]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

    rows = []
    for _, user in users.iterrows():
        user_id = user["user_id"]
        u_logins = logins[logins["user_id"] == user_id].copy()
        u_tx = transactions[transactions["user_id"] == user_id].copy()
        u_devices = devices[devices["user_id"] == user_id]
        u_addresses = addresses[addresses["user_id"] == user_id]
        u_feedback = feedback[feedback["user_id"] == user_id].copy()

        recent_logins = u_logins[u_logins["login_time"] >= since] if not u_logins.empty else u_logins
        recent_tx = u_tx[u_tx["transaction_time"] >= since] if not u_tx.empty else u_tx
        recent_feedback = u_feedback[u_feedback["created_at"] >= since] if not u_feedback.empty else u_feedback

        login_count = int(len(recent_logins))
        failed_login_count = int((recent_logins.get("success", pd.Series(dtype=int)) == 0).sum()) if login_count else 0
        failed_login_rate = _safe_div(failed_login_count, login_count)
        night_login_count = 0
        if login_count:
            hours = recent_logins["login_time"].dt.hour
            night_login_count = int(((hours <= 5) | (hours >= 22)).sum())
        night_login_rate = _safe_div(night_login_count, login_count)

        tx_count = int(len(recent_tx))
        amount_sum = float(recent_tx["amount"].sum()) if tx_count and "amount" in recent_tx.columns else 0.0
        amount_mean = float(recent_tx["amount"].mean()) if tx_count and "amount" in recent_tx.columns else 0.0
        amount_std = float(recent_tx["amount"].std()) if tx_count > 1 and "amount" in recent_tx.columns else 0.0
        avg_trust = float(recent_tx["trust_score"].mean()) if tx_count and "trust_score" in recent_tx.columns else 85.0
        min_trust = float(recent_tx["trust_score"].min()) if tx_count and "trust_score" in recent_tx.columns else avg_trust
        max_fraud_prob = float(recent_tx["fraud_probability"].max()) if tx_count and "fraud_probability" in recent_tx.columns else 0.0
        low_trust_count = int((recent_tx["trust_score"] < 50).sum()) if tx_count and "trust_score" in recent_tx.columns else 0
        low_trust_rate = _safe_div(low_trust_count, tx_count)

        recent_device_count = int(recent_logins["device_hash"].nunique()) if login_count and "device_hash" in recent_logins.columns else int(len(u_devices))
        recent_address_count = int(recent_logins["ip_address"].nunique()) if login_count and "ip_address" in recent_logins.columns else int(len(u_addresses))
        device_change_count = max(0, recent_device_count - 1)
        address_change_count = max(0, recent_address_count - 1)

        high_velocity_hours = 0
        if tx_count and "transaction_time" in recent_tx.columns:
            per_hour = recent_tx.groupby(recent_tx["transaction_time"].dt.floor("h")).size()
            high_velocity_hours = int((per_hour >= 5).sum())

        high_amount_count = 0
        if tx_count and amount_mean > 0 and amount_std > 0:
            high_amount_count = int((recent_tx["amount"] > (amount_mean + 2.0 * amount_std)).sum())

        feedback_count = int(len(recent_feedback))
        negative_feedback_count = 0
        if feedback_count and "decision" in recent_feedback.columns:
            negative_feedback_count = int(recent_feedback["decision"].isin(["need_review", "watchlist", "confirmed_risk", "rejected"]).sum())

        tenure_days = 0
        if pd.notna(user.get("created_at")):
            tenure_days = max(0, int((now - pd.Timestamp(user["created_at"])).days))

        risk = 0.0
        risk += min(20.0, failed_login_rate * 55.0)
        risk += min(15.0, night_login_rate * 30.0)
        risk += min(18.0, device_change_count * 6.0)
        risk += min(18.0, address_change_count * 6.0)
        risk += min(12.0, high_velocity_hours * 4.0)
        risk += min(25.0, low_trust_rate * 45.0)
        risk += min(8.0, high_amount_count * 2.0)
        risk += min(10.0, negative_feedback_count * 3.5)
        if max_fraud_prob >= 0.50:
            risk += 12.0
        if tx_count >= 80:
            risk += 4.0
        risk = float(max(0.0, min(100.0, risk)))

        behavior_trust = round(100.0 - risk, 2)
        long_term_score = round((0.55 * avg_trust) + (0.45 * behavior_trust), 2)

        reasons = explain_behavior_reasons(
            {
                "failed_login_rate_30d": failed_login_rate,
                "night_login_rate_30d": night_login_rate,
                "device_change_count_30d": device_change_count,
                "address_change_count_30d": address_change_count,
                "high_velocity_hours_30d": high_velocity_hours,
                "low_trust_transaction_rate_30d": low_trust_rate,
                "max_fraud_probability_30d": max_fraud_prob,
                "high_amount_transaction_count_30d": high_amount_count,
                "negative_feedback_count_30d": negative_feedback_count,
            }
        )

        rows.append(
            {
                "user_id": user_id,
                "full_name": user.get("full_name", user_id),
                "email": user.get("email", ""),
                "status": user.get("status", "active"),
                "tenure_days": tenure_days,
                "login_count_30d": login_count,
                "failed_login_count_30d": failed_login_count,
                "failed_login_rate_30d": round(failed_login_rate, 4),
                "night_login_rate_30d": round(night_login_rate, 4),
                "transaction_count_30d": tx_count,
                "transaction_amount_sum_30d": round(amount_sum, 2),
                "transaction_amount_mean_30d": round(amount_mean, 2),
                "transaction_amount_std_30d": round(amount_std, 2),
                "high_amount_transaction_count_30d": high_amount_count,
                "unique_devices_30d": recent_device_count,
                "unique_addresses_30d": recent_address_count,
                "device_change_count_30d": device_change_count,
                "address_change_count_30d": address_change_count,
                "high_velocity_hours_30d": high_velocity_hours,
                "ml_avg_trust_score_30d": round(avg_trust, 2),
                "min_trust_score_30d": round(min_trust, 2),
                "max_fraud_probability_30d": round(max_fraud_prob, 4),
                "low_trust_transaction_rate_30d": round(low_trust_rate, 4),
                "feedback_count_30d": feedback_count,
                "negative_feedback_count_30d": negative_feedback_count,
                "behavior_risk_score": round(risk, 2),
                "behavior_trust_score": behavior_trust,
                "long_term_trust_score": long_term_score,
                "long_term_risk_level": get_risk_level(long_term_score),
                "explanation": "; ".join(reasons) if reasons else "Hành vi ổn định trong cửa sổ phân tích",
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(["long_term_trust_score", "behavior_risk_score"], ascending=[True, False])


def explain_behavior_reasons(row: Dict) -> List[str]:
    reasons: List[str] = []
    if float(row.get("failed_login_rate_30d", 0)) >= 0.18:
        reasons.append("Tỷ lệ đăng nhập thất bại cao")
    if float(row.get("night_login_rate_30d", 0)) >= 0.25:
        reasons.append("Nhiều đăng nhập vào khung giờ bất thường")
    if int(row.get("device_change_count_30d", 0)) >= 2:
        reasons.append("Thay đổi thiết bị thường xuyên")
    if int(row.get("address_change_count_30d", 0)) >= 2:
        reasons.append("Thay đổi địa chỉ/IP thường xuyên")
    if int(row.get("high_velocity_hours_30d", 0)) >= 1:
        reasons.append("Có cụm giao dịch tần suất cao trong thời gian ngắn")
    if float(row.get("low_trust_transaction_rate_30d", 0)) >= 0.15:
        reasons.append("Tỷ lệ giao dịch điểm tin cậy thấp cao")
    if float(row.get("max_fraud_probability_30d", 0)) >= 0.50:
        reasons.append("Từng có giao dịch có xác suất gian lận cao")
    if int(row.get("high_amount_transaction_count_30d", 0)) >= 2:
        reasons.append("Có nhiều giao dịch số tiền cao bất thường")
    if int(row.get("negative_feedback_count_30d", 0)) >= 1:
        reasons.append("Có phản hồi kiểm duyệt rủi ro từ hệ thống")
    return reasons


def user_timeline(db_path: str, user_id: str) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    tx = pd.read_sql_query(
        "SELECT transaction_time, amount, trust_score, fraud_probability FROM transactions WHERE user_id = ?",
        conn,
        params=(user_id,),
    )
    login = pd.read_sql_query(
        "SELECT login_time, device_hash, ip_address, success FROM login_events WHERE user_id = ?",
        conn,
        params=(user_id,),
    )
    conn.close()

    frames = []
    if not tx.empty:
        tx["date"] = pd.to_datetime(tx["transaction_time"], errors="coerce").dt.date
        daily_tx = tx.groupby("date").agg(
            transaction_count=("amount", "size"),
            amount_sum=("amount", "sum"),
            avg_trust_score=("trust_score", "mean"),
            max_fraud_probability=("fraud_probability", "max"),
        ).reset_index()
        frames.append(daily_tx)

    if not login.empty:
        login["date"] = pd.to_datetime(login["login_time"], errors="coerce").dt.date
        daily_login = login.groupby("date").agg(
            login_count=("success", "size"),
            failed_login_count=("success", lambda s: int((s == 0).sum())),
            unique_devices=("device_hash", "nunique"),
            unique_addresses=("ip_address", "nunique"),
        ).reset_index()
        frames.append(daily_login)

    if not frames:
        return pd.DataFrame()

    timeline = frames[0]
    for frame in frames[1:]:
        timeline = timeline.merge(frame, on="date", how="outer")
    timeline = timeline.sort_values("date").fillna(0)
    return timeline


def read_audit_logs(db_path: str, limit: int = 200) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM audit_logs ORDER BY log_id DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    conn.close()
    return df


def read_feedback(db_path: str, limit: int = 200, reviewer: str | None = None) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    if reviewer:
        df = pd.read_sql_query(
            "SELECT * FROM user_feedback WHERE reviewer = ? ORDER BY feedback_id DESC LIMIT ?",
            conn,
            params=(reviewer, limit),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM user_feedback ORDER BY feedback_id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    conn.close()
    return df


def user_profile(db_path: str, user_id: str, limit: int = 50, reviewer: str | None = None) -> Dict[str, pd.DataFrame]:
    init_db(db_path)
    conn = get_connection(db_path)
    if reviewer:
        feedback_df = pd.read_sql_query(
            "SELECT * FROM user_feedback WHERE user_id = ? AND reviewer = ? ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(user_id, reviewer, limit),
        )
    else:
        feedback_df = pd.read_sql_query(
            "SELECT * FROM user_feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(user_id, limit),
        )
    result = {
        "user": pd.read_sql_query("SELECT * FROM users WHERE user_id = ?", conn, params=(user_id,)),
        "devices": pd.read_sql_query("SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC", conn, params=(user_id,)),
        "addresses": pd.read_sql_query("SELECT * FROM addresses WHERE user_id = ? ORDER BY last_seen DESC", conn, params=(user_id,)),
        "logins": pd.read_sql_query("SELECT * FROM login_events WHERE user_id = ? ORDER BY login_time DESC LIMIT ?", conn, params=(user_id, limit)),
        "transactions": pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? ORDER BY transaction_time DESC LIMIT ?", conn, params=(user_id, limit)),
        "risk_transactions": pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? ORDER BY trust_score ASC LIMIT ?", conn, params=(user_id, min(limit, 20))),
        "feedback": feedback_df,
    }
    conn.close()
    return result


def detect_anomalies(db_path: str, window_days: int = 30) -> pd.DataFrame:
    scores = compute_user_behavior_scores(db_path, window_days=window_days)
    if scores.empty:
        return pd.DataFrame()
    rows = []
    rules = [
        ("FAILED_LOGIN_RATE", "Tỷ lệ đăng nhập thất bại cao", "failed_login_rate_30d", 0.20, "High"),
        ("NIGHT_LOGIN_RATE", "Đăng nhập nhiều vào khung giờ bất thường", "night_login_rate_30d", 0.30, "Medium"),
        ("DEVICE_CHANGE", "Thay đổi thiết bị thường xuyên", "device_change_count_30d", 3, "Medium"),
        ("ADDRESS_CHANGE", "Thay đổi địa chỉ/IP thường xuyên", "address_change_count_30d", 3, "Medium"),
        ("HIGH_VELOCITY", "Có cụm giao dịch tần suất cao", "high_velocity_hours_30d", 1, "High"),
        ("LOW_TRUST_RATE", "Tỷ lệ giao dịch điểm tin cậy thấp cao", "low_trust_transaction_rate_30d", 0.15, "High"),
        ("MAX_FRAUD_PROB", "Từng có giao dịch có xác suất gian lận cao", "max_fraud_probability_30d", 0.50, "High"),
        ("NEGATIVE_FEEDBACK", "Có phản hồi kiểm duyệt rủi ro", "negative_feedback_count_30d", 1, "High"),
    ]
    for _, row in scores.iterrows():
        for rule_id, desc, col, threshold, severity in rules:
            value = row.get(col, 0)
            try:
                is_hit = float(value) >= float(threshold)
            except Exception:
                is_hit = False
            if is_hit:
                rows.append(
                    {
                        "user_id": row["user_id"],
                        "full_name": row.get("full_name", ""),
                        "rule_id": rule_id,
                        "severity": severity,
                        "value": value,
                        "threshold": threshold,
                        "long_term_trust_score": row.get("long_term_trust_score"),
                        "description": desc,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        result["severity_order"] = result["severity"].map(severity_order).fillna(3)
        result = result.sort_values(["severity_order", "long_term_trust_score"], ascending=[True, True]).drop(columns=["severity_order"])
    return result


def monitoring_summary(db_path: str) -> Dict[str, pd.DataFrame | Dict]:
    init_db(db_path)
    conn = get_connection(db_path)
    predictions = pd.read_sql_query("SELECT * FROM predictions", conn)
    audit = pd.read_sql_query("SELECT * FROM audit_logs", conn)
    tx = pd.read_sql_query("SELECT * FROM transactions", conn)
    feedback = pd.read_sql_query("SELECT * FROM user_feedback", conn)
    models = pd.read_sql_query("SELECT * FROM model_registry WHERE is_active=1 ORDER BY model_id DESC LIMIT 5", conn)
    conn.close()

    metrics = {
        "prediction_count": int(len(predictions)),
        "audit_count": int(len(audit)),
        "feedback_count": int(len(feedback)),
        "avg_trust_score": float(predictions["trust_score"].mean()) if not predictions.empty else 0.0,
        "high_risk_prediction_count": int((predictions["trust_score"] < 50).sum()) if not predictions.empty else 0,
        "transaction_count": int(len(tx)),
    }

    if not predictions.empty:
        predictions["created_date"] = pd.to_datetime(predictions["created_at"], errors="coerce").dt.date
        daily_predictions = predictions.groupby("created_date").agg(
            prediction_count=("prediction_id", "size"),
            avg_trust_score=("trust_score", "mean"),
            high_risk_count=("trust_score", lambda s: int((s < 50).sum())),
        ).reset_index()
        risk_distribution = predictions["risk_level"].value_counts().rename_axis("risk_level").to_frame("count").reset_index()
    else:
        daily_predictions = pd.DataFrame()
        risk_distribution = pd.DataFrame()

    if not audit.empty:
        audit["created_date"] = pd.to_datetime(audit["created_at"], errors="coerce").dt.date
        audit_daily = audit.groupby("created_date").size().reset_index(name="audit_count")
        audit_actions = audit["action"].value_counts().rename_axis("action").to_frame("count").reset_index()
    else:
        audit_daily = pd.DataFrame()
        audit_actions = pd.DataFrame()

    return {
        "metrics": metrics,
        "daily_predictions": daily_predictions,
        "risk_distribution": risk_distribution,
        "audit_daily": audit_daily,
        "audit_actions": audit_actions,
        "active_models": models,
    }


def explain_model_prediction(model, feature_vector: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Explain one prediction.

    The function uses real SHAP TreeExplainer when the optional ``shap``
    package is available. If SHAP is not installed or fails in the local demo
    environment, it automatically falls back to a lightweight feature-importance
    explanation so the web app still works during presentation.
    """
    return enhanced_explain_prediction(model, feature_vector, top_n=top_n)
