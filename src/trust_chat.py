from __future__ import annotations

import re
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    replacements = {
        "đ": "d",
        "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a", "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a", "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e", "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o", "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o", "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u", "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _contains(text: str, *keywords: str) -> bool:
    lower = _normalize(text)
    return any(_normalize(k) in lower for k in keywords)


def _fmt_num(value, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def _risk_label(score: float) -> str:
    if score >= 80:
        return "Tin cậy cao"
    if score >= 50:
        return "Cần theo dõi"
    return "Rủi ro cao"


def _df_columns(df: Optional[pd.DataFrame]) -> List[str]:
    return list(df.columns) if isinstance(df, pd.DataFrame) and not df.empty else []


def dataset_summary_text(
    reference_rows: int,
    feature_count: int,
    behavior_df: pd.DataFrame,
    upload_df: Optional[pd.DataFrame] = None,
    scored_df: Optional[pd.DataFrame] = None,
    upload_report: Optional[Dict] = None,
    db_stats: Optional[Dict] = None,
) -> str:
    user_count = len(behavior_df) if behavior_df is not None else 0
    high_risk = int((behavior_df["long_term_trust_score"] < 50).sum()) if behavior_df is not None and not behavior_df.empty and "long_term_trust_score" in behavior_df.columns else 0
    lines = [
        f"Dataset tham chiếu có {reference_rows:,} dòng giao dịch và {feature_count:,} feature model.",
        f"Database hành vi hiện có {user_count:,} hồ sơ user; trong đó {high_risk:,} user có Long-term Trust Score thấp hơn 50.",
    ]
    if db_stats:
        compact = []
        for key in ["users", "transactions", "predictions", "audit_logs", "user_feedback"]:
            if key in db_stats:
                compact.append(f"{key}: {_fmt_num(db_stats[key], 0)}")
        if compact:
            lines.append("Tóm tắt SQLite: " + "; ".join(compact) + ".")
    if isinstance(upload_df, pd.DataFrame) and not upload_df.empty:
        lines.append(f"CSV đang dùng có {len(upload_df):,} dòng và {len(upload_df.columns):,} cột.")
        if upload_report:
            mapped = upload_report.get("mapped_columns", "N/A")
            missing = upload_report.get("missing_model_feature_count", "N/A")
            lines.append(f"Kết quả validation CSV: {mapped} cột đã map, {missing} feature model được tự điền/bổ sung.")
    if isinstance(scored_df, pd.DataFrame) and not scored_df.empty and "Trust_Score" in scored_df.columns:
        high = int((scored_df["Trust_Score"] >= 80).sum())
        medium = int(((scored_df["Trust_Score"] >= 50) & (scored_df["Trust_Score"] < 80)).sum())
        low = int((scored_df["Trust_Score"] < 50).sum())
        lines.append(f"Kết quả scoring hiện tại: High Trust {high:,}, Medium {medium:,}, Low Trust {low:,}; Trust trung bình {_fmt_num(scored_df['Trust_Score'].mean())}/100.")
    return "\n".join(lines)


def _answer_user_lookup(q: str, behavior_df: pd.DataFrame, scored_df: Optional[pd.DataFrame]) -> Optional[str]:
    # Match U0001, user123, row ids, or explicit "user 12".
    candidates = re.findall(r"\bU\d{3,}\b|\buser[_\- ]?\d+\b|\bROW_\d{3,}\b", q, flags=re.IGNORECASE)
    if not candidates:
        m = re.search(r"user\s*(\d+)", q, flags=re.IGNORECASE)
        if m:
            candidates = [f"U{int(m.group(1)):04d}"]
    if not candidates:
        return None
    user_key = candidates[0].replace(" ", "").replace("-", "_")
    if isinstance(scored_df, pd.DataFrame) and not scored_df.empty and "UserID" in scored_df.columns:
        rows = scored_df[scored_df["UserID"].astype(str).str.lower() == user_key.lower()]
        if not rows.empty:
            msg = [
                f"User {user_key} trong CSV upload có {len(rows):,} giao dịch.",
            ]
            if "Trust_Score" in rows.columns:
                msg.append(f"Trust trung bình {_fmt_num(rows['Trust_Score'].mean())}/100, thấp nhất {_fmt_num(rows['Trust_Score'].min())}/100.")
                low_count = int((rows["Trust_Score"] < 50).sum())
                msg.append(f"Có {low_count:,} giao dịch Low Trust.")
            if "Fraud_Probability" in rows.columns:
                msg.append(f"Fraud Probability cao nhất {_fmt_num(rows['Fraud_Probability'].max(), 4)}.")
            return " ".join(msg)
    if isinstance(behavior_df, pd.DataFrame) and not behavior_df.empty and "user_id" in behavior_df.columns:
        rows = behavior_df[behavior_df["user_id"].astype(str).str.lower() == user_key.lower()]
        if not rows.empty:
            r = rows.iloc[0]
            return (
                f"User {r.get('user_id')} có Long-term Trust Score {_fmt_num(r.get('long_term_trust_score'))}/100 ({r.get('long_term_risk_level', 'N/A')}). "
                f"Tỷ lệ login thất bại 30 ngày: {_fmt_num(float(r.get('failed_login_rate_30d', 0)) * 100)}%. "
                f"Thay đổi thiết bị/IP: {int(r.get('device_change_count_30d', 0))}/{int(r.get('address_change_count_30d', 0))}. "
                f"Giải thích: {r.get('explanation', 'Chưa có giải thích.')}"
            )
    return f"Mình chưa tìm thấy {user_key} trong dữ liệu hiện tại. Hãy kiểm tra lại mã user hoặc upload CSV có cột UserID."


def _top_risk_answer(behavior_df: pd.DataFrame, scored_df: Optional[pd.DataFrame]) -> str:
    parts = []
    if isinstance(scored_df, pd.DataFrame) and not scored_df.empty and "Trust_Score" in scored_df.columns:
        worst = scored_df.sort_values("Trust_Score").head(5).copy()
        cols = [c for c in ["UserID", "TransactionID", "Fraud_Probability", "Trust_Score", "Risk_Level", "TransactionAmt"] if c in worst.columns]
        rows = []
        for _, r in worst[cols].iterrows():
            label = r.get("UserID", r.get("TransactionID", "dòng"))
            rows.append(f"{label}: Trust {_fmt_num(r.get('Trust_Score'))}/100, FraudProb {_fmt_num(r.get('Fraud_Probability'), 4)}")
        parts.append("Giao dịch rủi ro nhất trong CSV/scoring hiện tại: " + "; ".join(rows) + ".")
    if isinstance(behavior_df, pd.DataFrame) and not behavior_df.empty and "long_term_trust_score" in behavior_df.columns:
        row = behavior_df.sort_values("long_term_trust_score").iloc[0]
        parts.append(
            f"User rủi ro nổi bật trong database demo là {row.get('user_id')} với Long-term Trust Score {_fmt_num(row.get('long_term_trust_score'))}/100. "
            f"Lý do: {row.get('explanation', 'N/A')}"
        )
    return "\n".join(parts) if parts else "Chưa có đủ dữ liệu để xác định đối tượng rủi ro nhất."


def _column_answer(upload_df: Optional[pd.DataFrame], upload_report: Optional[Dict], feature_count: int) -> str:
    if isinstance(upload_df, pd.DataFrame) and not upload_df.empty:
        cols = list(upload_df.columns)
        preview = ", ".join(cols[:18]) + ("..." if len(cols) > 18 else "")
        msg = [f"CSV hiện có {len(cols):,} cột: {preview}."]
        if upload_report:
            missing = upload_report.get("missing_model_feature_count")
            mapped = upload_report.get("mapped_columns")
            warnings = upload_report.get("warnings") or []
            msg.append(f"So với schema model: {mapped} cột được map, {missing} feature còn thiếu/tự điền.")
            if warnings:
                msg.append("Cảnh báo đầu tiên: " + str(warnings[0]))
        return " ".join(msg)
    return f"Chưa có CSV upload trong phiên hiện tại. Model đang dùng {feature_count:,} feature đầu vào; hãy upload CSV ở trang Nhập dữ liệu để mình kiểm tra cột chi tiết."


def answer_trust_question(
    question: str,
    reference_rows: int,
    feature_count: int,
    behavior_df: pd.DataFrame,
    drift_df: Optional[pd.DataFrame] = None,
    upload_df: Optional[pd.DataFrame] = None,
    scored_df: Optional[pd.DataFrame] = None,
    upload_report: Optional[Dict] = None,
    db_stats: Optional[Dict] = None,
    current_page: str = "",
) -> str:
    q = (question or "").strip()
    nq = _normalize(q)
    if not q:
        return "Bạn có thể hỏi về dataset, CSV, Trust Score, user rủi ro, một user cụ thể, drift, SHAP, Docker/CI, feedback, audit log hoặc cách viết báo cáo."

    user_lookup = _answer_user_lookup(q, behavior_df, scored_df)
    if user_lookup:
        return user_lookup

    if _contains(q, "dataset", "dữ liệu", "du lieu", "data", "csv", "file", "bao nhiêu dòng", "tong dong", "số dòng", "so dong"):
        return dataset_summary_text(reference_rows, feature_count, behavior_df, upload_df, scored_df, upload_report, db_stats)

    if re.search(r"(^|\s)(chao|hello|hi|xin chao)(\s|$)", nq):
        return "Chào bạn. Mình là trợ lý rule-based của User Trust Platform. Bạn có thể hỏi về dữ liệu, rủi ro, Trust Score, user cụ thể, drift, SHAP, Docker/CI hoặc cách demo hệ thống."

    if _contains(q, "cột", "cot", "column", "schema", "mapping", "map", "thiếu cột", "thieu cot", "feature thiếu", "feature thieu"):
        return _column_answer(upload_df, upload_report, feature_count)

    if _contains(q, "trust score", "điểm", "diem", "tin cậy", "tin cay", "công thức", "cong thuc", "fraud probability", "xác suất", "xac suat"):
        return (
            "Trust Score giao dịch được tính theo công thức: Trust Score = (1 - Fraud Probability) × 100. "
            "Nếu điểm >= 80 thì Tin cậy cao, từ 50 đến dưới 80 là Cần theo dõi, dưới 50 là Rủi ro cao. "
            "Điểm hành vi dài hạn của user kết hợp ML Trust trung bình với tín hiệu login thất bại, đổi thiết bị/IP, giao dịch bất thường và feedback."
        )

    if _contains(q, "rủi ro nhất", "rui ro nhat", "nguy hiểm nhất", "nguy hiem nhat", "top risk", "user nào", "user nao", "ai rủi ro", "ai rui ro", "giao dịch rủi ro", "giao dich rui ro"):
        return _top_risk_answer(behavior_df, scored_df)

    if _contains(q, "rủi ro", "rui ro", "risk", "bất thường", "bat thuong", "anomaly", "nguy cơ", "nguy co", "cảnh báo", "canh bao"):
        return (
            _top_risk_answer(behavior_df, scored_df)
            + "\nCác tín hiệu rủi ro chính gồm: Trust Score thấp, Fraud Probability cao, login thất bại nhiều, đăng nhập ban đêm, thay đổi thiết bị/IP, giao dịch dồn dập và feedback tiêu cực."
        )

    if _contains(q, "drift", "lệch", "lech", "phân phối", "phan phoi", "monitoring", "giám sát", "giam sat"):
        if drift_df is not None and not drift_df.empty:
            high = int((drift_df["drift_level"] == "High").sum()) if "drift_level" in drift_df.columns else 0
            medium = int((drift_df["drift_level"] == "Medium").sum()) if "drift_level" in drift_df.columns else 0
            top_cols = []
            score_col = "psi" if "psi" in drift_df.columns else None
            if score_col:
                for _, r in drift_df.sort_values(score_col, ascending=False).head(3).iterrows():
                    top_cols.append(f"{r.get('feature')}: {score_col}={_fmt_num(r.get(score_col), 4)}")
            detail = " Top drift: " + "; ".join(top_cols) + "." if top_cols else ""
            return f"Drift report hiện ghi nhận {high} feature drift cao và {medium} feature drift trung bình.{detail} Nếu drift cao lặp lại, nên kiểm tra mapping CSV, nguồn dữ liệu mới và đánh giá/retrain model."
        return "Chưa có drift report. Hãy mở Monitoring & Drift hoặc chạy batch scoring để tạo dữ liệu giám sát."

    if _contains(q, "shap", "giải thích", "giai thich", "explain", "feature importance", "vì sao", "vi sao", "tại sao", "tai sao"):
        return (
            "Trang Giải thích mô hình ưu tiên SHAP TreeExplainer nếu môi trường có thư viện shap và ENABLE_SHAP=1. "
            "Nếu SHAP không khả dụng, hệ thống fallback sang feature importance để không làm hỏng demo. "
            "Bạn có thể chọn một dòng giao dịch để xem feature nào tác động mạnh nhất đến kết quả."
        )

    if _contains(q, "docker", "compose", "container", "chạy", "chay", "local"):
        return (
            "Để chạy Docker: mở terminal tại thư mục project, chạy `docker compose up --build`, sau đó mở `http://localhost:8501`. "
            "Backend FastAPI chạy ở `http://localhost:8000`. Chạy test bằng `docker compose exec backend pytest -q`."
        )

    if _contains(q, "ci", "github actions", "pipeline", "test", "pytest", "mlops"):
        return (
            "Phần MLOps cơ bản gồm Docker, Docker Compose, GitHub Actions CI, pytest, Monitoring & Drift, Model Card, audit log và feedback loop. "
            "CI sẽ cài dependencies rồi chạy `pytest -q`; khi pass có thể ghi trong báo cáo là hệ thống có kiểm thử tự động cơ bản."
        )

    if _contains(q, "feedback", "review", "human", "vòng lặp", "vong lap"):
        return (
            "Feedback là vòng lặp human-in-the-loop: người dùng/admin chọn user hoặc giao dịch, đưa ra quyết định review và ghi chú. "
            "Thông tin này được lưu vào SQLite và ghi audit log, dùng để minh họa dữ liệu phản hồi cho cải thiện model trong tương lai."
        )

    if _contains(q, "audit", "log", "truy vết", "truy vet"):
        return "Audit log dùng để lưu các thao tác quan trọng như đăng nhập, dự đoán, import dữ liệu, feedback và thao tác admin. Đây là phần giúp hệ thống minh bạch và dễ kiểm tra trong báo cáo."

    if _contains(q, "case study", "kịch bản", "kich ban", "demo"):
        return "Case study mô phỏng các tình huống user ổn định, user rủi ro cao và user nghi ngờ bị chiếm tài khoản. Từ đó có thể mở hồ sơ user, xem hành vi và gửi feedback/review."

    if _contains(q, "báo cáo", "bao cao", "viết", "viet", "trình bày", "trinh bay"):
        return (
            "Trong báo cáo nên trình bày theo thứ tự: giới thiệu bài toán, yêu cầu hệ thống, kiến trúc Streamlit/FastAPI/SQLite, dữ liệu, mô hình XGBoost, Trust Score, phân tích user, MLOps cơ bản, kiểm thử, demo giao diện, hạn chế và hướng phát triển. "
            "Nhấn mạnh đây là hệ thống demo hoàn chỉnh, MLOps ở mức lightweight."
        )

    if _contains(q, "hạn chế", "han che", "limitation", "thiếu", "thieu", "yếu", "yeu"):
        return (
            "Hạn chế chính: dữ liệu demo/synthetic, SQLite phù hợp demo hơn production, chưa có MLflow/DVC/model registry thật, chưa auto retraining và monitoring chưa realtime production. "
            "Có thể đưa các ý này vào phần hạn chế và hướng phát triển."
        )

    if _contains(q, "trang", "mở", "mo", "ở đâu", "o dau", "chức năng", "chuc nang", "sử dụng", "su dung"):
        return (
            "Luồng dùng nhanh: vào Nhập dữ liệu để upload CSV, sang Dự đoán giao dịch để xem từng dòng, Batch scoring để chấm hàng loạt, Xem theo từng user để xem hồ sơ/hành vi, Case study để demo tình huống, Monitoring & Drift để theo dõi model và Audit log để xem truy vết."
        )

    return (
        "Mình chưa hiểu chính xác câu hỏi này, nhưng có thể trả lời nhiều nhóm như: dataset/CSV, cột thiếu, Trust Score, user rủi ro, một user cụ thể, drift, SHAP, Docker, CI/MLOps, feedback, audit log, case study và cách viết báo cáo. "
        "Bạn có thể hỏi ví dụ: 'CSV này có bao nhiêu dòng?', 'User U0003 rủi ro không?', 'Top giao dịch rủi ro là gì?', hoặc 'MLOps trong đồ án gồm gì?'."
    )
