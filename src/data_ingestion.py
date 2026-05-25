from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

CONFIG_PATH = "configs/schema_mapping.json"
REQUIRED_MODEL_COLUMNS = ["TransactionAmt"]
IMPORTANT_MODEL_COLUMNS = ["TransactionID", "TransactionDT", "ProductCD", "DeviceType", "DeviceInfo", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    mapping: Dict[str, str]
    mapping_table: pd.DataFrame
    report: Dict
    valid: bool


def _normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_schema_mapping(path: str = CONFIG_PATH) -> Dict[str, List[str]]:
    """Load schema mapping from cwd or project root.

    Docker/Render normally runs from the project root, but local execution or
    tests can have a different working directory. Falling back to the project
    root prevents the upload page from recognizing only a few columns.
    """
    candidates = [path]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(project_root, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

def _with_dynamic_schema_aliases(schema: Dict[str, List[str]], feature_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Extend the configured schema with model and behavior columns.

    The manual schema is intentionally small, but uploaded CSV files can already
    contain many real feature names such as C1, D1, V12, id_01 or custom behavior
    signals. Adding them dynamically makes the mapping report recognize many more
    useful columns instead of marking them as ignored.
    """
    enhanced: Dict[str, List[str]] = {str(k): list(v) for k, v in (schema or {}).items()}

    def add_aliases(standard: str, aliases: List[str]) -> None:
        bucket = enhanced.setdefault(standard, [])
        for alias in [standard] + aliases:
            if alias not in bucket:
                bucket.append(alias)

    for col in feature_columns or []:
        snake = _normalize_name(col)
        add_aliases(str(col), [snake, snake.replace("_", " "), str(col).lower(), str(col).upper()])

    extra_aliases = {
        "TransactionID": ["transaction_id", "txn_id", "ma_giao_dich", "mã giao dịch", "id_giao_dich"],
        "TransactionDT": ["transaction_dt", "transaction_time", "timestamp", "time", "thoi_gian", "thời gian"],
        "UserID": ["user", "user id", "userid", "user_id", "customer", "customer id", "customer_id", "account", "account_id", "ma user", "mã user", "ma khach hang", "mã khách hàng"],
        "TransactionDate": ["transaction_date", "ngay gio", "ngày giờ", "ngay_gio", "created_at", "date", "datetime"],
        "IPAddress": ["ip", "ip_address", "ip address", "dia chi ip", "địa chỉ ip"],
        "IPCountry": ["ip_country", "country", "quoc gia", "quốc gia"],
        "login_count_30d": ["login_count", "so_lan_dang_nhap", "số lần đăng nhập", "dang_nhap_30_ngay"],
        "failed_login_count_30d": ["failed_login", "failed_login_count", "login_failed", "so_lan_dang_nhap_that_bai", "số lần đăng nhập thất bại"],
        "night_login_count_30d": ["night_login", "night_login_count", "dang_nhap_ban_dem", "đăng nhập ban đêm"],
        "unique_device_count_30d": ["unique_device_count", "device_count", "so_thiet_bi", "số thiết bị", "unique_devices_30d"],
        "unique_ip_count_30d": ["unique_ip_count", "ip_count", "so_ip", "số ip", "unique_addresses_30d"],
        "avg_transaction_amt_30d": ["avg_amount", "avg_transaction_amount", "gia_tri_tb_30_ngay", "giá trị tb 30 ngày"],
        "max_transaction_amt_30d": ["max_amount", "max_transaction_amount", "gia_tri_max_30_ngay"],
        "transaction_count_30d": ["transaction_count", "so_giao_dich", "số giao dịch", "giao_dich_30_ngay"],
        "chargeback_count_90d": ["chargeback", "chargeback_count", "khieu_nai", "khiếu nại", "hoan_tien"],
        "feedback_risk_flag": ["feedback_risk", "risk_feedback", "co_feedback_rui_ro", "cờ feedback rủi ro"],
        "account_age_days": ["account_age", "tuoi_tai_khoan", "tuổi tài khoản", "account_days"],
        "isFraud": ["is_fraud", "fraud", "label", "target", "nhan", "nhãn", "gian_lan", "gian lận"],
    }
    for standard, aliases in extra_aliases.items():
        add_aliases(standard, aliases)

    return enhanced


def read_csv_safely(uploaded_file_or_path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]
    separators = [",", ";", "\t", "|"]
    last_error = None
    for enc in encodings:
        for sep in separators:
            try:
                if hasattr(uploaded_file_or_path, "seek"):
                    uploaded_file_or_path.seek(0)
                df = pd.read_csv(uploaded_file_or_path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as exc:
                last_error = exc
    if hasattr(uploaded_file_or_path, "seek"):
        uploaded_file_or_path.seek(0)
    try:
        return pd.read_csv(uploaded_file_or_path)
    except Exception as exc:
        raise ValueError(f"Không thể đọc CSV. Lỗi gần nhất: {last_error or exc}") from exc


def infer_column_mapping(df: pd.DataFrame, schema_mapping: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    schema_mapping = schema_mapping or load_schema_mapping()
    norm_input = {_normalize_name(col): col for col in df.columns}
    mapping: Dict[str, str] = {}

    for standard_col, aliases in schema_mapping.items():
        candidate_norms = [_normalize_name(standard_col)] + [_normalize_name(a) for a in aliases]
        for normalized in candidate_norms:
            if normalized in norm_input:
                mapping[standard_col] = norm_input[normalized]
                break
        if standard_col not in mapping:
            best_score = 0.0
            best_col = None
            for input_norm, original_col in norm_input.items():
                score = max(SequenceMatcher(None, input_norm, cand).ratio() for cand in candidate_norms)
                if score > best_score:
                    best_score = score
                    best_col = original_col
            if best_score >= 0.86 and best_col:
                mapping[standard_col] = best_col

    return mapping


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9,\.\-]", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_uploaded_data(df: pd.DataFrame, mapping: Dict[str, str], reference_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for target_col, source_col in mapping.items():
        if source_col in df.columns:
            out[target_col] = df[source_col]

    if "TransactionID" not in out.columns:
        out["TransactionID"] = range(1, len(out) + 1)

    numeric_cols = [
        "TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
        "login_count_30d", "failed_login_count_30d", "night_login_count_30d", "unique_device_count_30d",
        "unique_ip_count_30d", "avg_transaction_amt_30d", "max_transaction_amt_30d", "transaction_count_30d",
        "chargeback_count_90d", "feedback_risk_flag", "account_age_days", "isFraud",
    ]
    numeric_cols += [c for c in out.columns if re.match(r"^(C|D|V)\d+$", str(c)) or re.match(r"^id_?\d+$", str(c), re.IGNORECASE) or str(c).startswith("dist")]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = _coerce_numeric(out[col])

    # Lightweight categorical normalization. The model pipeline can fill/encode later.
    for col in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "UserID", "TransactionDate", "IPAddress", "IPCountry"]:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": None, "None": None})

    if reference_df is not None:
        for col in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]:
            if col in out.columns and col in reference_df.columns and pd.api.types.is_numeric_dtype(reference_df[col]):
                # The reference/demo data stores these categorical fields as numeric codes.
                # However, user uploads may contain raw text values such as "W", "visa",
                # "desktop", "gmail.com". Converting those directly with errors="coerce"
                # turns the whole column into NaN, which creates the false warning
                # "Cột ... thiếu 100% dữ liệu" even when the uploaded CSV is filled.
                converted = pd.to_numeric(out[col], errors="coerce")
                if converted.notna().any() or out[col].isna().all():
                    out[col] = converted

    return out


def validate_standard_schema(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []

    for col in REQUIRED_MODEL_COLUMNS:
        if col not in df.columns:
            errors.append(f"Thiếu cột bắt buộc: {col}")
        elif df[col].isna().all():
            errors.append(f"Cột bắt buộc không có giá trị hợp lệ: {col}")
        elif (pd.to_numeric(df[col], errors="coerce") < 0).any():
            errors.append(f"Cột {col} có giá trị âm")

    for col in IMPORTANT_MODEL_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Thiếu cột nên có: {col}. Hệ thống sẽ dùng giá trị mặc định nếu có thể.")

    missing_rate = {}
    for col in df.columns:
        rate = float(df[col].isna().mean()) if len(df) else 0.0
        missing_rate[col] = round(rate, 4)
        if rate >= 0.7:
            warnings.append(f"Cột {col} thiếu {rate:.0%} dữ liệu")

    missing_model_features = []
    if feature_columns:
        missing_model_features = [c for c in feature_columns if c not in df.columns]

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings[:30],
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_rate": missing_rate,
        "missing_model_feature_count": int(len(missing_model_features)),
        "missing_model_features_sample": missing_model_features[:20],
    }


def build_mapping_table(input_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    reverse = {v: k for k, v in mapping.items()}
    rows = []
    for col in input_df.columns:
        rows.append(
            {
                "Cột trong file": col,
                "Cột chuẩn hệ thống": reverse.get(col, "Bỏ qua"),
                "Trạng thái": "Đã map" if col in reverse else "Không dùng",
            }
        )
    return pd.DataFrame(rows)


def process_uploaded_dataframe(df: pd.DataFrame, reference_df: Optional[pd.DataFrame], feature_columns: Optional[List[str]]) -> IngestionResult:
    schema = _with_dynamic_schema_aliases(load_schema_mapping(), feature_columns=feature_columns)
    mapping = infer_column_mapping(df, schema)
    standardized = standardize_uploaded_data(df, mapping, reference_df=reference_df)
    report = validate_standard_schema(standardized, feature_columns=feature_columns)
    mapping_table = build_mapping_table(df, mapping)
    report["mapped_columns"] = int(len(mapping))
    report["ignored_columns"] = int(len(df.columns) - len(set(mapping.values())))
    return IngestionResult(standardized, mapping, mapping_table, report, bool(report["valid"]))


def process_uploaded_csv(uploaded_file, reference_df: Optional[pd.DataFrame], feature_columns: Optional[List[str]]) -> IngestionResult:
    raw = read_csv_safely(uploaded_file)
    return process_uploaded_dataframe(raw, reference_df=reference_df, feature_columns=feature_columns)
