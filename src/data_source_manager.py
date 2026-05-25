from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

import pandas as pd

from src.data_ingestion import (
    IngestionResult,
    build_mapping_table,
    infer_column_mapping,
    read_csv_safely,
    standardize_uploaded_data,
    validate_standard_schema,
)
from src.database import DB_PATH, get_connection, init_db, now_iso, _ensure_feedback_user, _feedback_transaction_values

SOURCE_LABELS = {
    "transactions": "Giao dịch",
    "login_events": "Lịch sử đăng nhập",
    "users": "Người dùng",
    "devices_ip": "Thiết bị/IP",
    "feedback": "Feedback/Review",
}

SOURCE_SCHEMAS: Dict[str, Dict[str, list[str]]] = {
    "users": {
        "user_id": ["UserID", "user_id", "customer_id", "account_id", "ma_nguoi_dung", "Mã người dùng"],
        "full_name": ["full_name", "name", "Tên", "Ho ten", "Họ tên", "ten_nguoi_dung", "customer_name"],
        "email": ["email", "Email", "mail", "user_email"],
        "created_at": ["created_at", "created", "join_date", "ngay_tao", "Ngày tạo"],
        "status": ["status", "trang_thai", "Trạng thái"],
    },
    "login_events": {
        "user_id": ["UserID", "user_id", "customer_id", "Mã người dùng", "ma_nguoi_dung"],
        "login_time": ["login_time", "timestamp", "time", "Thời gian đăng nhập", "thoi_gian_dang_nhap", "ngay_dang_nhap"],
        "device_hash": ["device_hash", "device_id", "device", "Thiết bị", "thiet_bi"],
        "ip_address": ["ip_address", "ip", "Địa chỉ IP", "dia_chi_ip", "address_ip"],
        "success": ["success", "is_success", "login_success", "Đăng nhập thành công", "thanh_cong"],
        "risk_hint": ["risk_hint", "risk", "note", "Ghi chú", "ghi_chu"],
    },
    "devices_ip": {
        "user_id": ["UserID", "user_id", "customer_id", "Mã người dùng", "ma_nguoi_dung"],
        "device_hash": ["device_hash", "device_id", "device", "Thiết bị", "thiet_bi"],
        "device_type": ["device_type", "device_kind", "Loại thiết bị", "loai_thiet_bi"],
        "ip_address": ["ip_address", "ip", "Địa chỉ IP", "dia_chi_ip"],
        "city": ["city", "thanh_pho", "Thành phố"],
        "country": ["country", "quoc_gia", "Quốc gia"],
        "first_seen": ["first_seen", "seen_at", "created_at", "lan_dau", "first_time"],
        "last_seen": ["last_seen", "updated_at", "lan_cuoi", "last_time"],
    },
    "feedback": {
        "user_id": ["UserID", "user_id", "customer_id", "Mã người dùng", "ma_nguoi_dung"],
        "transaction_id": ["transaction_id", "TransactionID", "txn_id", "Mã giao dịch"],
        "reviewer": ["reviewer", "admin", "nguoi_duyet", "Người duyệt"],
        "decision": ["decision", "label", "status", "Quyết định", "ket_qua"],
        "note": ["note", "Ghi chú", "ghi_chu", "comment"],
        "created_at": ["created_at", "review_time", "time", "Thời gian"],
    },
}


def ensure_data_sources_table(db_path: str = DB_PATH) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            file_name TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            valid_rows INTEGER NOT NULL DEFAULT 0,
            invalid_rows INTEGER NOT NULL DEFAULT 0,
            uploaded_by TEXT,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL,
            validation_report TEXT,
            storage_path TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(data_sources)").fetchall()}
    if "storage_path" not in columns:
        conn.execute("ALTER TABLE data_sources ADD COLUMN storage_path TEXT")
    conn.commit()
    conn.close()


def read_data_sources(db_path: str = DB_PATH, limit: int = 200, uploaded_by: Optional[str] = None) -> pd.DataFrame:
    ensure_data_sources_table(db_path)
    conn = get_connection(db_path)
    if uploaded_by:
        df = pd.read_sql_query(
            "SELECT * FROM data_sources WHERE uploaded_by = ? ORDER BY source_id DESC LIMIT ?",
            conn,
            params=(uploaded_by, limit),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM data_sources ORDER BY source_id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    conn.close()
    return df


def record_data_source(
    source_name: str,
    source_type: str,
    file_name: str,
    row_count: int,
    valid_rows: int,
    invalid_rows: int,
    uploaded_by: str,
    status: str,
    validation_report: Dict,
    db_path: str = DB_PATH,
    storage_path: Optional[str] = None,
) -> int:
    ensure_data_sources_table(db_path)
    conn = get_connection(db_path)
    cur = conn.execute(
        """
        INSERT INTO data_sources(
            source_name, source_type, file_name, row_count, valid_rows, invalid_rows,
            uploaded_by, uploaded_at, status, validation_report, storage_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_name,
            source_type,
            file_name,
            int(row_count),
            int(valid_rows),
            int(invalid_rows),
            uploaded_by,
            now_iso(),
            status,
            json.dumps(validation_report, ensure_ascii=False),
            storage_path,
        ),
    )
    source_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return source_id


def _parse_time(value, default: Optional[str] = None) -> str:
    if pd.isna(value):
        return default or now_iso()
    if isinstance(value, (int, float)):
        # Treat numeric values as seconds offset from a stable demo origin.
        base = datetime(2026, 1, 1)
        try:
            return (base + timedelta(seconds=float(value))).isoformat(timespec="seconds")
        except Exception:
            return default or now_iso()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return default or now_iso()
    return parsed.to_pydatetime().replace(tzinfo=None).isoformat(timespec="seconds")


def _truthy(value) -> int:
    if pd.isna(value):
        return 1
    text = str(value).strip().lower()
    if text in {"0", "false", "fail", "failed", "no", "n", "that_bai", "thất bại", "khong", "không"}:
        return 0
    return 1


def _source_mapping_table(input_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return build_mapping_table(input_df, mapping)


def prepare_source_upload(
    uploaded_file,
    source_type: str,
    reference_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame, Dict, bool]:
    # The Upload page now defaults to transaction CSV and no longer exposes
    # a source-type selector. Guard against stale session/UI values that may
    # pass None here after a deploy or browser refresh.
    source_type = source_type or "transactions"
    raw = read_csv_safely(uploaded_file)
    if source_type == "transactions":
        mapping = infer_column_mapping(raw)
        standardized = standardize_uploaded_data(raw, mapping, reference_df=reference_df)
        report = validate_standard_schema(standardized, feature_columns=feature_columns)
        report["mapped_columns"] = int(len(mapping))
        report["ignored_columns"] = int(len(raw.columns) - len(set(mapping.values())))
        return standardized, mapping, _source_mapping_table(raw, mapping), report, bool(report["valid"])

    schema = SOURCE_SCHEMAS.get(source_type, {})
    mapping = infer_column_mapping(raw, schema)
    out = pd.DataFrame(index=raw.index)
    for target, source_col in mapping.items():
        if source_col in raw.columns:
            out[target] = raw[source_col]

    errors = []
    warnings = []
    required = {
        "users": ["full_name", "email"],
        "login_events": ["user_id", "login_time"],
        "devices_ip": ["user_id"],
        "feedback": ["user_id", "decision"],
    }.get(source_type, [])
    for col in required:
        if col not in out.columns or out[col].isna().all():
            errors.append(f"Thiếu cột bắt buộc: {col}")
    if source_type == "devices_ip" and "device_hash" not in out.columns and "ip_address" not in out.columns:
        errors.append("Thiếu device_hash hoặc ip_address")
    for col in out.columns:
        if len(out) and out[col].isna().mean() >= 0.7:
            warnings.append(f"Cột {col} thiếu nhiều dữ liệu")

    report = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings[:30],
        "row_count": int(len(out)),
        "column_count": int(out.shape[1]),
        "mapped_columns": int(len(mapping)),
        "ignored_columns": int(len(raw.columns) - len(set(mapping.values()))),
    }
    return out, mapping, _source_mapping_table(raw, mapping), report, bool(report["valid"])


def _save_uploaded_dataframe(storage_df: Optional[pd.DataFrame], uploaded_by: str, file_name: str, source_type: str) -> Optional[str]:
    if storage_df is None or storage_df.empty:
        return None
    base_dir = Path("data/upload_cache") / (uploaded_by or "anonymous")
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in (file_name or f"{source_type}.csv"))
    out_path = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    storage_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)


def load_saved_source_dataframe(source_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    ensure_data_sources_table(db_path)
    conn = get_connection(db_path)
    row = conn.execute("SELECT storage_path FROM data_sources WHERE source_id = ?", (int(source_id),)).fetchone()
    conn.close()
    if not row or not row[0]:
        return pd.DataFrame()
    storage_path = Path(row[0])
    if not storage_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(storage_path)
    except Exception:
        return pd.DataFrame()



def delete_data_source(source_id: int, uploaded_by: Optional[str] = None, db_path: str = DB_PATH) -> bool:
    ensure_data_sources_table(db_path)
    conn = get_connection(db_path)
    if uploaded_by:
        row = conn.execute(
            "SELECT storage_path FROM data_sources WHERE source_id = ? AND uploaded_by = ?",
            (int(source_id), uploaded_by),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT storage_path FROM data_sources WHERE source_id = ?",
            (int(source_id),),
        ).fetchone()
    if not row:
        conn.close()
        return False
    storage_path = row[0]
    if uploaded_by:
        conn.execute("DELETE FROM data_sources WHERE source_id = ? AND uploaded_by = ?", (int(source_id), uploaded_by))
    else:
        conn.execute("DELETE FROM data_sources WHERE source_id = ?", (int(source_id),))
    conn.commit()
    conn.close()
    if storage_path:
        try:
            Path(storage_path).unlink(missing_ok=True)
        except Exception:
            pass
    return True


def import_source_dataframe(
    df: pd.DataFrame,
    source_type: str,
    source_name: str,
    file_name: str,
    uploaded_by: str,
    db_path: str = DB_PATH,
    validation_report: Optional[Dict] = None,
    storage_df: Optional[pd.DataFrame] = None,
) -> Tuple[int, int]:
    """Import a standardized dataframe into the local app database.

    Returns: (source_id, inserted_rows)
    """
    validation_report = validation_report or {}
    ensure_data_sources_table(db_path)
    init_db(db_path)
    conn = get_connection(db_path)
    inserted = 0
    now = now_iso()

    if source_type == "users":
        for _, row in df.iterrows():
            email = str(row.get("email", "")).strip().lower()
            full_name = str(row.get("full_name", email or "Unknown User")).strip() or "Unknown User"
            user_id = str(row.get("user_id", "")).strip() or ("USR_" + str(abs(hash(email or full_name)))[:10])
            created_at = _parse_time(row.get("created_at"), now)
            status = str(row.get("status", "active")).strip() or "active"
            conn.execute(
                "INSERT OR REPLACE INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, ?)",
                (user_id, full_name, email, created_at, status),
            )
            inserted += 1

    elif source_type == "login_events":
        for _, row in df.iterrows():
            user_id = str(row.get("user_id", "UNKNOWN_USER")).strip() or "UNKNOWN_USER"
            login_time = _parse_time(row.get("login_time"), now)
            device_hash = str(row.get("device_hash", "unknown_device")).strip() or "unknown_device"
            ip_address = str(row.get("ip_address", "0.0.0.0")).strip() or "0.0.0.0"
            success = _truthy(row.get("success", 1))
            risk_hint = str(row.get("risk_hint", "imported")).strip()
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, 'active')",
                (user_id, user_id, None, now),
            )
            conn.execute(
                "INSERT INTO login_events(user_id, login_time, device_hash, ip_address, success, risk_hint) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, login_time, device_hash, ip_address, success, risk_hint),
            )
            inserted += 1

    elif source_type == "devices_ip":
        for _, row in df.iterrows():
            user_id = str(row.get("user_id", "UNKNOWN_USER")).strip() or "UNKNOWN_USER"
            device_hash = str(row.get("device_hash", "")).strip()
            ip_address = str(row.get("ip_address", "")).strip()
            device_type = str(row.get("device_type", "unknown")).strip() or "unknown"
            city = str(row.get("city", "unknown")).strip() or "unknown"
            country = str(row.get("country", "unknown")).strip() or "unknown"
            first_seen = _parse_time(row.get("first_seen"), now)
            last_seen = _parse_time(row.get("last_seen"), first_seen)
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, 'active')",
                (user_id, user_id, None, now),
            )
            if device_hash:
                conn.execute(
                    "INSERT INTO devices(user_id, device_hash, device_type, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (user_id, device_hash, device_type, first_seen, last_seen),
                )
                inserted += 1
            if ip_address:
                conn.execute(
                    "INSERT INTO addresses(user_id, ip_address, city, country, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, ip_address, city, country, first_seen, last_seen),
                )
                inserted += 1

    elif source_type == "feedback":
        for _, row in df.iterrows():
            user_id = str(row.get("user_id", "")).strip() or None
            tx_value = row.get("transaction_id", row.get("source_transaction_id", None))
            reviewer = str(row.get("reviewer", uploaded_by or "importer")).strip() or "importer"
            decision = str(row.get("decision", "need_review")).strip() or "need_review"
            note = str(row.get("note", "imported feedback")).strip()
            created_at = _parse_time(row.get("created_at"), now)

            # Feedback imported from CSV can reference users/transactions that are
            # not already present in the demo SQLite database. Create a light
            # user row when needed, and keep the original CSV transaction code in
            # source_transaction_id instead of forcing it into the integer FK.
            _ensure_feedback_user(conn, user_id)
            transaction_id, source_transaction_id = _feedback_transaction_values(conn, tx_value)
            feedback_cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_feedback)").fetchall()}
            if "source_transaction_id" in feedback_cols:
                conn.execute(
                    """
                    INSERT INTO user_feedback(user_id, transaction_id, source_transaction_id, reviewer, decision, note, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, transaction_id, source_transaction_id, reviewer, decision, note, created_at),
                )
            else:
                extra = f" [source_transaction_id={source_transaction_id}]" if source_transaction_id and transaction_id is None else ""
                conn.execute(
                    "INSERT INTO user_feedback(user_id, transaction_id, reviewer, decision, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, transaction_id, reviewer, decision, f"{note}{extra}".strip(), created_at),
                )
            inserted += 1

    elif source_type == "transactions":
        for _, row in df.iterrows():
            user_id = str(row.get("UserID", row.get("user_id", "IMPORT_USER"))).strip() or "IMPORT_USER"
            source_tx = str(row.get("TransactionID", "")).strip() or None
            tx_time = _parse_time(row.get("TransactionDT"), now)
            amount = pd.to_numeric(pd.Series([row.get("TransactionAmt", 0)]), errors="coerce").fillna(0).iloc[0]
            product = str(row.get("ProductCD", "imported")).strip()
            device = str(row.get("DeviceInfo", row.get("DeviceType", "imported_device"))).strip()
            ip_address = str(row.get("ip_address", row.get("addr1", "0.0.0.0"))).strip()
            fraud_probability = float(row.get("Fraud_Probability", 0.0)) if "Fraud_Probability" in row.index else 0.0
            trust_score = float(row.get("Trust_Score", 100.0)) if "Trust_Score" in row.index else 100.0
            risk_level = str(row.get("Risk_Level", "Độ tin cậy cao"))
            raw_json = row.to_json(force_ascii=False)
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, 'active')",
                (user_id, user_id, None, now),
            )
            cur = conn.execute(
                """
                INSERT INTO transactions(source_transaction_id, user_id, transaction_time, amount, product_code, device_hash, ip_address, fraud_probability, trust_score, risk_level, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_tx, user_id, tx_time, float(amount), product, device, ip_address, fraud_probability, trust_score, risk_level, raw_json),
            )
            transaction_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO predictions(user_id, transaction_id, created_at, fraud_probability, trust_score, risk_level, explanation_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, transaction_id, now, fraud_probability, trust_score, risk_level, json.dumps({"source": "data_import"}, ensure_ascii=False)),
            )
            inserted += 1

    conn.commit()
    conn.close()
    storage_path = _save_uploaded_dataframe(storage_df if storage_df is not None else df, uploaded_by, file_name, source_type)
    source_id = record_data_source(
        source_name=source_name,
        source_type=source_type,
        file_name=file_name,
        row_count=len(df),
        valid_rows=inserted,
        invalid_rows=max(0, len(df) - inserted),
        uploaded_by=uploaded_by,
        status="imported" if inserted else "empty",
        validation_report=validation_report,
        db_path=db_path,
        storage_path=storage_path,
    )
    return source_id, inserted
