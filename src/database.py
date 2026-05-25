import hashlib
import json
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DB_PATH = os.environ.get("TRUST_DB_PATH", "data/app/user_trust.db")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_sql(conn: sqlite3.Connection, table_name: str) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return row["sql"] if row and row["sql"] else ""


def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    # Small migration from old 4-role demo table to current 2-role design.
    app_account_sql = _table_sql(conn, "app_accounts")
    if app_account_sql and "'User'" not in app_account_sql:
        cur.execute("DROP TABLE IF EXISTS app_accounts")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_accounts (
            account_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT,
            role TEXT NOT NULL CHECK(role IN ('Admin','User')),
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    # Migrate existing demo databases created by older versions.
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(app_accounts)").fetchall()}
    if "email" not in existing_cols:
        cur.execute("ALTER TABLE app_accounts ADD COLUMN email TEXT")
    if "password_hash" not in existing_cols:
        cur.execute("ALTER TABLE app_accounts ADD COLUMN password_hash TEXT")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
            device_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            device_hash TEXT NOT NULL,
            device_type TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS addresses (
            address_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            city TEXT,
            country TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS login_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            login_time TEXT NOT NULL,
            device_hash TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            success INTEGER NOT NULL,
            risk_hint TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_transaction_id TEXT,
            user_id TEXT NOT NULL,
            transaction_time TEXT NOT NULL,
            amount REAL NOT NULL,
            product_code TEXT,
            device_hash TEXT,
            ip_address TEXT,
            fraud_probability REAL,
            trust_score REAL,
            risk_level TEXT,
            raw_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            transaction_id INTEGER,
            created_at TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            trust_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            explanation_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            created_at TEXT NOT NULL,
            detail TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            transaction_id INTEGER,
            reviewer TEXT NOT NULL,
            decision TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
        )
        """
    )

    # Migration: feedback created by uploaded CSV may reference user IDs or source transaction IDs
    # that are not yet present in the demo database tables. Keep the real source transaction
    # code separately and only use transaction_id for rows that exist in transactions.
    existing_feedback_cols = {row["name"] for row in cur.execute("PRAGMA table_info(user_feedback)").fetchall()}
    if "source_transaction_id" not in existing_feedback_cols:
        cur.execute("ALTER TABLE user_feedback ADD COLUMN source_transaction_id TEXT")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            feature_count INTEGER,
            dataset_version TEXT,
            metric_json TEXT,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """
    )


    cur.execute(
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
            validation_report TEXT
        )
        """
    )

    conn.commit()
    conn.close()



def hash_password(password: str) -> str:
    """Hash password for local demo authentication.

    This is intentionally lightweight for a local university demo. In a real
    production system, use bcrypt/argon2 with per-user salt.
    """
    return hashlib.sha256(("user-trust-local-salt::" + (password or "")).encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_registration_input(full_name: str, email: str, password: str) -> tuple[bool, str]:
    full_name = (full_name or "").strip()
    email = normalize_email(email)
    if len(full_name) < 2:
        return False, "Tên phải có ít nhất 2 ký tự."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Email chưa đúng định dạng."
    if len(password or "") < 6:
        return False, "Mật khẩu phải có ít nhất 6 ký tự."
    return True, "OK"


def create_app_account(
    full_name: str,
    email: str,
    password: str,
    role: str = "User",
    db_path: str = DB_PATH,
) -> tuple[bool, str, dict | None]:
    """Create a login account for Streamlit app.

    Registered accounts are normal users. Admin is seeded by the demo database.
    """
    init_db(db_path)
    valid, message = validate_registration_input(full_name, email, password)
    if not valid:
        return False, message, None
    role = role if role in {"Admin", "User"} else "User"
    email = normalize_email(email)
    conn = get_connection(db_path)
    existing = conn.execute(
        "SELECT account_id FROM app_accounts WHERE lower(username)=lower(?) OR lower(email)=lower(?)",
        (email, email),
    ).fetchone()
    if existing:
        conn.close()
        return False, "Email này đã được đăng ký.", None
    created_at = now_iso()
    conn.execute(
        """
        INSERT INTO app_accounts(username, email, password_hash, role, display_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (email, email, hash_password(password), role, full_name.strip(), created_at),
    )

    # Also create a lightweight user profile row so Admin can see registered users.
    user_id = "APP_" + hashlib.sha1(email.encode("utf-8")).hexdigest()[:10].upper()
    conn.execute(
        """
        INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        (user_id, full_name.strip(), email, created_at),
    )
    conn.commit()
    account = {"username": email, "email": email, "role": role, "display_name": full_name.strip()}
    conn.close()
    return True, "Đăng ký thành công.", account


def authenticate_app_account(login: str, password: str, db_path: str = DB_PATH) -> tuple[bool, dict | None]:
    """Authenticate by username/email and return account role.

    Admin is seeded as username `admin`, password `Admin@123`. Registered users
    log in with their email and password. The function also repairs older demo
    databases where the admin account existed but had no password_hash column
    or a NULL password_hash value.
    """
    init_db(db_path)
    login_value = (login or "").strip().lower()
    if not login_value or not password:
        return False, None

    conn = get_connection(db_path)

    # Ensure default admin account always exists and always has a valid password.
    admin_hash = hash_password("Admin@123")
    conn.execute(
        """
        INSERT OR IGNORE INTO app_accounts(username, email, password_hash, role, display_name, created_at)
        VALUES ('admin', 'admin@local', ?, 'Admin', 'Quản trị viên', ?)
        """,
        (admin_hash, now_iso()),
    )
    conn.execute(
        """
        UPDATE app_accounts
        SET email = COALESCE(email, 'admin@local'), password_hash = ?
        WHERE lower(username) = 'admin'
        """,
        (admin_hash,),
    )
    conn.commit()

    row = conn.execute(
        """
        SELECT username, email, password_hash, role, display_name
        FROM app_accounts
        WHERE lower(username)=?
           OR lower(COALESCE(email, ''))=?
           OR lower(COALESCE(display_name, ''))=?
        ORDER BY CASE
            WHEN lower(username)=? THEN 0
            WHEN lower(COALESCE(email, ''))=? THEN 1
            ELSE 2
        END
        LIMIT 1
        """,
        (login_value, login_value, login_value, login_value, login_value),
    ).fetchone()
    conn.close()
    if not row or not row["password_hash"]:
        return False, None
    if row["password_hash"] != hash_password(password):
        return False, None
    return True, {
        "username": row["username"],
        "email": row["email"] or row["username"],
        "role": row["role"],
        "display_name": row["display_name"],
    }


def reset_account_password(login: str, new_password: str, db_path: str = DB_PATH) -> tuple[bool, str]:
    """Reset password for a local demo account by username or email.

    This is a local university-demo flow, so it does not send email OTP.
    It still validates that the account exists and stores only a password hash.
    """
    init_db(db_path)
    login_value = (login or "").strip().lower()
    if not login_value:
        return False, "Vui lòng nhập email hoặc tài khoản."
    if len(new_password or "") < 6:
        return False, "Mật khẩu mới phải có ít nhất 6 ký tự."

    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT account_id, username, email, role, display_name
        FROM app_accounts
        WHERE lower(username)=?
           OR lower(COALESCE(email, ''))=?
           OR lower(COALESCE(display_name, ''))=?
        ORDER BY CASE
            WHEN lower(username)=? THEN 0
            WHEN lower(COALESCE(email, ''))=? THEN 1
            ELSE 2
        END
        LIMIT 1
        """,
        (login_value, login_value, login_value, login_value, login_value),
    ).fetchone()
    if not row:
        conn.close()
        return False, "Không tìm thấy tài khoản hoặc email này."

    conn.execute(
        "UPDATE app_accounts SET password_hash=? WHERE account_id=?",
        (hash_password(new_password), int(row["account_id"])),
    )

    # Make sure registered normal accounts also have a user profile row.
    if row["role"] == "User":
        email = normalize_email(row["email"] or row["username"])
        user_id = "APP_" + hashlib.sha1(email.encode("utf-8")).hexdigest()[:10].upper()
        conn.execute(
            """
            INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (user_id, row["display_name"] or email, email, now_iso()),
        )

    conn.commit()
    conn.close()
    return True, "Đã đặt lại mật khẩu. Bạn có thể đăng nhập bằng mật khẩu mới."


def change_account_password(login: str, old_password: str, new_password: str, db_path: str = DB_PATH) -> tuple[bool, str]:
    """Change password for an authenticated account."""
    ok, _ = authenticate_app_account(login, old_password, db_path)
    if not ok:
        return False, "Mật khẩu hiện tại chưa đúng."
    return reset_account_password(login, new_password, db_path)


def ensure_app_account_profiles(db_path: str = DB_PATH) -> None:
    """Create missing user profile rows for registered User accounts."""
    init_db(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT username, email, display_name FROM app_accounts WHERE role='User'"
    ).fetchall()
    for row in rows:
        email = normalize_email(row["email"] or row["username"])
        if not email:
            continue
        user_id = "APP_" + hashlib.sha1(email.encode("utf-8")).hexdigest()[:10].upper()
        conn.execute(
            """
            INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (user_id, row["display_name"] or email, email, now_iso()),
        )
    conn.commit()
    conn.close()


def table_count(conn: sqlite3.Connection, table_name: str) -> int:
    cur = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}")
    return int(cur.fetchone()["n"])


def log_audit(
    actor: str,
    role: str,
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    detail: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO audit_logs(actor, role, action, entity_type, entity_id, created_at, detail)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (actor or "unknown", role or "unknown", action, entity_type, entity_id, now_iso(), detail),
    )
    conn.commit()
    conn.close()


def read_table(table_name: str, db_path: str = DB_PATH, limit: Optional[int] = None) -> pd.DataFrame:
    init_db(db_path)
    allowed = {
        "app_accounts", "users", "devices", "addresses", "login_events", "transactions",
        "predictions", "audit_logs", "user_feedback", "model_registry", "data_sources"
    }
    if table_name not in allowed:
        raise ValueError(f"Unsupported table: {table_name}")
    conn = get_connection(db_path)
    sql = f"SELECT * FROM {table_name}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def _feedback_transaction_values(conn: sqlite3.Connection, transaction_id: Optional[object]) -> tuple[Optional[int], Optional[str]]:
    """Return (transaction_pk, source_transaction_id) for feedback inserts.

    Uploaded CSV files often contain TransactionID values such as TX001 or ORDER_123.
    The demo database table user_feedback.transaction_id is an INTEGER foreign key to
    transactions.transaction_id, so inserting a raw CSV ID there causes FOREIGN KEY
    errors. We store the original value in source_transaction_id and only set the
    integer FK when it really exists in the transactions table.
    """
    if transaction_id is None or str(transaction_id).strip() == "":
        return None, None
    source_tx = str(transaction_id).strip()
    try:
        tx_pk = int(float(source_tx))
    except (TypeError, ValueError):
        return None, source_tx
    row = conn.execute("SELECT transaction_id FROM transactions WHERE transaction_id = ?", (tx_pk,)).fetchone()
    return (tx_pk if row else None), source_tx


def _ensure_feedback_user(conn: sqlite3.Connection, user_id: Optional[str]) -> None:
    """Create a lightweight placeholder user for CSV-upload feedback when needed."""
    if not user_id:
        return
    exists = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (str(user_id),)).fetchone()
    if exists:
        return
    conn.execute(
        "INSERT INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, ?)",
        (str(user_id), f"CSV user {user_id}", None, now_iso(), "active"),
    )


def add_feedback(
    user_id: str,
    transaction_id: Optional[object],
    reviewer: str,
    decision: str,
    note: str,
    db_path: str = DB_PATH,
) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        _ensure_feedback_user(conn, user_id)
        tx_pk, source_tx = _feedback_transaction_values(conn, transaction_id)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(user_feedback)").fetchall()}
        if "source_transaction_id" in cols:
            conn.execute(
                """
                INSERT INTO user_feedback(user_id, transaction_id, source_transaction_id, reviewer, decision, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, tx_pk, source_tx, reviewer, decision, note, now_iso()),
            )
        else:
            # Backward-compatible fallback for very old databases.
            extra = f" [source_transaction_id={source_tx}]" if source_tx and tx_pk is None else ""
            conn.execute(
                """
                INSERT INTO user_feedback(user_id, transaction_id, reviewer, decision, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, tx_pk, reviewer, decision, f"{note or ''}{extra}".strip(), now_iso()),
            )
        conn.commit()
    finally:
        conn.close()


def _risk_level_from_score(score: float) -> str:
    if score >= 80:
        return "Độ tin cậy cao"
    if score >= 50:
        return "Độ tin cậy trung bình"
    return "Độ tin cậy thấp"


def _repeat_to_size(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df) >= n_rows:
        return df.head(n_rows).copy()
    reps = int(np.ceil(n_rows / len(df)))
    out = pd.concat([df] * reps, ignore_index=True).head(n_rows).copy()
    if "TransactionID" in out.columns:
        out["TransactionID"] = np.arange(3300000, 3300000 + len(out))
    return out


def seed_demo_database(
    db_path: str = DB_PATH,
    demo_csv_path: str = "data/demo/demo_transactions.csv",
    reset: bool = False,
    n_users: int = 100,
    n_transactions: int = 5000,
) -> None:
    """Create a larger local demo database for behavior analytics.

    Public fraud datasets usually hide stable UserID, IP, login, audit and review
    history. This seed creates synthetic long-term behavior linked to transaction
    samples so the app can demonstrate user behavior analytics end-to-end.
    """
    random.seed(42)
    np.random.seed(42)
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    if reset:
        for table in [
            "user_feedback", "audit_logs", "predictions", "transactions", "login_events",
            "addresses", "devices", "users", "model_registry", "data_sources"
        ]:
            cur.execute(f"DELETE FROM {table}")
        conn.commit()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    accounts = [
        ("admin", "admin@local", "Admin", "Quản trị viên", "Admin@123"),
    ]
    for username, email, role, display_name, password in accounts:
        cur.execute(
            """
            INSERT OR IGNORE INTO app_accounts(username, email, password_hash, role, display_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, email, hash_password(password), role, display_name, now.isoformat(timespec="seconds")),
        )
        cur.execute(
            """
            UPDATE app_accounts
            SET email = ?, password_hash = ?, role = ?, display_name = ?
            WHERE lower(username) = lower(?)
            """,
            (email, hash_password(password), role, display_name, username),
        )
    conn.commit()

    # Recreate profile rows for registered user accounts after a demo reset.
    registered_accounts = conn.execute(
        "SELECT username, email, display_name FROM app_accounts WHERE role='User'"
    ).fetchall()
    for account in registered_accounts:
        account_email = normalize_email(account["email"] or account["username"])
        if not account_email:
            continue
        app_user_id = "APP_" + hashlib.sha1(account_email.encode("utf-8")).hexdigest()[:10].upper()
        cur.execute(
            """
            INSERT OR IGNORE INTO users(user_id, full_name, email, created_at, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (app_user_id, account["display_name"] or account_email, account_email, now.isoformat(timespec="seconds")),
        )
    conn.commit()

    # If the database already has behavior users and this is not an explicit reset,
    # leave all user data untouched. Login accounts remain persistent.
    if table_count(conn, "users") > len(registered_accounts) and not reset:
        conn.close()
        return

    metric_path = "models/demo_training_metrics.json"
    metrics = {}
    if os.path.exists(metric_path):
        try:
            metrics = json.load(open(metric_path, "r", encoding="utf-8"))
        except Exception:
            metrics = {}
    cur.execute(
        """
        INSERT INTO model_registry(model_name, model_version, artifact_path, feature_count, dataset_version, metric_json, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            "XGBoost User Trust Demo",
            "v1.0-demo",
            "models/trust_xgb_demo_model.pkl",
            420,
            f"demo-users-{n_users}-tx-{n_transactions}",
            json.dumps(metrics, ensure_ascii=False),
            now.isoformat(timespec="seconds"),
        ),
    )

    cities = ["HCM", "Ha Noi", "Da Nang", "Can Tho", "Hue", "Nha Trang", "Bien Hoa", "Vung Tau"]
    user_ids = [f"U{idx:04d}" for idx in range(1, n_users + 1)]
    suspicious_users = set(user_ids[2::13]) | set(user_ids[6::17])
    if not suspicious_users:
        suspicious_users = set(user_ids[: max(1, n_users // 10)])

    for idx, user_id in enumerate(user_ids, start=1):
        created = now - timedelta(days=random.randint(75, 720))
        status = "watchlist" if user_id in suspicious_users and random.random() < 0.35 else "active"
        cur.execute(
            "INSERT INTO users(user_id, full_name, email, created_at, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, f"Demo User {idx:03d}", f"user{idx:03d}@example.com", created.isoformat(timespec="seconds"), status),
        )

        device_count = random.randint(1, 3) + (random.randint(1, 4) if user_id in suspicious_users else 0)
        for d in range(device_count):
            first_seen = created + timedelta(days=random.randint(0, 80))
            last_seen = now - timedelta(days=random.randint(0, 45))
            if last_seen < first_seen:
                last_seen = first_seen + timedelta(days=random.randint(1, 30))
            cur.execute(
                """
                INSERT INTO devices(user_id, device_hash, device_type, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, f"dev-{idx:03d}-{d:02d}", random.choice(["desktop", "mobile", "tablet"]), first_seen.isoformat(timespec="seconds"), last_seen.isoformat(timespec="seconds")),
            )

        address_count = random.randint(1, 2) + (random.randint(1, 4) if user_id in suspicious_users else 0)
        for a in range(address_count):
            first_seen = created + timedelta(days=random.randint(0, 90))
            last_seen = now - timedelta(days=random.randint(0, 55))
            if last_seen < first_seen:
                last_seen = first_seen + timedelta(days=random.randint(1, 30))
            ip = f"10.{idx % 255}.{a}.{random.randint(2, 250)}"
            cur.execute(
                """
                INSERT INTO addresses(user_id, ip_address, city, country, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, ip, random.choice(cities), "VN", first_seen.isoformat(timespec="seconds"), last_seen.isoformat(timespec="seconds")),
            )

    device_map = pd.read_sql_query("SELECT user_id, device_hash FROM devices", conn).groupby("user_id")["device_hash"].apply(list).to_dict()
    ip_map = pd.read_sql_query("SELECT user_id, ip_address FROM addresses", conn).groupby("user_id")["ip_address"].apply(list).to_dict()

    # Long-term login events.
    for user_id in user_ids:
        base_count = random.randint(20, 90)
        if user_id in suspicious_users:
            base_count += random.randint(45, 120)
        for _ in range(base_count):
            days_ago = random.randint(0, 90)
            hour = random.choice(range(7, 23))
            if user_id in suspicious_users and random.random() < 0.36:
                hour = random.choice([0, 1, 2, 3, 23])
            login_time = now - timedelta(days=days_ago, minutes=random.randint(0, 1439))
            login_time = login_time.replace(hour=hour)
            success_prob = 0.95 if user_id not in suspicious_users else 0.74
            success = 1 if random.random() < success_prob else 0
            risk_hint = None if success else random.choice(["failed_password", "otp_failed", "blocked_ip", "unknown_device"])
            cur.execute(
                """
                INSERT INTO login_events(user_id, login_time, device_hash, ip_address, success, risk_hint)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, login_time.isoformat(timespec="seconds"), random.choice(device_map[user_id]), random.choice(ip_map[user_id]), success, risk_hint),
            )

    if os.path.exists(demo_csv_path):
        tx_df = pd.read_csv(demo_csv_path)
        tx_df = _repeat_to_size(tx_df, n_transactions)
    else:
        tx_df = pd.DataFrame(
            {
                "TransactionID": np.arange(3300000, 3300000 + n_transactions),
                "TransactionAmt": np.random.lognormal(mean=4.3, sigma=0.8, size=n_transactions).round(2),
                "ProductCD": np.random.choice(["W", "C", "H", "R", "S"], size=n_transactions),
            }
        )

    amount_series = pd.to_numeric(tx_df.get("TransactionAmt", pd.Series(np.random.rand(len(tx_df)) * 400)), errors="coerce").fillna(50)
    p95_amount = float(np.percentile(amount_series, 95)) if len(amount_series) else 500.0

    for i, row in tx_df.iterrows():
        user_id = random.choice(list(suspicious_users)) if (i % 41 == 0 and suspicious_users) else random.choice(user_ids)
        days_ago = random.randint(0, 120)
        tx_time = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        amount = float(pd.to_numeric(row.get("TransactionAmt", random.uniform(5, 350)), errors="coerce"))
        if not np.isfinite(amount):
            amount = random.uniform(5, 350)
        product_code = str(row.get("ProductCD", "W"))
        source_tx = str(row.get("TransactionID", 3300000 + i))

        base_prob = 0.02 + min(amount / 10000.0, 0.12)
        if user_id in suspicious_users:
            base_prob += random.uniform(0.08, 0.28)
        if amount > p95_amount:
            base_prob += 0.12
        # Some bursty suspicious users.
        if user_id in suspicious_users and random.random() < 0.08:
            tx_time = now - timedelta(days=random.randint(0, 8), hours=random.randint(0, 2), minutes=random.randint(0, 59))
            base_prob += 0.15
        fraud_probability = float(max(0.001, min(0.98, base_prob + random.uniform(-0.02, 0.05))))
        trust_score = round((1 - fraud_probability) * 100, 2)
        risk_level = _risk_level_from_score(trust_score)
        raw_json = json.dumps({k: row[k] for k in tx_df.columns[:40] if k in row.index}, ensure_ascii=False, default=str)

        cur.execute(
            """
            INSERT INTO transactions(
                source_transaction_id, user_id, transaction_time, amount, product_code,
                device_hash, ip_address, fraud_probability, trust_score, risk_level, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source_tx, user_id, tx_time.isoformat(timespec="seconds"), amount, product_code, random.choice(device_map[user_id]), random.choice(ip_map[user_id]), fraud_probability, trust_score, risk_level, raw_json),
        )
        transaction_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO predictions(user_id, transaction_id, created_at, fraud_probability, trust_score, risk_level, explanation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, transaction_id, now.isoformat(timespec="seconds"), fraud_probability, trust_score, risk_level, json.dumps({"source": "demo_seed", "main_reason": "synthetic behavior profile"}, ensure_ascii=False)),
        )

    for user_id in list(suspicious_users)[:8]:
        tx_row = cur.execute("SELECT transaction_id FROM transactions WHERE user_id=? ORDER BY trust_score ASC LIMIT 1", (user_id,)).fetchone()
        tx_id = int(tx_row["transaction_id"]) if tx_row else None
        cur.execute(
            """
            INSERT INTO user_feedback(user_id, transaction_id, reviewer, decision, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, tx_id, "admin", random.choice(["need_review", "watchlist", "confirmed_risk"]), "Dữ liệu demo: user có nhiều tín hiệu hành vi bất thường", now.isoformat(timespec="seconds")),
        )

    audit_rows = [
        ("admin", "Admin", "INIT_DATABASE", "system", "db", f"Khởi tạo demo database: {n_users} users, {n_transactions} transactions"),
        ("user", "User", "BATCH_SCORE", "transactions", "demo", "Sinh điểm tin cậy cho dữ liệu demo"),
        ("admin", "Admin", "REVIEW_HIGH_RISK", "users", list(suspicious_users)[0] if suspicious_users else "U0001", "Đưa user demo vào danh sách kiểm tra"),
    ]
    for actor, role, action, entity_type, entity_id, detail in audit_rows:
        cur.execute(
            """
            INSERT INTO audit_logs(actor, role, action, entity_type, entity_id, created_at, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (actor, role, action, entity_type, entity_id, now.isoformat(timespec="seconds"), detail),
        )

    conn.commit()
    conn.close()
