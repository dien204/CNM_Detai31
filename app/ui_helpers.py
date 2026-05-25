import json
import os
import sys
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.behavior import compute_user_behavior_scores, explain_model_prediction, read_audit_logs, user_timeline
from src.database import DB_PATH, authenticate_app_account, create_app_account, get_connection, init_db, reset_account_password, seed_demo_database
from src.inference import DEFAULT_MANUAL_FIELDS, ID_COL, TARGET, build_manual_sample, load_json_if_exists, score_transactions
from src.utils import load_model

REAL_MODEL_PATH = "models/trust_xgb_model.pkl"
REAL_FEATURE_COLUMNS_PATH = "models/feature_columns.json"
REAL_TEST_DATA_PATH = "data/processed/processed_test.csv"

DEMO_MODEL_PATH = "models/trust_xgb_demo_model.pkl"
DEMO_FEATURE_COLUMNS_PATH = "models/demo_feature_columns.json"
DEMO_TEST_DATA_PATH = "data/demo/demo_transactions.csv"

PREPROCESSING_METADATA_PATH = "data/processed/preprocessing_metadata.json"
LABEL_ENCODERS_PATH = "data/processed/label_encoders.json"
METRICS_PATHS = ["reports/evaluation_metrics.json", "models/training_metrics.json"]

PERMISSIONS = {
    "Admin": {
        "view_dashboard", "predict", "upload", "export", "analyze", "explain",
        "profile", "anomaly", "monitoring", "feedback", "view_audit", "review",
        "manage", "reset_db"
    },
    "User": {"view_dashboard", "predict", "upload", "export", "analyze", "explain", "profile", "anomaly", "monitoring", "feedback"},
}

ROLE_DESCRIPTIONS = {
    "Admin": "Quản trị hệ thống, xem database, audit log, phân quyền, reset dữ liệu demo và theo dõi trạng thái vận hành.",
    "User": "Sử dụng các chức năng phân tích: xem dashboard, dự đoán giao dịch, upload CSV, batch scoring, export kết quả và phân tích hành vi dài hạn.",
}

ROLE_DISPLAY = {
    "Admin": "Admin",
    "User": "Người dùng",
}

PAGE_ACCESS = {
    "Tổng quan": {"Admin", "User"},
    "Dự đoán giao dịch": {"Admin", "User"},
    "Phân tích hành vi": {"Admin", "User"},
    "Hồ sơ người dùng": {"Admin", "User"},
    "Phát hiện bất thường": {"Admin", "User"},
    "Monitoring": {"Admin", "User"},
    "Batch scoring": {"Admin", "User"},
    "Nhập dữ liệu": {"Admin", "User"},
    "Giải thích mô hình": {"Admin", "User"},
    "Feedback": {"Admin", "User"},
    "Quản trị hệ thống": {"Admin"},
    "Audit log": {"Admin"},
}

PAGE_FILES = {}

def configure_page(title: str, icon: str = "🛡️"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")


def inject_css():
    """Load UI styles from an external CSS file.

    Keeping CSS outside the Python page makes the interface easier to
    maintain and reduces the amount of HTML/CSS mixed into business logic.
    """
    css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
    except FileNotFoundError:
        css = ""
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def metric_card(label: str, value: str, helper: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def hero(title: str, description: str, tag: Optional[str] = None):
    tag_html = f'<div class="tag">{tag}</div>' if tag else ''
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
            {tag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_theme(score: float) -> Dict[str, str]:
    if score >= 80:
        return {"class": "green", "color": "#34d399", "text": "Tin cậy cao"}
    if score >= 50:
        return {"class": "amber", "color": "#fbbf24", "text": "Cần theo dõi"}
    return {"class": "red", "color": "#fb7185", "text": "Rủi ro cao"}


@st.cache_resource(show_spinner=False)
def load_trust_model(path: str):
    return load_model(path)


@st.cache_data(show_spinner=False)
def load_json_cached(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv_cached(path: str):
    return pd.read_csv(path)


@st.cache_data(ttl=60, show_spinner=False)
def load_behavior_cached(db_path: str):
    return compute_user_behavior_scores(db_path)


@st.cache_data(ttl=15, show_spinner=False)
def load_audit_cached(db_path: str):
    return read_audit_logs(db_path)


@st.cache_data(ttl=60, show_spinner=False)
def load_timeline_cached(db_path: str, user_id: str):
    return user_timeline(db_path, user_id)


@st.cache_data(show_spinner=False)
def load_metrics(is_demo: bool):
    paths = ["models/demo_training_metrics.json"] if is_demo else METRICS_PATHS
    for path in paths:
        if os.path.exists(path):
            return load_json_if_exists(path, {})
    return {}


def select_assets():
    if os.path.exists(REAL_MODEL_PATH) and os.path.exists(REAL_TEST_DATA_PATH):
        return {
            "mode": "Real model",
            "model_path": REAL_MODEL_PATH,
            "feature_path": REAL_FEATURE_COLUMNS_PATH,
            "data_path": REAL_TEST_DATA_PATH,
            "is_demo": False,
        }
    return {
        "mode": "Demo model",
        "model_path": DEMO_MODEL_PATH,
        "feature_path": DEMO_FEATURE_COLUMNS_PATH,
        "data_path": DEMO_TEST_DATA_PATH,
        "is_demo": True,
    }


def ensure_demo_db():
    init_db(DB_PATH)
    conn = get_connection(DB_PATH)
    user_count = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    conn.close()
    if user_count == 0:
        seed_demo_database(DB_PATH, DEMO_TEST_DATA_PATH)


@st.cache_data(ttl=300, show_spinner=False)
def load_app_data():
    ensure_demo_db()
    assets = select_assets()
    feature_columns = load_json_cached(assets["feature_path"])
    reference_df = load_csv_cached(assets["data_path"])
    metrics = load_metrics(assets["is_demo"])
    behavior_df = load_behavior_cached(DB_PATH)
    return assets, feature_columns, reference_df, metrics, behavior_df


def has_permission(role: str, action: str) -> bool:
    return action in PERMISSIONS.get(role, set())


def page_allowed(page_name: str, role: str) -> bool:
    return role in PAGE_ACCESS.get(page_name, set())


def authenticate(login: str, password: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    return authenticate_app_account(login, password, DB_PATH)


def register_user_account(full_name: str, email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, str]]]:
    return create_app_account(full_name, email, password, role="User", db_path=DB_PATH)


def reset_user_password(login: str, new_password: str) -> Tuple[bool, str]:
    return reset_account_password(login, new_password, DB_PATH)


def logout():
    for key in ["logged_in", "username", "role", "display_name", "active_uploaded_df", "active_uploaded_report", "active_uploaded_file_name", "active_uploaded_source_page", "active_uploaded_scored_df", "assistant_open", "assistant_text_input"]:
        if key in st.session_state:
            del st.session_state[key]


def ensure_auth_state():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("role", "")
    st.session_state.setdefault("display_name", "")


def require_login(page_title: str = "") -> Dict[str, str]:
    ensure_auth_state()
    if not st.session_state.get("logged_in"):
        st.warning("Bạn cần đăng nhập để truy cập chức năng này.")
        st.info("Hãy mở trang chính `streamlit_app.py` để đăng nhập.")
        st.stop()
    if page_title and not page_allowed(page_title, st.session_state.get("role", "")):
        st.error("Vai trò hiện tại không có quyền truy cập trang này.")
        st.stop()
    return {
        "username": st.session_state.get("username", ""),
        "role": st.session_state.get("role", ""),
        "display_name": st.session_state.get("display_name", ""),
    }


def render_sidebar(page_title: str, role: str, username: str, assets: Dict, data_rows: int):
    st.sidebar.markdown("## User Trust Platform")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <b>Người dùng</b>: {username}<br>
            <b>Vai trò</b>: {role}<br>
            <b>Trang hiện tại</b>: {page_title}<br>
            <b>Mô hình</b>: {assets['mode']}<br>
            <b>Dữ liệu demo</b>: {data_rows:,} dòng<br>
            <b>Database</b>: SQLite local
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Trang được phép truy cập")
    for title, path in PAGE_FILES.items():
        if page_allowed(title, role):
            try:
                st.sidebar.page_link(path, label=title)
            except Exception:
                st.sidebar.markdown(f"- {title}")
    st.sidebar.write("")
    if st.sidebar.button("Đăng xuất"):
        logout()
        st.rerun()

    st.sidebar.markdown("### Quyền hiện tại")
    role_permissions = sorted(PERMISSIONS.get(role, set()))
    st.sidebar.markdown("\n".join([f"- `{p}`" for p in role_permissions]))

def read_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def manual_input_form(reference_df: pd.DataFrame, feature_columns):
    available = [f for f in DEFAULT_MANUAL_FIELDS if f in feature_columns]
    base = reference_df[feature_columns].median(numeric_only=True).to_dict()
    overrides = {}
    for field in available[:12]:
        default = float(base.get(field, 0.0))
        step = 10.0 if field == "TransactionAmt" else 1.0
        overrides[field] = st.number_input(field, value=max(default, 1.0) if field == "TransactionAmt" else default, step=step, format="%.4f")
    return build_manual_sample(reference_df, overrides, feature_columns)


def prepare_scoring_context(reference_df: pd.DataFrame, feature_columns):
    metadata = load_json_if_exists(PREPROCESSING_METADATA_PATH, {})
    label_encoders = load_json_if_exists(LABEL_ENCODERS_PATH, {})
    fill_values = metadata.get("fill_values", {}) if isinstance(metadata, dict) else {}
    return {"reference_df": reference_df, "feature_columns": feature_columns, "fill_values": fill_values, "label_encoders": label_encoders}


def _validate_scoring_context(context: Dict):
    if not isinstance(context, dict) or not context.get("feature_columns"):
        raise RuntimeError("Scoring context chưa được khởi tạo. Hãy tải lại trang hoặc đăng nhập lại.")
    return context


def score_single(model, sample_df: pd.DataFrame, context: Dict):
    context = _validate_scoring_context(context)
    return score_transactions(
        model,
        sample_df,
        context["feature_columns"],
        reference_df=context["reference_df"],
        fill_values=context["fill_values"],
        label_encoders=context["label_encoders"],
    )


def score_batch(model, input_df: pd.DataFrame, context: Dict):
    context = _validate_scoring_context(context)
    return score_transactions(
        model,
        input_df,
        context["feature_columns"],
        reference_df=context["reference_df"],
        fill_values=context["fill_values"],
        label_encoders=context["label_encoders"],
    )


def render_prediction_panel(scored_row: pd.DataFrame, title: str = "Kết quả giao dịch"):
    p = float(scored_row["Fraud_Probability"].iloc[0])
    score = float(scored_row["Trust_Score"].iloc[0])
    risk = str(scored_row["Risk_Level"].iloc[0])
    theme = risk_theme(score)
    section_title(title)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-label">Trust Score</div>
                <div class="trust-score" style="color:{theme['color']};">{score:.2f}/100</div>
                <span class="pill {theme['class']}">{risk}</span>
                <div class="track" style="margin-top:1.05rem;"><div class="fill" style="width:{score}%; background:{theme['color']};"></div></div>
                <p class="muted" style="margin-top:.85rem;">Điểm càng cao thì rủi ro càng thấp và mức độ tin cậy càng tốt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Fraud Probability", f"{p:.4f}", "Xác suất gian lận")
        with c2:
            metric_card("Decision", theme["text"], "Khuyến nghị xử lý")
        st.markdown(
            f"""
            <div class="panel" style="margin-top:1rem;">
                <div class="metric-label">Fraud Risk Bar</div>
                <div class="track" style="margin-top:.8rem;"><div class="fill" style="width:{min(100, p*100)}%; background:#fb7185;"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def model_explanation(model, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return explain_model_prediction(model, X, top_n=top_n)


def db_overview() -> Dict[str, int]:
    ensure_demo_db()
    conn = get_connection(DB_PATH)
    tables = [
        "app_accounts", "users", "devices", "addresses", "login_events",
        "transactions", "predictions", "audit_logs", "user_feedback",
        "model_registry", "data_sources",
    ]
    result = {}
    for table in tables:
        try:
            result[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
        except Exception:
            result[table] = 0
    conn.close()
    return result
