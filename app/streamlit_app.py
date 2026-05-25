import os
import sys
from html import escape
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

os.environ.setdefault("ENABLE_SHAP", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app.api_client import backend_enabled, backend_health, batch_predict_dataframe_backend, predict_dataframe_backend
from app.ui_components import topbar_html
from app.ui_helpers import (
    PERMISSIONS,
    ROLE_DESCRIPTIONS,
    ROLE_DISPLAY,
    authenticate,
    configure_page,
    db_overview,
    ensure_auth_state,
    hero,
    inject_css,
    load_app_data,
    load_audit_cached,
    load_timeline_cached,
    load_trust_model,
    logout,
    manual_input_form,
    metric_card,
    model_explanation,
    prepare_scoring_context,
    render_prediction_panel,
    risk_theme,
    register_user_account,
    reset_user_password,
    score_batch,
    score_single,
    section_title,
)
from src.behavior import detect_anomalies, monitoring_summary, read_feedback, user_profile
from src.data_ingestion import IngestionResult, process_uploaded_csv
from src.data_source_manager import (
    SOURCE_LABELS,
    delete_data_source,
    import_source_dataframe,
    load_saved_source_dataframe,
    prepare_source_upload,
    read_data_sources,
)
from src.drift_monitoring import compute_drift_report, drift_from_database, drift_metrics
from src.explainability import shap_status
from src.trust_chat import answer_trust_question
from src.database import DB_PATH, add_feedback, log_audit, read_table, seed_demo_database
from src.inference import ID_COL, TARGET

configure_page("Identity trust, risk and behavior analytics", "🛡️")
inject_css()
ensure_auth_state()

# Load model/data only after login. Streamlit reruns the whole file when users
# switch pages; lazy cached loading keeps navigation lighter and avoids loading
# large objects on the login screen.
assets = None
feature_columns = None
reference_df = None
metrics = None
behavior_df = None
model = None
scoring_context = None


def ensure_runtime_loaded() -> None:
    global assets, feature_columns, reference_df, metrics, behavior_df, model, scoring_context
    if reference_df is not None and model is not None and scoring_context is not None:
        return
    assets, feature_columns, reference_df, metrics, behavior_df = load_app_data()
    model = load_trust_model(assets["model_path"])
    scoring_context = prepare_scoring_context(reference_df, feature_columns)


@st.cache_data(ttl=180, show_spinner=False)
def cached_monitoring_summary(db_path: str):
    return monitoring_summary(db_path)


@st.cache_data(ttl=180, show_spinner=False)
def cached_drift_report(db_path: str, reference_head: pd.DataFrame):
    return drift_from_database(reference_head, db_path, limit=5000)


def get_drift_report_cached() -> pd.DataFrame:
    return cached_drift_report(DB_PATH, reference_df.head(5000).copy())


def preload_after_login() -> None:
    """Warm the expensive cached objects once after login.

    Streamlit reruns this file on every interaction, so module-level globals
    such as model/scoring_context are reset. Even when the preload cache is
    already warm, we still restore those globals from cached loaders before
    returning. This prevents NoneType errors during prediction after page
    navigation or Render reruns.
    """
    if st.session_state.get("preload_ready"):
        ensure_runtime_loaded()
        return
    loading_slot = st.empty()
    loading_slot.markdown(
        '<div class="boot-loading-overlay"><div class="boot-spinner" aria-label="Loading"></div></div>',
        unsafe_allow_html=True,
    )
    try:
        ensure_runtime_loaded()
        _ = db_overview()
        _ = get_overview_sample_cached()
        _ = cached_monitoring_summary(DB_PATH)
        _ = load_audit_cached(DB_PATH)
        # Drift is useful but can be a little heavier, so keep the sampled report.
        _ = get_drift_report_cached()
        st.session_state["preload_ready"] = True
    finally:
        loading_slot.empty()


PAGES = {
    "Tổng quan": {"roles": {"Admin", "User"}, "desc": "Bức tranh tổng thể về độ tin cậy và rủi ro hành vi."},
    "Dự đoán và phân tích": {"roles": {"Admin", "User"}, "desc": "Chấm điểm Trust Score và giải thích mô hình cho từng giao dịch."},
    "Phân tích hành vi": {"roles": {"Admin", "User"}, "desc": "Phân tích đăng nhập, thiết bị, IP và giao dịch theo thời gian."},
    "Hồ sơ người dùng": {"roles": {"Admin", "User"}, "desc": "Xem chi tiết từng user: thiết bị, IP, login, giao dịch và feedback."},
    "Xem theo từng user": {"roles": {"Admin", "User"}, "desc": "Gộp hồ sơ người dùng và hành vi của user trong cùng một trang."},
    "Case study": {"roles": {"Admin", "User"}, "desc": "Kịch bản demo user bình thường, rủi ro cao và nghi ngờ chiếm tài khoản."},
    "Phát hiện bất thường": {"roles": {"Admin", "User"}, "desc": "Rule-based anomaly detection theo hành vi 30 ngày."},
    "Monitoring & Drift": {"roles": {"Admin", "User"}, "desc": "Theo dõi prediction, risk distribution, drift và model version."},
    "Batch scoring": {"roles": {"Admin", "User"}, "desc": "Xử lý hàng loạt file CSV và xuất kết quả."},
    "Nhập dữ liệu": {"roles": {"Admin", "User"}, "desc": "Import nhiều nguồn dữ liệu: transactions, login, users và devices/IP."},
    "Trợ lý dữ liệu": {"roles": {"Admin", "User"}, "desc": "Chat rule-based để hỏi nhanh về dataset, rủi ro và giới hạn hệ thống."},
    "Feedback": {"roles": {"Admin", "User"}, "desc": "Ghi nhận review/feedback để tạo vòng lặp cải thiện dữ liệu."},
    "Quản trị hệ thống": {"roles": {"Admin"}, "desc": "Quản lý dữ liệu nền, phân quyền và reset database demo."},
    "Audit log": {"roles": {"Admin"}, "desc": "Theo dõi sự kiện đăng nhập, dự đoán và thao tác quản trị."},
}



def allowed_pages(role: str) -> List[str]:
    return [name for name, meta in PAGES.items() if role in meta["roles"]]


def set_page(page_name: str):
    """Change page without touching URL query params.

    Raw HTML links such as ?page=... can make Streamlit create a fresh
    browser session on some setups, which looks like being kicked back to
    the login screen. Keeping navigation inside session_state avoids that.
    """
    # If the user leaves the Case study sub-flow through normal navigation,
    # remove the contextual back button so it does not appear unexpectedly later.
    if page_name not in {"Hồ sơ người dùng", "Xem theo từng user", "Feedback"}:
        st.session_state.pop("return_to_case_study", None)
        st.session_state.pop("case_force_db_user_view", None)
        st.session_state.pop("return_to_user360", None)
    if page_name not in {"Xem theo từng user", "Feedback"}:
        st.session_state.pop("return_to_user360", None)
    st.session_state["active_page"] = page_name


def role_name(role: str) -> str:
    return ROLE_DISPLAY.get(role, role)


def _handle_extra_page_change():
    selected = st.session_state.get("extra_page_selector")
    if not selected:
        return
    target_page = "Nhập dữ liệu" if selected == "Upload dữ liệu" else selected
    set_page(target_page)


def _nav_button_row(pages: List[str], active_page: str, variant: str) -> None:
    if not pages:
        return
    cols = st.columns(len(pages), gap="small")
    for col, page in zip(cols, pages):
        with col:
            label = ("● " if page == active_page else "") + page
            if st.button(label, key=f"nav_{variant}_{page}", use_container_width=True, disabled=(page == active_page)):
                set_page(page)
                st.rerun()


def top_nav(role: str, username: str):
    pages = allowed_pages(role)

    if st.session_state.get("active_page") == "Quản lý dữ liệu":
        st.session_state["active_page"] = "Nhập dữ liệu"
    if st.session_state.get("active_page") in {"Dự đoán giao dịch", "Giải thích mô hình"}:
        st.session_state["active_page"] = "Dự đoán và phân tích"
    if st.session_state.get("active_page") not in pages:
        st.session_state["active_page"] = "Tổng quan" if "Tổng quan" in pages else pages[0]

    active_page = st.session_state["active_page"]
    runtime = "FastAPI" if backend_enabled() else "Local"
    header_cols = st.columns([6.4, 1.4], gap="medium")
    with header_cols[0]:
        st.markdown(
            topbar_html(username=username, role_label=role_name(role), runtime_label=runtime, database_label="SQLite"),
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown('<div class="logout-spacer"></div><span id="logout-action-anchor"></span>', unsafe_allow_html=True)
        if st.button("Đăng xuất", key="logout_button", use_container_width=True):
            log_audit(username, role, "LOGOUT", "session", username, "Đăng xuất khỏi hệ thống", DB_PATH)
            logout()
            st.rerun()

    primary_pages = ["Tổng quan", "Nhập dữ liệu"]
    if role == "Admin":
        primary_pages.append("Quản trị hệ thống")
    primary_pages = [p for p in primary_pages if p in pages]

    nav_labels = {
        "Tổng quan": "Tổng quan",
        "Nhập dữ liệu": "Nhập dữ liệu",
        "Quản trị hệ thống": "Quản trị hệ thống",
    }
    cols = st.columns(len(primary_pages), gap="medium") if primary_pages else []
    for col, page in zip(cols, primary_pages):
        with col:
            label = ("● " if page == active_page else "") + nav_labels.get(page, page)
            if st.button(label, key=f"nav_primary_{page}", use_container_width=True, disabled=(page == active_page)):
                set_page(page)
                st.rerun()

    preferred_extra_order = [
        "Dự đoán và phân tích",
        "Xem theo từng user",
        "Case study",
        "Phát hiện bất thường",
        "Batch scoring",
        "Audit log",
        "Monitoring & Drift",
    ]
    extra_pages = [p for p in preferred_extra_order if p in pages]
    if extra_pages and active_page not in {"Tổng quan", "Quản trị hệ thống"}:
        # Keep the original selectbox-like function menu, but make the popover
        # tall enough to show all items at once. This avoids the duplicate
        # scrollbar issue while preserving the old compact UI.
        options = ["Upload dữ liệu"] + extra_pages
        current_label = "Upload dữ liệu" if active_page == "Nhập dữ liệu" else active_page
        # Some pages such as Feedback are opened from Case study and are not
        # shown as normal menu entries. Keep the current internal page in the
        # selectbox options for that run so the menu cannot immediately reset
        # active_page back to the previous/first option.
        if current_label not in options and active_page in pages and active_page not in primary_pages:
            options.append(current_label)
        options = list(dict.fromkeys(options))
        current_index = options.index(current_label) if current_label in options else 0
        selected = st.selectbox(
            "Chức năng phụ trợ",
            options,
            index=current_index,
            # Include active_page in the widget key so a programmatic navigation
            # from a page button, e.g. Case study -> Xem theo từng user/Feedback,
            # is not overwritten by the old selectbox value kept in session_state.
            key=f"function_menu_select_{role}_{username}_{active_page}",
            label_visibility="visible",
        )
        target_page = "Nhập dữ liệu" if selected == "Upload dữ liệu" else selected
        if target_page != active_page:
            set_page(target_page)
            st.rerun()

    return st.session_state["active_page"], role, username


def render_page_loading(page_name: str):
    st.markdown(
        """
        <div class="page-loading-card spinner-only" aria-label="Đang tải">
            <div class="mini-spinner"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workspace_card(title: str, desc: str, page_name: str, allowed: bool = True):
    st.markdown(
        f"""
        <div class="nav-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"Mở {title}", key=f"open_{page_name}", disabled=not allowed, use_container_width=True):
        set_page(page_name)
        st.rerun()


def render_ingestion_report(result: IngestionResult):
    report = result.report
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dòng dữ liệu", f"{report.get('row_count', 0):,}", "Số dòng đọc được")
    with c2:
        metric_card("Cột đã map", f"{report.get('mapped_columns', 0):,}", "Cột nhận diện được")
    with c3:
        metric_card("Feature tự điền", f"{report.get('missing_model_feature_count', 0):,}", "Thiếu so với model")
    with c4:
        metric_card("Trạng thái", "Hợp lệ" if result.valid else "Chưa hợp lệ", "Validation")

    st.markdown('<div class="mapping-report-spacer">&nbsp;</div>', unsafe_allow_html=True)
    with st.expander("Mapping cột và kiểm tra dữ liệu", expanded=not result.valid):
        st.dataframe(result.mapping_table, use_container_width=True, hide_index=True)
        if report.get("errors"):
            st.error("; ".join(report["errors"]))
        if report.get("warnings"):
            st.warning("; ".join(report["warnings"][:8]))
        if report.get("missing_model_features_sample"):
            st.caption("Một số feature model sẽ tự điền mặc định: " + ", ".join(report["missing_model_features_sample"][:12]))

def save_uploaded_dataset(ingestion: IngestionResult, file_name: str = "uploaded.csv", source_page: str = "CSV upload") -> None:
    """Persist the latest valid CSV in Streamlit session.

    Several pages (model explanation, behavior analysis, drift/batch views) should
    keep using the user's uploaded CSV until the user explicitly finishes that
    working session. This avoids jumping back to the demo dataset on rerun or
    when moving to another page.
    """
    st.session_state["active_uploaded_df"] = ingestion.dataframe.copy()
    st.session_state["active_uploaded_report"] = dict(ingestion.report)
    st.session_state["active_uploaded_file_name"] = file_name
    st.session_state["active_uploaded_source_page"] = source_page
    st.session_state["active_uploaded_version"] = st.session_state.get("active_uploaded_version", 0) + 1
    st.session_state.pop("active_uploaded_scored_df", None)
    st.session_state.pop("active_uploaded_scored_version", None)


def clear_uploaded_dataset() -> None:
    for key in [
        "active_uploaded_df",
        "active_uploaded_report",
        "active_uploaded_file_name",
        "active_uploaded_source_page",
        "active_uploaded_scored_df",
        "active_uploaded_version",
        "active_uploaded_scored_version",
    ]:
        st.session_state.pop(key, None)


def get_uploaded_dataset() -> pd.DataFrame:
    df = st.session_state.get("active_uploaded_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return pd.DataFrame()


def get_uploaded_report() -> Dict:
    report = st.session_state.get("active_uploaded_report", {})
    return dict(report) if isinstance(report, dict) else {}


def using_uploaded_dataset() -> bool:
    return not get_uploaded_dataset().empty


def render_active_dataset_notice() -> bool:
    uploaded_df = get_uploaded_dataset()
    if uploaded_df.empty:
        return False

    file_name = st.session_state.get("active_uploaded_file_name", "uploaded.csv")
    st.markdown(
        f"""
        <div class="active-data-card compact-active-data">
            <b>Dữ liệu đang dùng:</b> {file_name} <span>• {len(uploaded_df):,} dòng • được giữ trong phiên đăng nhập hiện tại</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True


def require_uploaded_dataset(page_label: str) -> pd.DataFrame:
    uploaded_df = get_uploaded_dataset()
    if uploaded_df.empty:
        st.info(f"Chưa có dữ liệu để xem ở mục {page_label}. Hãy chọn Upload dữ liệu trong ô chức năng để upload CSV mới hoặc chọn lại một file trong lịch sử upload.")
        return pd.DataFrame()
    render_active_dataset_notice()
    return uploaded_df

def get_scored_uploaded_dataset() -> pd.DataFrame:
    cached = st.session_state.get("active_uploaded_scored_df")
    uploaded_df = get_uploaded_dataset()
    version = st.session_state.get("active_uploaded_version", 0)
    cached_version = st.session_state.get("active_uploaded_scored_version")
    if uploaded_df.empty:
        return pd.DataFrame()
    if (
        isinstance(cached, pd.DataFrame)
        and not cached.empty
        and len(cached) == len(uploaded_df)
        and cached_version == version
    ):
        return cached.copy()
    scored = run_batch_prediction(uploaded_df)
    st.session_state["active_uploaded_scored_df"] = scored.copy()
    st.session_state["active_uploaded_scored_version"] = version
    return scored



def get_case_study_uploaded_dataset() -> pd.DataFrame:
    """Return uploaded data for Case study, preserving CSV Trust Score when available.

    The prediction flow can re-score uploaded CSVs with the model. For case-study
    storytelling, however, users often upload a prepared demo CSV that already
    contains transaction-level Trust_Score/Fraud_Probability. Preserve those
    values so the trust timeline reflects the uploaded file instead of becoming
    a flat model fallback output.
    """
    raw = get_uploaded_dataset()
    if raw.empty:
        return pd.DataFrame()

    if "Trust_Score" in raw.columns:
        df = raw.copy()
        df["Trust_Score"] = pd.to_numeric(df["Trust_Score"], errors="coerce")
        if df["Trust_Score"].isna().all():
            return get_scored_uploaded_dataset()
        df["Trust_Score"] = df["Trust_Score"].fillna(df["Trust_Score"].median()).clip(0, 100)
        if "Fraud_Probability" not in df.columns:
            df["Fraud_Probability"] = (1 - df["Trust_Score"] / 100).clip(0, 1)
        else:
            df["Fraud_Probability"] = pd.to_numeric(df["Fraud_Probability"], errors="coerce")
            df["Fraud_Probability"] = df["Fraud_Probability"].fillna((1 - df["Trust_Score"] / 100)).clip(0, 1)
        if "Risk_Level" not in df.columns:
            df["Risk_Level"] = df["Trust_Score"].apply(risk_label_from_score)
        else:
            df["Risk_Level"] = df["Risk_Level"].fillna(df["Trust_Score"].apply(risk_label_from_score))
        return df

    return get_scored_uploaded_dataset()


def sort_uploaded_user_timeline(user_rows: pd.DataFrame) -> pd.DataFrame:
    """Sort uploaded user rows by a stable time/order column before plotting."""
    if user_rows.empty:
        return user_rows.copy()
    timeline = user_rows.copy()
    for col in ["TransactionDate", "timestamp", "created_at", "event_time"]:
        if col in timeline.columns:
            parsed = pd.to_datetime(timeline[col], errors="coerce")
            if parsed.notna().any():
                timeline = timeline.assign(_timeline_order=parsed).sort_values("_timeline_order")
                return timeline.drop(columns=["_timeline_order"])
    for col in ["TransactionDT", ID_COL, "TransactionID"]:
        if col in timeline.columns:
            order = pd.to_numeric(timeline[col], errors="coerce")
            if order.notna().any():
                timeline = timeline.assign(_timeline_order=order).sort_values("_timeline_order")
                return timeline.drop(columns=["_timeline_order"])
    return timeline.reset_index(drop=False).sort_values("index").drop(columns=["index"])

def get_overview_sample_cached() -> pd.DataFrame:
    cached = st.session_state.get("overview_demo_scored_df")
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()
    scored = run_batch_prediction(reference_df.head(600).copy())
    st.session_state["overview_demo_scored_df"] = scored.copy()
    return scored


def summarize_uploaded_behavior(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Create a behavior-like summary from the uploaded CSV.

    Uploaded transaction CSVs usually do not contain login/device history, so this
    page shows transaction behavior available in the file instead of pretending to
    have 30-day database signals.
    """
    if scored_df.empty:
        return pd.DataFrame()
    df = scored_df.copy()
    if "UserID" not in df.columns:
        df["UserID"] = df[ID_COL].astype(str) if ID_COL in df.columns else [f"row_{i}" for i in range(len(df))]
    if "Trust_Score" not in df.columns:
        return pd.DataFrame()
    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = pd.cut(df["Trust_Score"], bins=[-1, 50, 80, 101], labels=["Low Trust", "Medium", "High Trust"])
    amount_col = "TransactionAmt" if "TransactionAmt" in df.columns else None
    agg_kwargs = {
        "transaction_count": (ID_COL if ID_COL in df.columns else "UserID", "count"),
        "avg_trust_score": ("Trust_Score", "mean"),
        "min_trust_score": ("Trust_Score", "min"),
        "avg_fraud_probability": ("Fraud_Probability", "mean") if "Fraud_Probability" in df.columns else ("Trust_Score", lambda s: 1 - (s.mean() / 100.0)),
        "high_risk_transactions": ("Trust_Score", lambda s: int((s < 50).sum())),
    }
    if amount_col:
        agg_kwargs["total_amount"] = (amount_col, "sum")
        agg_kwargs["avg_amount"] = (amount_col, "mean")
    summary = df.groupby("UserID", dropna=False).agg(**agg_kwargs).reset_index().rename(columns={"UserID": "user_id"})
    summary["risk_level"] = summary["avg_trust_score"].apply(lambda x: "Cao" if x < 50 else ("Trung bình" if x < 80 else "Thấp"))
    summary["explanation"] = summary.apply(
        lambda r: (
            f"Trong CSV upload, user này có {int(r['transaction_count'])} giao dịch; "
            f"Trust trung bình {float(r['avg_trust_score']):.2f}, thấp nhất {float(r['min_trust_score']):.2f}; "
            f"số giao dịch Trust < 50 là {int(r['high_risk_transactions'])}."
        ),
        axis=1,
    )
    return summary.sort_values(["avg_trust_score", "high_risk_transactions"], ascending=[True, False]).reset_index(drop=True)


def run_single_prediction(sample_df: pd.DataFrame):
    # Backend is optional. If it is unavailable, always ensure local model
    # context is restored before falling back to local inference.
    if backend_enabled():
        try:
            scored, explanation = predict_dataframe_backend(sample_df, top_n=12)
            if isinstance(scored, pd.DataFrame) and not scored.empty:
                return scored, explanation if isinstance(explanation, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            st.warning(f"Backend tạm thời không phản hồi, hệ thống chuyển sang dự đoán local. Chi tiết: {exc}")
    ensure_runtime_loaded()
    if model is None or scoring_context is None:
        raise RuntimeError("Model/scoring context chưa sẵn sàng. Vui lòng tải lại trang hoặc đăng nhập lại.")
    scored, X = score_single(model, sample_df, scoring_context)
    explanation = model_explanation(model, X, top_n=12)
    return scored, explanation


def run_batch_prediction(input_df: pd.DataFrame):
    if backend_enabled():
        try:
            scored = batch_predict_dataframe_backend(input_df)
            if isinstance(scored, pd.DataFrame) and not scored.empty:
                return scored
        except Exception as exc:
            st.warning(f"Backend tạm thời không phản hồi, hệ thống chuyển sang batch local. Chi tiết: {exc}")
    ensure_runtime_loaded()
    if model is None or scoring_context is None:
        raise RuntimeError("Model/scoring context chưa sẵn sàng. Vui lòng tải lại trang hoặc đăng nhập lại.")
    scored, _ = score_batch(model, input_df, scoring_context)
    return scored


def audit_once(username: str, role: str, action: str, entity_type: str | None = None, entity_id: str | None = None, detail: str | None = None) -> None:
    """Write an audit log once per browser session for the same event.

    Streamlit reruns the script whenever a widget changes, so logging directly in
    page rendering code can create duplicate audit rows. This helper keeps audit
    events automatic while preventing repeated rows for the same user/action/entity
    during the current session.
    """
    key = f"audit_once::{username}::{role}::{action}::{entity_type or ''}::{entity_id or ''}"
    if st.session_state.get(key):
        return
    log_audit(username, role, action, entity_type, entity_id, detail, DB_PATH)
    st.session_state[key] = True
    try:
        load_audit_cached.clear()
    except Exception:
        pass


def show_explanation_table(explanation_df: pd.DataFrame):
    if explanation_df.empty:
        st.info("Chưa có dữ liệu giải thích cho giao dịch này.")
        return
    display_df = explanation_df.copy()
    rename_map = {
        "feature": "Feature",
        "value": "Giá trị đầu vào",
        "model_importance": "Độ quan trọng mô hình",
        "explanation_score": "Điểm giải thích",
        "relative_impact_pct": "Tỷ lệ ảnh hưởng (%)",
        "shap_value": "SHAP value",
        "abs_shap_value": "|SHAP|",
        "direction": "Chiều tác động",
        "method": "Phương pháp",
        "method_note": "Ghi chú phương pháp",
        "interpretation": "Diễn giải",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def set_logged_in_user(user: Dict[str, str]) -> None:
    st.session_state["logged_in"] = True
    st.session_state["username"] = user["username"]
    st.session_state["role"] = user["role"]
    st.session_state["display_name"] = user["display_name"]
    st.session_state["active_page"] = "Tổng quan"
    st.session_state["preload_ready"] = False


def password_input(label: str, key: str, show: bool, placeholder: str = ""):
    """Password field with external show/hide toggle.

    When hidden, use Streamlit's native password type for stable masking.
    CSS in ui_helpers hides the built-in eye icon, so the only control is the
    explicit toggle above the field.
    """
    return st.text_input(
        label,
        type="default" if show else "password",
        placeholder=placeholder,
        key=key,
    )


def render_login():
    st.markdown('<div class="login-page-wrap">', unsafe_allow_html=True)
    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown(
            """
            <div class="login-shell compact-login-hero login-hero-fill">
                <div class="login-brand-block">
                    <div class="login-kicker">USER TRUST PLATFORM</div>
                    <h1>Identity trust, risk and behavior analytics</h1>
                    <p>Nền tảng đánh giá độ tin cậy người dùng, hỗ trợ nhập dữ liệu giao dịch, phân tích hành vi và theo dõi rủi ro trong một giao diện thống nhất.</p>
                    <div class="login-color-cards login-color-cards-compact">
                        <div class="color-card blue-card"><b>Trust Score</b><span>Chấm điểm tin cậy giao dịch</span></div>
                        <div class="color-card sky-card"><b>Upload dữ liệu</b><span>Nhập file CSV và dùng lại trong phiên đăng nhập</span></div>
                        <div class="color-card navy-card"><b>Phân quyền</b><span>Admin quản trị • User khai thác dữ liệu</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="login-form-title login-form-title-clean">Đăng nhập hệ thống</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Chọn tác vụ",
            ["Đăng nhập", "Đăng ký", "Quên mật khẩu"],
            horizontal=True,
            label_visibility="collapsed",
            key="login_mode_selector",
        )

        if mode == "Đăng nhập":
            login_name = st.text_input("Tên, email hoặc tài khoản", placeholder="Nhập tên, email hoặc tài khoản", key="login_name")
            show_password = st.toggle("Hiện mật khẩu", value=False, key="login_show_password")
            login_password = password_input(
                "Mật khẩu đăng nhập",
                key="login_password",
                show=show_password,
                placeholder="Nhập mật khẩu",
            )
            submitted = st.button("Đăng nhập", use_container_width=True, key="login_submit")
            if submitted:
                ok, user = authenticate(login_name, login_password)
                if ok and user:
                    set_logged_in_user(user)
                    log_audit(user["username"], user["role"], "LOGIN_SUCCESS", "session", user["username"], "Đăng nhập thành công", DB_PATH)
                    st.markdown('<div class="boot-loading-overlay login-transition-overlay"><div class="boot-spinner" aria-label="Loading"></div></div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    log_audit(login_name.strip() or "unknown", "Unknown", "LOGIN_FAILED", "session", login_name.strip(), "Sai tài khoản hoặc mật khẩu", DB_PATH)
                    st.error("Tài khoản hoặc mật khẩu chưa đúng.")

        elif mode == "Đăng ký":
            full_name = st.text_input("Tên", placeholder="Nhập họ tên", key="register_full_name")
            email = st.text_input("Email", placeholder="name@example.com", key="register_email")
            show_register_password = st.toggle("Hiện mật khẩu đăng ký", value=False, key="register_show_password")
            password = password_input(
                "Mật khẩu đăng ký",
                key="register_password",
                show=show_register_password,
                placeholder="Tối thiểu 6 ký tự",
            )
            confirm_password = password_input(
                "Nhập lại mật khẩu đăng ký",
                key="register_confirm_password",
                show=show_register_password,
                placeholder="Nhập lại mật khẩu",
            )
            register_submitted = st.button("Tạo tài khoản", use_container_width=True, key="register_submit")
            if register_submitted:
                if password != confirm_password:
                    st.error("Mật khẩu nhập lại chưa khớp.")
                else:
                    ok, message, user = register_user_account(full_name, email, password)
                    if ok and user:
                        set_logged_in_user(user)
                        log_audit(user["username"], user["role"], "REGISTER_SUCCESS", "account", user["username"], "Người dùng tự đăng ký tài khoản", DB_PATH)
                        st.markdown('<div class="boot-loading-overlay login-transition-overlay"><div class="boot-spinner" aria-label="Loading"></div></div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        log_audit(email.strip() or "unknown", "User", "REGISTER_FAILED", "account", email.strip(), message, DB_PATH)
                        st.error(message)

        else:
            reset_login = st.text_input("Tên, email hoặc tài khoản", placeholder="Nhập tên, email hoặc tài khoản", key="reset_login")
            show_reset_password = st.toggle("Hiện mật khẩu mới", value=False, key="reset_show_password")
            new_password = password_input(
                "Mật khẩu mới",
                key="reset_new_password",
                show=show_reset_password,
                placeholder="Tối thiểu 6 ký tự",
            )
            confirm_new_password = password_input(
                "Nhập lại mật khẩu mới",
                key="reset_confirm_password",
                show=show_reset_password,
                placeholder="Nhập lại mật khẩu mới",
            )
            reset_submitted = st.button("Cập nhật mật khẩu", use_container_width=True, key="reset_password_submit")
            if reset_submitted:
                if new_password != confirm_new_password:
                    st.error("Mật khẩu nhập lại chưa khớp.")
                else:
                    ok, message = reset_user_password(reset_login, new_password)
                    if ok:
                        log_audit(reset_login.strip() or "unknown", "Unknown", "PASSWORD_RESET", "account", reset_login.strip(), "Người dùng đặt lại mật khẩu tại màn hình đăng nhập", DB_PATH)
                        st.success(message)
                    else:
                        log_audit(reset_login.strip() or "unknown", "Unknown", "PASSWORD_RESET_FAILED", "account", reset_login.strip(), message, DB_PATH)
                        st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)


def render_score_like_panel(label: str, score: float, helper: str):
    score = max(0.0, min(100.0, float(score)))
    theme = risk_theme(score)
    st.markdown(
        f"""
        <div class="panel score-like-panel">
            <div class="metric-label">{label}</div>
            <div class="trust-score" style="color:{theme['color']};">{score:.2f}/100</div>
            <span class="pill {theme['class']}">{theme['text']}</span>
            <div class="track" style="margin-top:1.05rem;"><div class="fill" style="width:{score}%; background:{theme['color']};"></div></div>
            <p class="muted" style="margin-top:.85rem;">{helper}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(role: str, username: str):
    uploaded_df = get_uploaded_dataset()
    using_upload = not uploaded_df.empty

    if using_upload:
        tx_sample = get_scored_uploaded_dataset()
        source_name = st.session_state.get("active_uploaded_file_name", "CSV upload")
        source_label = "CSV upload"
        source_help = f"{len(tx_sample):,} dòng từ {source_name}"
        hero_text = (
            f"Trang chủ đang dùng dữ liệu bạn đã upload: {source_name}. "
            "Các chỉ số, biểu đồ và bảng ưu tiên bên dưới được tính từ CSV hiện tại."
        )
    else:
        tx_sample = get_overview_sample_cached()
        source_name = "Dữ liệu demo"
        source_label = "Demo dataset"
        source_help = f"{len(tx_sample):,} dòng ví dụ"
        hero_text = (
            "Trang này đang dùng dữ liệu ví dụ để minh họa dashboard. "
            "Sau khi upload CSV, Tổng quan sẽ tự chuyển sang thống kê theo dữ liệu upload."
        )

    st.markdown(
        f"""
        <div class="overview-hero-large">
            <div class="overview-kicker">TRANG CHỦ</div>
            <h2>User Trust Platform</h2>
            <p>{hero_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if using_upload:
        render_active_dataset_notice()

    total_tx = len(tx_sample)
    if tx_sample.empty or "Trust_Score" not in tx_sample.columns:
        st.warning("Chưa đủ dữ liệu đã chấm điểm để hiển thị tổng quan.")
        return

    high_trust = int((tx_sample["Trust_Score"] >= 80).sum())
    medium_trust = int(((tx_sample["Trust_Score"] >= 50) & (tx_sample["Trust_Score"] < 80)).sum())
    low_trust = int((tx_sample["Trust_Score"] < 50).sum())
    avg_trust = float(tx_sample["Trust_Score"].mean())

    if using_upload:
        upload_behavior = summarize_uploaded_behavior(tx_sample)
        if not upload_behavior.empty:
            behavior_score = float(upload_behavior["avg_trust_score"].mean())
            behavior_help = f"{upload_behavior['user_id'].nunique():,} user từ CSV"
        else:
            behavior_score = avg_trust
            behavior_help = source_help
        behavior_label = "Hành vi user upload"
        tx_label = "CSV đang phân tích"
        trust_help = "Từ dữ liệu upload"
        priority_help = "Low Trust trong CSV"
    else:
        if not behavior_df.empty and "long_term_trust_score" in behavior_df.columns:
            behavior_score = float(behavior_df["long_term_trust_score"].mean())
            behavior_help = f"{len(behavior_df):,} user demo"
        elif not behavior_df.empty and "behavior_risk_score" in behavior_df.columns:
            behavior_score = max(0.0, 100.0 - float(behavior_df["behavior_risk_score"].mean()))
            behavior_help = f"{len(behavior_df):,} user demo"
        else:
            behavior_score = avg_trust
            behavior_help = source_help
        behavior_label = "Hành vi user demo"
        tx_label = "Dữ liệu ví dụ"
        trust_help = "Ví dụ trang chủ"
        priority_help = "Low Trust cần xem"

    behavior_left, behavior_right = st.columns([1.12, 1], gap="large")
    with behavior_left:
        render_score_like_panel(behavior_label, behavior_score, f"Điểm hành vi trung bình trên {behavior_help}.")
    with behavior_right:
        score_cols = st.columns(2, gap="small")
        with score_cols[0]:
            metric_card("Transactions", f"{total_tx:,}", tx_label)
        with score_cols[1]:
            metric_card("Trust trung bình", f"{avg_trust:.2f}", trust_help)
        st.markdown('<div class="card-row-spacer">&nbsp;</div>', unsafe_allow_html=True)
        score_cols_2 = st.columns(2, gap="small")
        with score_cols_2[0]:
            metric_card("Phân bố Trust", f"{high_trust:,} / {medium_trust:,} / {low_trust:,}", "High / Medium / Low")
        with score_cols_2[1]:
            metric_card("User ưu tiên" if "UserID" in tx_sample.columns else "Giao dịch ưu tiên", f"{low_trust:,}", priority_help)

    chart_left, chart_right = st.columns([1, 1], gap="large")
    with chart_left:
        section_title("Phân bố kết quả")
        if "Risk_Level" in tx_sample.columns:
            risk_counts = tx_sample["Risk_Level"].value_counts().rename_axis("Mức rủi ro").to_frame("Số lượng")
            st.bar_chart(risk_counts, use_container_width=True)
        else:
            trust_bins = pd.cut(tx_sample["Trust_Score"], bins=[-1, 50, 80, 101], labels=["Low Trust", "Medium", "High Trust"])
            st.bar_chart(trust_bins.value_counts().rename_axis("Mức rủi ro").to_frame("Số lượng"), use_container_width=True)
    with chart_right:
        section_title("Trust Score theo dòng")
        line_df = tx_sample[["Trust_Score"]].head(120).reset_index(drop=True)
        line_df.index = line_df.index + 1
        st.line_chart(line_df, use_container_width=True)

    section_title("Bảng giao dịch user ưu tiên")
    priority_cols = [c for c in ["UserID", ID_COL, "TransactionAmt", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in tx_sample.columns]
    sort_cols = [c for c in ["Trust_Score", "Fraud_Probability"] if c in tx_sample.columns]
    if sort_cols == ["Trust_Score", "Fraud_Probability"]:
        priority_df = tx_sample.sort_values(sort_cols, ascending=[True, False]).head(12)
    elif sort_cols:
        priority_df = tx_sample.sort_values(sort_cols[0], ascending=True).head(12)
    else:
        priority_df = tx_sample.head(12)
    st.dataframe(priority_df[priority_cols] if priority_cols else priority_df, use_container_width=True, hide_index=True)

def render_prediction(role: str, username: str):
    hero(
        "Dự đoán và phân tích",
        "Chấm điểm Trust Score cho giao dịch và hiển thị phần giải thích mô hình/SHAP ngay bên dưới để tránh tách trùng chức năng.",
        "Transaction Scoring • Model Explainability",
    )
    active_uploaded = require_uploaded_dataset("Dự đoán và phân tích")
    if active_uploaded.empty:
        return

    idx = st.number_input(
        "Dòng cần xem chi tiết",
        min_value=0,
        max_value=max(len(active_uploaded) - 1, 0),
        value=0,
        step=1,
        key="prediction_analysis_row_index",
    )
    sample_df = active_uploaded.iloc[[idx]].copy()
    scored_row, explanation_df = run_single_prediction(sample_df)

    total_cols = st.columns(3, gap="small")
    with total_cols[0]:
        metric_card("Tổng dòng dataset", f"{len(active_uploaded):,}", "CSV đang phân tích")
    with total_cols[1]:
        metric_card("Dòng đang xem", f"{idx + 1:,}", "Chi tiết giao dịch")
    with total_cols[2]:
        metric_card("Nguồn dữ liệu", st.session_state.get("active_uploaded_file_name", "CSV upload"), "Phiên hiện tại")

    render_prediction_panel(scored_row)

    section_title("Thông tin giao dịch đầu vào")
    show_cols = [c for c in [ID_COL, "UserID", TARGET, "TransactionAmt", "ProductCD", "card4", "addr1", "addr2", "DeviceType", "DeviceInfo", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in scored_row.columns]
    st.dataframe(scored_row[show_cols], use_container_width=True, hide_index=True)

    section_title("Giải thích mô hình")
    status = shap_status()
    if status.get("available"):
        st.success(f"SHAP khả dụng trong môi trường hiện tại, version: {status.get('version')}.")
    else:
        st.info("SHAP chưa khả dụng trong môi trường hiện tại. App dùng giải thích dự phòng bằng feature importance.")
    show_explanation_table(explanation_df)

    tx_id = str(scored_row[ID_COL].iloc[0]) if ID_COL in scored_row.columns else f"row_{idx + 1}"
    detail = (
        f"Auto prediction audit; source={st.session_state.get('active_uploaded_file_name', 'CSV upload')}; "
        f"row={idx + 1}; Trust={float(scored_row['Trust_Score'].iloc[0]):.2f}; "
        f"FraudProb={float(scored_row['Fraud_Probability'].iloc[0]):.4f}"
    )
    audit_once(username, role, "PREDICT_TRANSACTION", "transaction", tx_id, detail)
    st.caption("Audit log dự đoán được hệ thống ghi tự động cho Admin theo dõi; User không cần bấm ghi thủ công.")

def render_behavior(role: str, username: str):
    hero("Phân tích hành vi", "Phân tích hành vi giao dịch từ file CSV đang được chọn; nếu chưa upload, hệ thống dùng dữ liệu demo để không hiển thị số liệu rỗng.", "Behavior Analytics")
    active_uploaded = get_uploaded_dataset()

    if not active_uploaded.empty:
        render_active_dataset_notice()
        scored_upload = get_scored_uploaded_dataset()
        dataset_label = "CSV upload"
    else:
        scored_upload = get_overview_sample_cached().copy()
        dataset_label = "Demo dataset"
        st.info("Chưa có CSV upload. Trang đang dùng dữ liệu demo để minh họa phân tích hành vi.")

    total_dataset_rows = len(scored_upload)
    upload_behavior = summarize_uploaded_behavior(scored_upload)
    if upload_behavior.empty:
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            metric_card("Tổng dòng dataset", f"{total_dataset_rows:,}", dataset_label)
        with c1:
            metric_card("User", f"{behavior_df['user_id'].nunique() if not behavior_df.empty and 'user_id' in behavior_df.columns else 0:,}", "Demo behavior")
        with c2:
            metric_card("Avg behavior", f"{behavior_df['long_term_trust_score'].mean():.2f}" if not behavior_df.empty and 'long_term_trust_score' in behavior_df.columns else "0.00", "Trên 100")
        with c3:
            metric_card("User rủi ro", f"{int((behavior_df['long_term_trust_score'] < 50).sum()) if not behavior_df.empty and 'long_term_trust_score' in behavior_df.columns else 0:,}", "Trust < 50")
        if not behavior_df.empty:
            section_title("Bảng hành vi user demo")
            display_cols = [c for c in ["user_id", "full_name", "long_term_trust_score", "behavior_risk_score", "failed_login_rate_30d", "device_change_count_30d", "address_change_count_30d", "explanation"] if c in behavior_df.columns]
            st.dataframe(behavior_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("Dataset hiện tại chưa đủ dữ liệu để phân tích hành vi giao dịch.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Tổng dòng dataset", f"{total_dataset_rows:,}", dataset_label)
    with c2:
        metric_card("User trong dataset", f"{upload_behavior['user_id'].nunique():,}", "Theo UserID nếu có")
    with c3:
        metric_card("Avg Trust", f"{scored_upload['Trust_Score'].mean():.2f}", "Trung bình file")
    with c4:
        metric_card("Trust < 50", f"{int((scored_upload['Trust_Score'] < 50).sum()):,}", "Giao dịch rủi ro")
    with c5:
        metric_card("Hành vi user", f"{upload_behavior['avg_trust_score'].mean():.2f}/100", "Trung bình user")

    section_title("Tổng hợp hành vi giao dịch")
    st.dataframe(upload_behavior, use_container_width=True, hide_index=True)

    selected_user = st.selectbox("Chọn user trong dataset", upload_behavior["user_id"].astype(str).tolist(), key="uploaded_behavior_user")
    user_rows = scored_upload.copy()
    if "UserID" in user_rows.columns:
        user_rows = user_rows[user_rows["UserID"].astype(str) == str(selected_user)]

    section_title("Giao dịch của user được chọn")
    show_cols = [c for c in [ID_COL, "UserID", "TransactionDT", "TransactionAmt", "ProductCD", "DeviceType", "DeviceInfo", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in user_rows.columns]
    st.dataframe(user_rows[show_cols].sort_values("Trust_Score") if "Trust_Score" in user_rows.columns else user_rows[show_cols], use_container_width=True, hide_index=True)

    section_title("Phân bố rủi ro")
    if "Risk_Level" in scored_upload.columns:
        risk_counts = scored_upload["Risk_Level"].value_counts().rename_axis("Risk_Level").to_frame("count")
        st.bar_chart(risk_counts, use_container_width=True)

def render_user_by_user(role: str, username: str):
    hero("Xem theo từng user", "Xem hồ sơ người dùng ở phía trên, sau đó xem hành vi và giao dịch của chính người dùng đó ở phía dưới.", "User 360 View")

    active_uploaded = get_uploaded_dataset()
    force_db_view = bool(st.session_state.get("case_force_db_user_view"))
    if not active_uploaded.empty and not force_db_view:
        scored_upload = get_scored_uploaded_dataset()
        if scored_upload.empty:
            st.warning("CSV upload hiện chưa có dữ liệu để xem theo user.")
            return
        working = scored_upload.copy()
        if "UserID" not in working.columns:
            working["UserID"] = [f"ROW_{i+1:04d}" for i in range(len(working))]
        user_options = working["UserID"].astype(str).dropna().unique().tolist()
        default_user = st.session_state.get("profile_user_id")
        default_index = user_options.index(default_user) if default_user in user_options else 0
        selected_user = st.selectbox("Chọn user trong CSV upload", user_options, index=default_index, key="user360_uploaded_user")
        user_rows = working[working["UserID"].astype(str) == str(selected_user)].copy()

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Tổng dòng dataset", f"{len(scored_upload):,}", "CSV upload")
        with c2:
            metric_card("User", str(selected_user), "Từ file upload")
        with c3:
            metric_card("Số giao dịch", f"{len(user_rows):,}", "Của user đang chọn")
        with c4:
            metric_card("Trust trung bình", f"{user_rows['Trust_Score'].mean():.2f}", "Theo user")
        with c5:
            metric_card("Giao dịch rủi ro", f"{int((user_rows['Trust_Score'] < 50).sum()):,}", "Trust < 50")

        section_title("Hồ sơ người dùng")
        profile_cols = [c for c in ["UserID", "DeviceType", "DeviceInfo", "P_emaildomain", "R_emaildomain", "addr1", "addr2", "TransactionAmt"] if c in user_rows.columns]
        if profile_cols:
            st.dataframe(user_rows[profile_cols].head(1), use_container_width=True, hide_index=True)
        else:
            st.info("CSV chưa có đủ trường hồ sơ chi tiết, hệ thống hiển thị thông tin giao dịch liên quan.")

        section_title("Hành vi của người dùng")
        behavior_cols = [c for c in [ID_COL, "TransactionDT", "TransactionAmt", "ProductCD", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in user_rows.columns]
        st.dataframe(user_rows[behavior_cols].sort_values("Trust_Score"), use_container_width=True, hide_index=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            section_title("Trust Score theo giao dịch")
            chart_df = user_rows[["Trust_Score"]].reset_index(drop=True)
            chart_df.index = chart_df.index + 1
            st.line_chart(chart_df, use_container_width=True)
        with chart_right:
            section_title("Phân bố rủi ro")
            if "Risk_Level" in user_rows.columns:
                st.bar_chart(user_rows["Risk_Level"].value_counts().rename_axis("Risk_Level").to_frame("count"), use_container_width=True)
            else:
                st.info("Chưa có cột Risk_Level.")

        st.markdown("---")
        if st.button("Gửi feedback cho user này", key="send_feedback_from_user360_upload", use_container_width=True):
            st.session_state["feedback_user_id"] = str(selected_user)
            st.session_state["profile_user_id"] = str(selected_user)
            st.session_state["return_to_user360"] = True
            set_page("Feedback")
            st.rerun()

        if st.session_state.get("return_to_case_study"):
            st.markdown("---")
            if st.button("Quay lại Case study", key="back_to_case_from_user360_upload", use_container_width=True):
                st.session_state.pop("return_to_case_study", None)
                set_page("Case study")
                st.rerun()
        return

    if behavior_df.empty:
        st.warning("Chưa có dữ liệu user trong database.")
        return

    user_options = behavior_df["user_id"].astype(str).tolist()
    default_user = st.session_state.get("profile_user_id")
    default_index = user_options.index(default_user) if default_user in user_options else 0
    user_id = st.selectbox("Chọn user_id", user_options, index=default_index, key="user360_db_user")
    row = behavior_df[behavior_df["user_id"].astype(str) == str(user_id)].iloc[0]
    profile = user_profile(DB_PATH, str(user_id), reviewer=None if role == "Admin" else username)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Tổng dòng dataset", f"{len(behavior_df):,}", "User demo")
    with c2:
        metric_card("Trust dài hạn", f"{row['long_term_trust_score']:.2f}", row["long_term_risk_level"])
    with c3:
        metric_card("Đăng nhập 30 ngày", f"{int(row['login_count_30d']):,}", f"Failed: {row['failed_login_rate_30d']*100:.1f}%")
    with c4:
        metric_card("Giao dịch 30 ngày", f"{int(row['transaction_count_30d']):,}", f"Avg ML Trust: {row['ml_avg_trust_score_30d']:.2f}")
    with c5:
        metric_card("Thiết bị / IP", f"{int(row['device_change_count_30d'])}/{int(row['address_change_count_30d'])}", "Biến động 30 ngày")

    section_title("Hồ sơ người dùng")
    top_left, top_right = st.columns([1.05, 0.95], gap="large")
    with top_left:
        st.dataframe(profile["user"], use_container_width=True, hide_index=True)
        if not profile.get("devices", pd.DataFrame()).empty:
            st.markdown("**Thiết bị đã ghi nhận**")
            st.dataframe(profile["devices"], use_container_width=True, hide_index=True)
    with top_right:
        if not profile.get("addresses", pd.DataFrame()).empty:
            st.markdown("**Địa chỉ / IP**")
            st.dataframe(profile["addresses"], use_container_width=True, hide_index=True)
        if not profile.get("logins", pd.DataFrame()).empty:
            st.markdown("**Đăng nhập gần đây**")
            st.dataframe(profile["logins"].head(10), use_container_width=True, hide_index=True)

    section_title("Hành vi của người dùng")
    st.markdown(f"<div class='soft-card muted'>{row['explanation']}</div>", unsafe_allow_html=True)

    time_left, time_right = st.columns([1, 1], gap="large")
    with time_left:
        timeline = load_timeline_cached(DB_PATH, str(user_id))
        section_title("Diễn biến theo thời gian")
        if not timeline.empty:
            chart_df = timeline.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.set_index("date")
            available = [c for c in ["avg_trust_score", "login_count", "failed_login_count", "unique_devices", "unique_addresses", "transaction_count"] if c in chart_df.columns]
            st.line_chart(chart_df[available], use_container_width=True)
        else:
            st.info("Chưa có timeline.")
    with time_right:
        section_title("Bảng chỉ số hành vi")
        behavior_cols = [
            "user_id", "full_name", "long_term_trust_score", "behavior_risk_score", "failed_login_rate_30d",
            "night_login_rate_30d", "device_change_count_30d", "address_change_count_30d", "high_velocity_hours_30d",
            "negative_feedback_count_30d", "explanation",
        ]
        st.dataframe(behavior_df[behavior_cols][behavior_df["user_id"].astype(str) == str(user_id)], use_container_width=True, hide_index=True)

    section_title("Giao dịch của người dùng")
    tx_tabs = st.tabs(["Giao dịch gần đây", "Giao dịch rủi ro"])
    with tx_tabs[0]:
        st.dataframe(profile.get("transactions", pd.DataFrame()), use_container_width=True, hide_index=True)
    with tx_tabs[1]:
        st.dataframe(profile.get("risk_transactions", pd.DataFrame()), use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("Gửi feedback cho user này", key="send_feedback_from_user360_db", use_container_width=True):
        st.session_state["feedback_user_id"] = str(user_id)
        st.session_state["profile_user_id"] = str(user_id)
        st.session_state["return_to_user360"] = True
        if force_db_view:
            st.session_state["case_force_db_user_view"] = True
        set_page("Feedback")
        st.rerun()

    if st.session_state.get("return_to_case_study"):
        st.markdown("---")
        if st.button("Quay lại Case study", key="back_to_case_from_user360_db", use_container_width=True):
            st.session_state.pop("return_to_case_study", None)
            set_page("Case study")
            st.rerun()

def render_user_profile(role: str, username: str):
    hero("Hồ sơ người dùng", "Xem chi tiết từng người dùng: thông tin cơ bản, thiết bị, địa chỉ/IP, đăng nhập, giao dịch rủi ro và feedback.", "User Profile")
    if behavior_df.empty:
        st.warning("Chưa có dữ liệu user.")
        return
    user_options = behavior_df["user_id"].astype(str).tolist()
    default_user = str(st.session_state.get("profile_user_id", user_options[0] if user_options else ""))
    default_index = user_options.index(default_user) if default_user in user_options else 0
    user_id = st.selectbox("Chọn user_id", user_options, index=default_index)
    row = behavior_df[behavior_df["user_id"].astype(str) == str(user_id)].iloc[0]
    profile = user_profile(DB_PATH, user_id, reviewer=None if role == "Admin" else username)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Trust dài hạn", f"{row['long_term_trust_score']:.2f}", row["long_term_risk_level"])
    with c2:
        metric_card("Thiết bị", f"{int(row['unique_devices_30d']):,}", "30 ngày")
    with c3:
        metric_card("Địa chỉ/IP", f"{int(row['unique_addresses_30d']):,}", "30 ngày")
    with c4:
        metric_card("Feedback rủi ro", f"{int(row['negative_feedback_count_30d']):,}", "30 ngày")

    tabs = st.tabs(["Thông tin", "Thiết bị/IP", "Đăng nhập", "Giao dịch", "Feedback"])
    with tabs[0]:
        st.dataframe(profile["user"], use_container_width=True, hide_index=True)
        st.markdown(f"<div class='soft-card muted'>{row['explanation']}</div>", unsafe_allow_html=True)
    with tabs[1]:
        st.dataframe(profile["devices"], use_container_width=True, hide_index=True)
        st.dataframe(profile["addresses"], use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(profile["logins"], use_container_width=True, hide_index=True)
    with tabs[3]:
        st.markdown("**Giao dịch gần đây**")
        st.dataframe(profile["transactions"], use_container_width=True, hide_index=True)
        st.markdown("**Giao dịch rủi ro nhất**")
        st.dataframe(profile["risk_transactions"], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(profile.get("feedback", pd.DataFrame()), use_container_width=True, hide_index=True)

    if st.session_state.get("return_to_case_study"):
        st.markdown("---")
        if st.button("Quay lại Case study", key="back_to_case_from_profile", use_container_width=True):
            st.session_state.pop("return_to_case_study", None)
            set_page("Case study")
            st.rerun()



def _pick_case_users() -> Dict[str, str]:
    if behavior_df.empty:
        return {}
    cases = {}
    sorted_df = behavior_df.sort_values("long_term_trust_score")
    cases["User rủi ro cao"] = str(sorted_df.iloc[0]["user_id"])
    cases["User ổn định"] = str(sorted_df.iloc[-1]["user_id"])
    takeover_df = behavior_df.copy()
    takeover_df["takeover_signal"] = (
        takeover_df.get("failed_login_rate_30d", 0) * 100
        + takeover_df.get("device_change_count_30d", 0) * 10
        + takeover_df.get("address_change_count_30d", 0) * 10
        + takeover_df.get("night_login_rate_30d", 0) * 40
    )
    cases["Nghi ngờ chiếm tài khoản"] = str(takeover_df.sort_values("takeover_signal", ascending=False).iloc[0]["user_id"])
    return cases



def _pick_uploaded_case_users(scored_df: pd.DataFrame) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Pick case-study users from the currently uploaded/scored CSV.

    The original case study uses the demo database because it has login/device
    history. When a CSV is uploaded, the case study should be based on that CSV
    instead, so this helper creates scenario labels from transaction-level
    Trust Score and risk statistics.
    """
    if scored_df.empty:
        return {}, pd.DataFrame()
    working = scored_df.copy()
    if "UserID" not in working.columns:
        working["UserID"] = [f"ROW_{i+1:04d}" for i in range(len(working))]
    summary = summarize_uploaded_behavior(working)
    if summary.empty:
        return {}, pd.DataFrame()

    summary = summary.copy()
    summary["user_id"] = summary["user_id"].astype(str)
    summary["risk_weight"] = (
        (100 - summary["avg_trust_score"].astype(float))
        + summary["high_risk_transactions"].astype(float) * 8
        + summary["transaction_count"].astype(float).clip(upper=50) * 0.15
    )

    stable_row = summary.sort_values(["avg_trust_score", "high_risk_transactions"], ascending=[False, True]).iloc[0]
    risky_row = summary.sort_values(["avg_trust_score", "high_risk_transactions"], ascending=[True, False]).iloc[0]
    suspicious_row = summary.sort_values(["risk_weight", "transaction_count"], ascending=[False, False]).iloc[0]

    cases = {
        "User ổn định": str(stable_row["user_id"]),
        "User rủi ro cao": str(risky_row["user_id"]),
        "Nghi ngờ chiếm tài khoản": str(suspicious_row["user_id"]),
    }
    return cases, summary

def render_case_study(role: str, username: str):
    using_upload = using_uploaded_dataset()
    hero(
        "Case study hành vi người dùng",
        "Nếu đã upload CSV, các kịch bản case study sẽ được tạo từ dữ liệu upload hiện tại; nếu chưa upload, hệ thống dùng dữ liệu demo trong database.",
        "Demo Storytelling • Risk Scenario • User Journey",
    )

    if using_upload:
        st.session_state.pop("case_force_db_user_view", None)
        render_active_dataset_notice()
        scored_upload = get_case_study_uploaded_dataset()
        cases, upload_summary = _pick_uploaded_case_users(scored_upload)
        if not cases or upload_summary.empty:
            st.info("CSV upload chưa đủ dữ liệu đã chấm điểm để tạo case study.")
            return

        selected_case = st.selectbox("Chọn kịch bản demo", list(cases.keys()))
        user_id = str(cases[selected_case])
        row = upload_summary[upload_summary["user_id"].astype(str) == user_id].iloc[0]
        working = scored_upload.copy()
        if "UserID" not in working.columns:
            working["UserID"] = [f"ROW_{i+1:04d}" for i in range(len(working))]
        user_rows = working[working["UserID"].astype(str) == user_id].copy()
        user_rows = sort_uploaded_user_timeline(user_rows)
        risk_rows = user_rows[user_rows["Trust_Score"] < 50].copy() if "Trust_Score" in user_rows.columns else pd.DataFrame()

        avg_trust = float(row.get("avg_trust_score", 0.0))
        min_trust = float(row.get("min_trust_score", avg_trust))
        high_risk_count = int(row.get("high_risk_transactions", 0))
        tx_count = int(row.get("transaction_count", len(user_rows)))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("User", user_id, selected_case)
        with c2:
            metric_card("Trust trung bình", f"{avg_trust:.2f}", "Từ CSV upload")
        with c3:
            metric_card("Trust thấp nhất", f"{min_trust:.2f}", "Giao dịch rủi ro nhất")
        with c4:
            metric_card("Giao dịch rủi ro", f"{high_risk_count:,}/{tx_count:,}", "Trust < 50")

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            section_title("Câu chuyện rủi ro từ CSV upload")
            if selected_case == "User ổn định":
                narrative = (
                    f"User {user_id} có Trust trung bình {avg_trust:.2f}/100 và số giao dịch rủi ro thấp trong CSV hiện tại. "
                    "Đây là ví dụ cho nhóm user có hành vi giao dịch tương đối ổn định trong file upload."
                )
            elif selected_case == "Nghi ngờ chiếm tài khoản":
                narrative = (
                    f"User {user_id} có tổ hợp tín hiệu cần chú ý: Trust trung bình {avg_trust:.2f}/100, "
                    f"Trust thấp nhất {min_trust:.2f}/100 và {high_risk_count} giao dịch Trust < 50. "
                    "Kịch bản này phù hợp để minh họa nguy cơ bất thường trong dữ liệu upload."
                )
            else:
                narrative = (
                    f"User {user_id} có điểm tin cậy thấp hơn các user khác trong CSV, với Trust trung bình {avg_trust:.2f}/100 "
                    f"và {high_risk_count} giao dịch rủi ro. Nên ưu tiên review hoặc theo dõi thêm."
                )
            st.markdown(f"<div class='soft-card muted'>{narrative}<br><br><b>Lý do hệ thống:</b> {row.get('explanation', '')}</div>", unsafe_allow_html=True)

            section_title("Khuyến nghị xử lý")
            recommendations = []
            if avg_trust < 50 or high_risk_count >= 2:
                recommendations.append("Đưa user vào danh sách cần review và kiểm tra các giao dịch Trust thấp.")
            if min_trust < 35:
                recommendations.append("Ưu tiên kiểm tra giao dịch có Trust thấp nhất vì rủi ro cao hơn phần còn lại.")
            if tx_count >= 20 and avg_trust < 80:
                recommendations.append("Theo dõi thêm vì user có nhiều giao dịch và Trust chưa thật sự ổn định.")
            if not recommendations:
                recommendations.append("Tiếp tục theo dõi định kỳ, chưa cần can thiệp mạnh.")
            st.markdown("\n".join([f"- {rec}" for rec in recommendations]))

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Mở hồ sơ user này", key="case_open_user_profile_upload", use_container_width=True):
                    st.session_state["profile_user_id"] = user_id
                    st.session_state["return_to_case_study"] = True
                    st.session_state["case_context_user_id"] = user_id
                    st.session_state.pop("case_force_db_user_view", None)
                    set_page("Xem theo từng user")
                    st.rerun()
            with col_b:
                if st.button("Gửi sang Feedback", key="case_send_to_feedback_upload", use_container_width=True):
                    st.session_state["feedback_user_id"] = user_id
                    st.session_state["return_to_case_study"] = True
                    st.session_state["case_context_user_id"] = user_id
                    set_page("Feedback")
                    st.rerun()

        with right:
            section_title("Diễn biến Trust Score trong CSV")
            if not user_rows.empty and "Trust_Score" in user_rows.columns:
                chart_df = user_rows[["Trust_Score"]].copy().reset_index(drop=True)
                chart_df.index = chart_df.index + 1
                st.line_chart(chart_df, use_container_width=True)
                if chart_df["Trust_Score"].nunique(dropna=True) <= 1:
                    st.caption("Trust Score của user này không đổi giữa các giao dịch trong CSV, nên đường biểu đồ nằm ngang. Hãy chọn kịch bản user rủi ro/cần theo dõi hoặc dùng CSV có Trust Score biến thiên để thấy diễn biến rõ hơn.")
            else:
                st.info("Chưa có dữ liệu Trust Score cho user này.")

            section_title("Phân bố Risk Level")
            if not user_rows.empty and "Risk_Level" in user_rows.columns:
                st.bar_chart(user_rows["Risk_Level"].value_counts().rename_axis("Risk_Level").to_frame("count"), use_container_width=True)
            else:
                st.info("Chưa có Risk Level.")

        tabs = st.tabs(["Giao dịch của user", "Giao dịch rủi ro", "Feedback"])
        display_cols = [c for c in [ID_COL, "UserID", "TransactionAmt", "ProductCD", "Fraud_Probability", "Trust_Score", "Risk_Level", "DeviceType", "DeviceInfo"] if c in user_rows.columns]
        with tabs[0]:
            st.dataframe(user_rows[display_cols] if display_cols else user_rows, use_container_width=True, hide_index=True)
        with tabs[1]:
            st.dataframe(risk_rows[display_cols] if not risk_rows.empty and display_cols else risk_rows, use_container_width=True, hide_index=True)
        with tabs[2]:
            st.dataframe(read_feedback(DB_PATH, limit=200, reviewer=None if role == "Admin" else username).query("user_id == @user_id") if not read_feedback(DB_PATH, limit=200, reviewer=None if role == "Admin" else username).empty else pd.DataFrame(), use_container_width=True, hide_index=True)
        return

    hero_note = st.empty()
    cases = _pick_case_users()
    if not cases:
        st.info("Chưa có dữ liệu hành vi để tạo case study.")
        return

    selected_case = st.selectbox("Chọn kịch bản demo", list(cases.keys()))
    user_id = cases[selected_case]
    row = behavior_df[behavior_df["user_id"].astype(str) == str(user_id)].iloc[0]
    profile = user_profile(DB_PATH, user_id, reviewer=None if role == "Admin" else username)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("User", user_id, selected_case)
    with c2:
        metric_card("Long-term Trust", f"{row['long_term_trust_score']:.2f}", row["long_term_risk_level"])
    with c3:
        metric_card("Login failed", f"{row['failed_login_rate_30d']*100:.1f}%", "30 ngày")
    with c4:
        metric_card("Device/IP changes", f"{int(row['device_change_count_30d'])}/{int(row['address_change_count_30d'])}", "Thiết bị / địa chỉ")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        section_title("Câu chuyện rủi ro")
        if selected_case == "User ổn định":
            narrative = (
                "User này có Trust Score dài hạn cao, tần suất đăng nhập và giao dịch ổn định, ít thay đổi thiết bị/IP. "
                "Đây là ví dụ cho nhóm người dùng có hành vi đáng tin cậy."
            )
        elif selected_case == "Nghi ngờ chiếm tài khoản":
            narrative = (
                "User này có nhiều tín hiệu bất thường liên quan đến đăng nhập, thiết bị hoặc địa chỉ/IP. "
                "Kịch bản phù hợp để mô tả nguy cơ tài khoản bị truy cập từ môi trường lạ."
            )
        else:
            narrative = (
                "User này có điểm tin cậy thấp và nhiều tín hiệu rủi ro trong 30 ngày gần đây. "
                "Nên đưa vào danh sách theo dõi hoặc yêu cầu kiểm tra thủ công."
            )
        st.markdown(f"<div class='soft-card muted'>{narrative}<br><br><b>Lý do hệ thống:</b> {row['explanation']}</div>", unsafe_allow_html=True)

        section_title("Khuyến nghị xử lý")
        recommendations = []
        if row["long_term_trust_score"] < 50:
            recommendations.append("Đưa user vào danh sách watchlist và yêu cầu review thủ công.")
        if row["failed_login_rate_30d"] >= 0.18:
            recommendations.append("Yêu cầu đổi mật khẩu hoặc xác thực bổ sung do tỷ lệ login thất bại cao.")
        if row["device_change_count_30d"] >= 2 or row["address_change_count_30d"] >= 2:
            recommendations.append("Kiểm tra thiết bị/IP mới và xác minh người dùng.")
        if not recommendations:
            recommendations.append("Tiếp tục theo dõi định kỳ, chưa cần can thiệp mạnh.")
        st.markdown("\n".join([f"- {rec}" for rec in recommendations]))

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Mở hồ sơ user này", key="case_open_user_profile", use_container_width=True):
                st.session_state["profile_user_id"] = user_id
                st.session_state["return_to_case_study"] = True
                st.session_state["case_context_user_id"] = user_id
                st.session_state["case_force_db_user_view"] = True
                set_page("Xem theo từng user")
                st.rerun()
        with col_b:
            if st.button("Gửi sang Feedback", key="case_send_to_feedback", use_container_width=True):
                st.session_state["feedback_user_id"] = user_id
                st.session_state["return_to_case_study"] = True
                st.session_state["case_context_user_id"] = user_id
                set_page("Feedback")
                st.rerun()

    with right:
        section_title("Timeline hành vi")
        timeline = load_timeline_cached(DB_PATH, user_id)
        if not timeline.empty:
            chart_df = timeline.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.set_index("date")
            cols = [c for c in ["avg_trust_score", "login_count", "failed_login_count", "unique_devices", "unique_addresses"] if c in chart_df.columns]
            st.line_chart(chart_df[cols], use_container_width=True)
        else:
            st.info("Chưa có timeline.")

    tabs = st.tabs(["Login gần đây", "Giao dịch rủi ro", "Feedback"])
    with tabs[0]:
        st.dataframe(profile.get("logins", pd.DataFrame()), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(profile.get("risk_transactions", pd.DataFrame()), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(profile.get("feedback", pd.DataFrame()), use_container_width=True, hide_index=True)


def detect_uploaded_anomalies(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based anomaly detection for the currently uploaded CSV.

    The original anomaly page reads long-term behavior from SQLite demo tables.
    When a CSV is uploaded, this helper derives equivalent signals directly from
    the uploaded file so the demo stays consistent with the active dataset.
    """
    if scored_df.empty:
        return pd.DataFrame()

    df = scored_df.copy()
    if "UserID" not in df.columns:
        if ID_COL in df.columns:
            df["UserID"] = df[ID_COL].astype(str)
        else:
            df["UserID"] = [f"row_{i + 1:04d}" for i in range(len(df))]
    df["UserID"] = df["UserID"].astype(str)

    if "Trust_Score" not in df.columns:
        if "Fraud_Probability" in df.columns:
            df["Fraud_Probability"] = pd.to_numeric(df["Fraud_Probability"], errors="coerce").fillna(0).clip(0, 1)
            df["Trust_Score"] = (100 * (1 - df["Fraud_Probability"])).clip(0, 100)
        else:
            return pd.DataFrame()
    df["Trust_Score"] = pd.to_numeric(df["Trust_Score"], errors="coerce").fillna(100).clip(0, 100)
    if "Fraud_Probability" not in df.columns:
        df["Fraud_Probability"] = (1 - df["Trust_Score"] / 100).clip(0, 1)
    else:
        df["Fraud_Probability"] = pd.to_numeric(df["Fraud_Probability"], errors="coerce").fillna(0).clip(0, 1)

    def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
        if col not in row.index:
            return default
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else default

    rows = []
    for user_id, group in df.groupby("UserID", dropna=False):
        tx_count = int(len(group))
        avg_trust = float(group["Trust_Score"].mean()) if tx_count else 100.0
        low_trust_rate = float((group["Trust_Score"] < 50).mean()) if tx_count else 0.0
        max_fraud = float(group["Fraud_Probability"].max()) if tx_count else 0.0

        # Prefer explicit behavior columns from the uploaded CSV. If they are
        # absent, derive simple counts from available device/IP columns.
        first = group.iloc[0]
        failed_login = max(float(pd.to_numeric(group.get("failed_login_count_30d", pd.Series([0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)
        night_login = max(float(pd.to_numeric(group.get("night_login_count_30d", pd.Series([0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)
        unique_devices = max(float(pd.to_numeric(group.get("unique_device_count_30d", pd.Series([group["DeviceInfo"].nunique() if "DeviceInfo" in group.columns else 0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)
        unique_ips = max(float(pd.to_numeric(group.get("unique_ip_count_30d", pd.Series([group["IPAddress"].nunique() if "IPAddress" in group.columns else 0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)
        chargebacks = max(float(pd.to_numeric(group.get("chargeback_count_90d", pd.Series([0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)
        feedback_flag = max(float(pd.to_numeric(group.get("feedback_risk_flag", pd.Series([0] * tx_count)), errors="coerce").fillna(0).max()), 0.0)

        rules = [
            ("LOW_TRUST_RATE", "Tỷ lệ giao dịch Trust Score thấp cao trong CSV upload", low_trust_rate, 0.15, "High", "tỷ lệ"),
            ("MAX_FRAUD_PROB", "Có giao dịch có xác suất gian lận cao trong CSV upload", max_fraud, 0.50, "High", "xác suất"),
            ("FAILED_LOGIN", "Số lần đăng nhập thất bại cao trong CSV upload", failed_login, 3, "High", "lần"),
            ("NIGHT_LOGIN", "Có nhiều đăng nhập ban đêm trong CSV upload", night_login, 3, "Medium", "lần"),
            ("DEVICE_CHANGE", "Thay đổi/nhiều thiết bị trong CSV upload", unique_devices, 3, "Medium", "thiết bị"),
            ("ADDRESS_CHANGE", "Thay đổi/nhiều IP trong CSV upload", unique_ips, 3, "Medium", "IP"),
            ("CHARGEBACK", "Có chargeback/hoàn tiền rủi ro trong CSV upload", chargebacks, 1, "High", "lần"),
            ("NEGATIVE_FEEDBACK", "Có cờ feedback rủi ro trong CSV upload", feedback_flag, 1, "High", "cờ"),
        ]
        for rule_id, desc, value, threshold, severity, unit in rules:
            try:
                is_hit = float(value) >= float(threshold)
            except Exception:
                is_hit = False
            if is_hit:
                rows.append(
                    {
                        "user_id": str(user_id),
                        "rule_id": rule_id,
                        "severity": severity,
                        "value": round(float(value), 4),
                        "threshold": threshold,
                        "unit": unit,
                        "transaction_count": tx_count,
                        "long_term_trust_score": round(avg_trust, 2),
                        "description": desc,
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        result["severity_order"] = result["severity"].map(severity_order).fillna(3)
        result = result.sort_values(["severity_order", "long_term_trust_score", "user_id"], ascending=[True, True, True]).drop(columns=["severity_order"])
    return result

def render_anomaly(role: str, username: str):
    hero("Phát hiện bất thường", "Rule-based anomaly detection dựa trên login thất bại, đăng nhập ban đêm, thay đổi thiết bị/IP, giao dịch dồn dập và feedback rủi ro.", "Anomaly Detection")

    if using_uploaded_dataset():
        render_active_dataset_notice()
        source_name = st.session_state.get("active_uploaded_file_name", "CSV upload")
        scored_upload = get_case_study_uploaded_dataset()
        anomalies = detect_uploaded_anomalies(scored_upload)
        source_label = f"CSV upload: {source_name}"
        source_note = "Đang phát hiện bất thường từ dữ liệu CSV đã upload, không dùng database demo."
    else:
        anomalies = detect_anomalies(DB_PATH, window_days=30)
        source_label = "Database demo"
        source_note = "Chưa có CSV upload nên hệ thống dùng dữ liệu demo trong database."

    st.caption(source_note)
    if anomalies.empty:
        st.success("Không phát hiện bất thường trong dữ liệu hiện tại.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Nguồn dữ liệu", source_label, "Anomaly input")
    with c2:
        metric_card("Tổng cảnh báo", f"{len(anomalies):,}", "Rule hits")
    with c3:
        metric_card("High severity", f"{int((anomalies['severity'] == 'High').sum()):,}", "Cảnh báo cao")
    with c4:
        metric_card("User bị cảnh báo", f"{anomalies['user_id'].nunique():,}", "Unique users")

    severity_options = sorted(anomalies["severity"].dropna().unique().tolist())
    severity = st.multiselect("Lọc severity", severity_options, default=severity_options)
    filtered = anomalies[anomalies["severity"].isin(severity)].copy()

    if using_uploaded_dataset() and not filtered.empty:
        cols_to_show = [
            "user_id", "rule_id", "severity", "value", "threshold", "unit",
            "transaction_count", "long_term_trust_score", "description",
        ]
        cols_to_show = [c for c in cols_to_show if c in filtered.columns]
        st.dataframe(filtered[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_monitoring(role: str, username: str):
    hero("Monitoring & Drift", "Theo dõi số lượng prediction, phân phối rủi ro, audit action, model version và tín hiệu vận hành của hệ thống.", "Operational Monitoring")
    summary = cached_monitoring_summary(DB_PATH)
    m = summary["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Predictions", f"{m['prediction_count']:,}", "Tổng dự đoán")
    with c2:
        metric_card("Avg Trust", f"{m['avg_trust_score']:.2f}", "Điểm trung bình")
    with c3:
        metric_card("High risk", f"{m['high_risk_prediction_count']:,}", "Trust < 50")
    with c4:
        if role == "Admin":
            metric_card("Feedback", f"{m['feedback_count']:,}", "Toàn bộ review")
        else:
            user_feedback_count = len(read_feedback(DB_PATH, limit=1000, reviewer=username))
            metric_card("Feedback của tôi", f"{user_feedback_count:,}", "Review của tài khoản")

    left, right = st.columns(2)
    with left:
        section_title("Prediction theo ngày")
        if not summary["daily_predictions"].empty:
            chart = summary["daily_predictions"].copy()
            chart["created_date"] = pd.to_datetime(chart["created_date"])
            st.line_chart(chart.set_index("created_date")[["prediction_count", "high_risk_count"]], use_container_width=True)
        section_title("Active model registry")
        st.dataframe(summary["active_models"], use_container_width=True, hide_index=True)
    with right:
        section_title("Phân phối Risk Level")
        if not summary["risk_distribution"].empty:
            st.bar_chart(summary["risk_distribution"].set_index("risk_level"), use_container_width=True)
        if role == "Admin":
            section_title("Audit actions")
            if not summary["audit_actions"].empty:
                st.bar_chart(summary["audit_actions"].set_index("action"), use_container_width=True)
        else:
            section_title("Audit actions")
            st.info("Chi tiết audit chỉ dành cho Admin. User thường không xem nhật ký hệ thống.")

    section_title("Data drift so với dữ liệu tham chiếu")
    drift_df = get_drift_report_cached()
    drift_m = drift_metrics(drift_df)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        metric_card("Features monitored", f"{drift_m['total']:,}", "Số feature kiểm tra")
    with d2:
        metric_card("High drift", f"{drift_m['high']:,}", "Cần kiểm tra")
    with d3:
        metric_card("Medium drift", f"{drift_m['medium']:,}", "Theo dõi")
    with d4:
        metric_card("Low drift", f"{drift_m['low']:,}", "Ổn định")
    if not drift_df.empty:
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa đủ dữ liệu để tính drift report.")


def render_batch(role: str, username: str):
    hero("Batch scoring", "Chấm điểm hàng loạt cho file CSV đang được chọn. Nếu muốn xem file khác, hãy upload file mới hoặc chọn lại từ lịch sử upload.", "Bulk Prediction • Export")
    active_df = require_uploaded_dataset("Batch scoring")
    if active_df.empty:
        return

    scored_batch = get_scored_uploaded_dataset().head(5000).copy()
    total = len(scored_batch)
    low = int((scored_batch["Trust_Score"] < 50).sum())
    medium = int(((scored_batch["Trust_Score"] >= 50) & (scored_batch["Trust_Score"] < 80)).sum())
    high = int((scored_batch["Trust_Score"] >= 80).sum())

    batch_entity = str(st.session_state.get("active_uploaded_file_name", "CSV upload"))
    batch_detail = f"Auto batch scoring audit; rows={total}; low_trust={low}; medium={medium}; high_trust={high}"
    audit_once(username, role, "BATCH_SCORING", "dataset", batch_entity, batch_detail)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Transactions", f"{total:,}", "Số dòng phân tích")
    with c2:
        metric_card("High Trust", f"{high:,}", "Trust ≥ 80")
    with c3:
        metric_card("Medium", f"{medium:,}", "50 ≤ Trust < 80")
    with c4:
        metric_card("Low Trust", f"{low:,}", "Trust < 50")

    chart_left, chart_right = st.columns([0.95, 1.05], gap="large")
    with chart_left:
        section_title("Phân bố kết quả")
        chart_df = pd.DataFrame({"Nhóm": ["Cao", "Trung bình", "Thấp"], "Số lượng": [high, medium, low]}).set_index("Nhóm")
        st.bar_chart(chart_df, use_container_width=True)
    with chart_right:
        section_title("Trust Score theo dòng")
        line_df = scored_batch[["Trust_Score"]].head(100).reset_index(drop=True)
        line_df.index = line_df.index + 1
        st.line_chart(line_df, use_container_width=True)

    section_title("Giao dịch rủi ro nhất")
    display_cols = [c for c in [ID_COL, TARGET, "UserID", "Fraud_Probability", "Trust_Score", "Risk_Level", "TransactionAmt", "ProductCD"] if c in scored_batch.columns]
    st.dataframe(scored_batch.sort_values("Trust_Score").head(50)[display_cols], use_container_width=True, hide_index=True)
    st.download_button("Tải kết quả CSV", data=scored_batch.to_csv(index=False, encoding="utf-8-sig"), file_name="trust_score_predictions.csv", mime="text/csv", use_container_width=True)


def render_explain(role: str, username: str):
    hero("Giải thích mô hình", "Giải thích kết quả của giao dịch trong file CSV đang được chọn. Nếu chưa có file, hãy upload hoặc chọn lại từ lịch sử upload.", "SHAP Explainable ML")
    status = shap_status()
    if status.get("available"):
        st.success(f"SHAP khả dụng trong môi trường hiện tại, version: {status.get('version')}.")
    else:
        st.info("SHAP chưa khả dụng trong môi trường hiện tại. App sẽ dùng giải thích dự phòng bằng feature importance.")

    active_df = require_uploaded_dataset("Giải thích mô hình")
    if active_df.empty:
        return

    idx = st.number_input(
        "Chọn index giao dịch để giải thích",
        min_value=0,
        max_value=max(len(active_df) - 1, 0),
        value=0,
        step=1,
        key="explain_row_index",
    )
    sample_df = active_df.iloc[[idx]].copy()
    scored_row, explanation_df = run_single_prediction(sample_df)
    render_prediction_panel(scored_row, "Kết quả giao dịch cần giải thích")

    section_title("Thông tin giao dịch")
    show_cols = [c for c in [ID_COL, "UserID", TARGET, "TransactionAmt", "ProductCD", "DeviceType", "DeviceInfo", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in scored_row.columns]
    st.dataframe(scored_row[show_cols], use_container_width=True, hide_index=True)

    section_title("Top feature ảnh hưởng")
    show_explanation_table(explanation_df)


def render_trust_chat(role: str, username: str):
    hero(
        "Trợ lý",
        "Trả lời các câu hỏi liên quan đến dataset CSV đã upload, rủi ro, drift, feedback/audit và cách hệ thống hoạt động.",
        "Dataset • System Q&A",
    )
    if not using_uploaded_dataset():
        st.info("Vui lòng vào trang Nhập dữ liệu và upload CSV trước. Trợ lý sẽ xuất hiện sau khi có dữ liệu upload.")
        return
    drift_df = get_drift_report_cached()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.chat_input("Hỏi về dataset hoặc hệ thống...")
    if user_question:
        active_upload = get_uploaded_dataset()
        scored_upload = get_scored_uploaded_dataset() if not active_upload.empty else pd.DataFrame()
        answer = answer_trust_question(
            user_question,
            reference_rows=len(reference_df),
            feature_count=len(feature_columns),
            behavior_df=pd.DataFrame(),
            drift_df=drift_df,
            upload_df=active_upload,
            scored_df=scored_upload,
            upload_report=get_uploaded_report(),
            db_stats={},
            current_page="Trợ lý",
        )
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    section_title("Thông tin nền để trợ lý trả lời")
    m = drift_metrics(drift_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Reference rows", f"{len(reference_df):,}", "Dataset giao dịch")
    with c2:
        metric_card("Features", f"{len(feature_columns):,}", "Model input")
    with c3:
        metric_card("Users", f"{len(behavior_df):,}", "Behavior DB")
    with c4:
        metric_card("High drift", f"{m['high']:,}", "Feature drift cao")

    with st.expander("Xem drift report dùng cho trợ lý"):
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

def render_feedback(role: str, username: str):
    hero("Feedback và review", "Ghi nhận quyết định kiểm tra user/giao dịch để tạo vòng lặp phản hồi cho hệ thống đánh giá độ tin cậy.", "Human-in-the-loop")
    user_ids = behavior_df["user_id"].astype(str).tolist() if not behavior_df.empty else []
    default_user = str(st.session_state.get("feedback_user_id", user_ids[0] if user_ids else ""))
    if default_user and default_user not in user_ids:
        # Cho phép gửi feedback cho user xuất hiện trong CSV upload dù chưa có trong database demo.
        user_ids = [default_user] + user_ids
    if not user_ids:
        st.warning("Chưa có dữ liệu user để ghi feedback.")
        if st.session_state.get("return_to_case_study"):
            st.markdown("---")
            if st.button("Quay lại Case study", key="back_to_case_from_feedback_empty", use_container_width=True):
                st.session_state.pop("return_to_case_study", None)
                set_page("Case study")
                st.rerun()
        if st.session_state.get("return_to_user360"):
            st.markdown("---")
            if st.button("Quay lại Xem theo từng user", key="back_to_user360_from_feedback_empty", use_container_width=True):
                st.session_state.pop("return_to_user360", None)
                set_page("Xem theo từng user")
                st.rerun()
        return
    user_id = st.selectbox("User cần review", user_ids, index=user_ids.index(default_user) if default_user in user_ids else 0)
    tx_df = pd.DataFrame()
    if user_id and user_id in behavior_df["user_id"].astype(str).tolist():
        tx_df = user_profile(DB_PATH, user_id).get("risk_transactions", pd.DataFrame())
    elif user_id:
        uploaded_scored = get_scored_uploaded_dataset()
        if not uploaded_scored.empty and "UserID" in uploaded_scored.columns:
            tx_df = uploaded_scored[uploaded_scored["UserID"].astype(str) == str(user_id)].copy()
    tx_id_col = "transaction_id" if "transaction_id" in tx_df.columns else (ID_COL if ID_COL in tx_df.columns else None)
    # TransactionID trong dữ liệu thực tế có thể là số hoặc chuỗi như TX001/ORDER_123.
    # Không ép int để tránh lỗi khi Feedback nhận user/giao dịch từ CSV upload.
    tx_options = [None] + (tx_df[tx_id_col].dropna().astype(str).tolist() if tx_id_col and not tx_df.empty else [])
    tx_id = st.selectbox("Giao dịch liên quan", tx_options, format_func=lambda x: "Không gắn giao dịch" if x is None else str(x))
    decision = st.selectbox("Quyết định", ["approved", "need_review", "watchlist", "confirmed_risk", "rejected"])
    note = st.text_area("Ghi chú", value="Ghi nhận review thủ công từ giao diện.")
    if st.button("Lưu feedback", use_container_width=True):
        add_feedback(user_id, tx_id, username, decision, note, DB_PATH)
        log_audit(username, role, "ADD_FEEDBACK", "user", user_id, f"decision={decision}; tx_id={tx_id}", DB_PATH)
        st.cache_data.clear()
        st.success("Đã lưu feedback và ghi audit log.")
    section_title("Feedback gần đây")
    st.dataframe(read_feedback(DB_PATH, limit=200, reviewer=None if role == "Admin" else username), use_container_width=True, hide_index=True)

    if st.session_state.get("return_to_user360"):
        st.markdown("---")
        if st.button("Quay lại Xem theo từng user", key="back_to_user360_from_feedback", use_container_width=True):
            st.session_state.pop("return_to_user360", None)
            set_page("Xem theo từng user")
            st.rerun()

    if st.session_state.get("return_to_case_study"):
        st.markdown("---")
        if st.button("Quay lại Case study", key="back_to_case_from_feedback", use_container_width=True):
            st.session_state.pop("return_to_case_study", None)
            set_page("Case study")
            st.rerun()


def render_data_manager(role: str, username: str):
    hero(
        "Upload dữ liệu",
        "Upload CSV để dùng ngay trong phiên đăng nhập. Nếu muốn dùng lại sau khi đăng xuất, hãy lưu file vào database và chọn lại từ lịch sử upload.",
        "Schema Mapping • Validation • Upload History",
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        section_title("Import nguồn dữ liệu")
        source_type = "transactions"
        source_name = st.text_input("Tên nguồn dữ liệu", value="transactions_import")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="source_upload_transactions")

        if uploaded is not None:
            try:
                standardized, mapping, mapping_table, report, valid = prepare_source_upload(
                    uploaded,
                    source_type,
                    reference_df=reference_df,
                    feature_columns=feature_columns,
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Dòng đọc được", f"{report.get('row_count', len(standardized)):,}", "Từ file upload")
                with c2:
                    metric_card("Cột đã map", f"{report.get('mapped_columns', len(mapping)):,}", "Theo schema chuẩn")
                with c3:
                    metric_card("Cột bỏ qua", f"{report.get('ignored_columns', 0):,}", "Không dùng")
                with c4:
                    metric_card("Trạng thái", "Hợp lệ" if valid else "Cần sửa", "Validation")

                with st.expander("Mapping cột và kiểm tra dữ liệu", expanded=not valid):
                    st.dataframe(mapping_table, use_container_width=True, hide_index=True)
                    if report.get("errors"):
                        st.error("; ".join(report["errors"]))
                    if report.get("warnings"):
                        st.warning("; ".join(report["warnings"][:10]))
                    preview_cols = list(standardized.columns[:12])
                    if preview_cols:
                        st.dataframe(standardized[preview_cols].head(20), use_container_width=True, hide_index=True)

                import_df = standardized.copy()
                if source_type == "transactions" and valid:
                    # Save the uploaded CSV to the current session first. Even if scoring
                    # fails temporarily, users can still move to other pages and retry.
                    st.session_state["active_uploaded_df"] = standardized.copy()
                    st.session_state["active_uploaded_report"] = dict(report)
                    st.session_state["active_uploaded_file_name"] = getattr(uploaded, "name", "uploaded.csv")
                    st.session_state["active_uploaded_source_page"] = "Nhập dữ liệu"
                    st.session_state["active_uploaded_version"] = st.session_state.get("active_uploaded_version", 0) + 1
                    st.session_state.pop("active_uploaded_scored_df", None)
                    st.session_state.pop("active_uploaded_scored_version", None)
                    st.success("CSV giao dịch đã được dùng cho các trang phân tích trong phiên đăng nhập hiện tại.")
                    upload_file_name = getattr(uploaded, "name", "uploaded.csv")
                    upload_detail = f"Auto upload audit; file={upload_file_name}; rows={report.get('row_count', len(standardized))}; mapped_columns={report.get('mapped_columns', len(mapping))}; valid={valid}"
                    audit_once(username, role, "UPLOAD_DATA", "dataset", upload_file_name, upload_detail)
                    try:
                        scored_preview = run_batch_prediction(import_df.head(5000))
                        if isinstance(scored_preview, pd.DataFrame) and not scored_preview.empty:
                            import_df = scored_preview
                    except Exception as score_exc:
                        st.warning(f"CSV đã được nhận, nhưng hệ thống chưa scoring được ngay lúc này. Bạn vẫn có thể lưu/upload và thử lại ở trang Dự đoán. Chi tiết: {score_exc}")

                if st.button("Lưu vào database", disabled=not valid, use_container_width=True):
                    source_id, inserted = import_source_dataframe(
                        import_df,
                        source_type=source_type,
                        source_name=source_name.strip() or source_type,
                        file_name=getattr(uploaded, "name", "uploaded.csv"),
                        uploaded_by=username,
                        db_path=DB_PATH,
                        validation_report=report,
                        storage_df=import_df,
                    )
                    log_audit(username, role, "IMPORT_DATA_SOURCE", "data_source", str(source_id), f"Imported {inserted} rows as {source_type}", DB_PATH)
                    st.cache_data.clear()
                    st.success(f"Đã lưu {inserted:,} dòng vào database. Source ID: {source_id}")
                    st.rerun()
            except Exception as exc:
                st.error(f"Không thể xử lý file upload: {exc}")

    with right:
        # Dùng lại đúng khu vực lịch sử upload hiện có, không tạo thêm menu mới.
        # User chỉ xem lịch sử của mình; Admin xem được toàn bộ file/dataset đã lưu vào database.
        is_admin = role == "Admin"
        if is_admin:
            section_title("Lịch sử upload toàn hệ thống")
            history = read_data_sources(DB_PATH, uploaded_by=None)
        else:
            section_title("Lịch sử upload theo tài khoản")
            history = read_data_sources(DB_PATH, uploaded_by=username)

        if history.empty:
            st.info("Chưa có lịch sử upload nào." if is_admin else "Tài khoản hiện tại chưa có lịch sử upload.")
        else:
            base_cols = ["source_id", "source_name", "source_type", "file_name", "row_count", "uploaded_at", "status"]
            show_cols = ["uploaded_by"] + base_cols if is_admin and "uploaded_by" in history.columns else base_cols
            table = history[show_cols].copy()
            table.insert(0, "Dùng", False)
            table.insert(1, "Xem", False)
            table.insert(2, "Xóa", False)
            edited = st.data_editor(
                table,
                use_container_width=True,
                hide_index=True,
                disabled=show_cols,
                key="upload_history_editor_admin" if is_admin else "upload_history_editor",
            )

            if edited["Xóa"].any():
                source_id = int(edited.loc[edited["Xóa"] == True, "source_id"].iloc[0])
                delete_owner = None if is_admin else username
                if delete_data_source(source_id, uploaded_by=delete_owner, db_path=DB_PATH):
                    action = "ADMIN_DELETE_DATA_SOURCE" if is_admin else "DELETE_DATA_SOURCE"
                    log_audit(username, role, action, "data_source", str(source_id), "Xóa nguồn dữ liệu đã lưu", DB_PATH)
                    st.cache_data.clear()
                    st.success("Đã xóa nguồn dữ liệu khỏi lịch sử upload.")
                    st.rerun()

            if edited["Dùng"].any():
                source_id = int(edited.loc[edited["Dùng"] == True, "source_id"].iloc[0])
                source_row = history[history["source_id"] == source_id].iloc[0]
                if source_row["source_type"] != "transactions":
                    st.warning("Chỉ nguồn dữ liệu giao dịch mới có thể nạp lại để phân tích.")
                else:
                    restored_df = load_saved_source_dataframe(source_id, DB_PATH)
                    if restored_df.empty:
                        st.error("Không thể khôi phục dữ liệu đã lưu từ lịch sử upload.")
                    else:
                        st.session_state["active_uploaded_df"] = restored_df.copy()
                        st.session_state["active_uploaded_file_name"] = str(source_row["file_name"])
                        st.session_state["active_uploaded_source_page"] = "Lịch sử upload"
                        st.session_state["active_uploaded_version"] = st.session_state.get("active_uploaded_version", 0) + 1
                        st.session_state.pop("active_uploaded_scored_df", None)
                        st.session_state.pop("active_uploaded_scored_version", None)
                        action = "ADMIN_LOAD_DATA_SOURCE" if is_admin else "LOAD_DATA_SOURCE"
                        owner = str(source_row.get("uploaded_by", "")) if hasattr(source_row, "get") else ""
                        log_audit(username, role, action, "data_source", str(source_id), f"Nạp lại nguồn dữ liệu đã lưu; owner={owner}", DB_PATH)
                        st.success("Đã nạp lại dữ liệu từ lịch sử upload.")
                        st.rerun()

            if edited["Xem"].any():
                source_id = int(edited.loc[edited["Xem"] == True, "source_id"].iloc[0])
                source_row = history[history["source_id"] == source_id].iloc[0]
                saved_df = load_saved_source_dataframe(source_id, DB_PATH)
                if saved_df.empty:
                    st.warning("Không tìm thấy dữ liệu đã lưu hoặc file lưu tạm không còn tồn tại.")
                else:
                    viewer_action = "ADMIN_VIEW_SAVED_DATA_SOURCE" if is_admin else "VIEW_SAVED_DATA_SOURCE"
                    owner = str(source_row.get("uploaded_by", "")) if hasattr(source_row, "get") else ""
                    audit_key = f"{viewer_action}_{source_id}_{len(saved_df)}"
                    if st.session_state.get("last_viewed_source_audit") != audit_key:
                        log_audit(username, role, viewer_action, "data_source", str(source_id), f"Xem dữ liệu đã lưu; owner={owner}; rows={len(saved_df)}", DB_PATH)
                        st.session_state["last_viewed_source_audit"] = audit_key
                    st.markdown("**Xem nhanh dữ liệu đã lưu**")
                    st.caption(
                        f"Source ID {source_id} • File: {source_row['file_name']} • "
                        f"Người upload: {source_row.get('uploaded_by', username)} • "
                        f"Hiển thị 100 dòng đầu / {len(saved_df):,} dòng"
                    )
                    st.dataframe(saved_df.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Tải CSV đã lưu",
                        data=saved_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=str(source_row["file_name"] or f"source_{source_id}.csv"),
                        mime="text/csv",
                        use_container_width=True,
                    )


def render_admin(role: str, username: str):
    hero("Xem theo quyền", "Khu vực dành cho Admin để kiểm tra dữ liệu nền, bảng SQLite, ma trận quyền và model registry.", "Admin Console")
    stats = db_overview()
    cols = st.columns(5)
    for i, (table, count) in enumerate(stats.items()):
        with cols[i % 5]:
            metric_card(table, f"{count:,}", "SQLite")
    section_title("Ma trận quyền")
    permission_rows = []
    all_permissions = sorted(set().union(*PERMISSIONS.values()))
    for role_name_, perms in PERMISSIONS.items():
        row = {"Vai trò": ROLE_DISPLAY.get(role_name_, role_name_)}
        for perm in all_permissions:
            row[perm] = "✓" if perm in perms else ""
        permission_rows.append(row)
    st.dataframe(pd.DataFrame(permission_rows), use_container_width=True, hide_index=True)
    section_title("Xem bảng dữ liệu")
    table = st.selectbox("Chọn bảng", ["app_accounts", "users", "login_events", "transactions", "predictions", "audit_logs", "user_feedback", "model_registry", "data_sources", "devices", "addresses"])
    limit = st.slider("Số dòng hiển thị", 10, 300, 50)
    st.dataframe(read_table(table, DB_PATH, limit=limit), use_container_width=True, hide_index=True)
    section_title("Thao tác quản trị")
    if st.button("Ghi audit kiểm tra hệ thống", use_container_width=True):
        log_audit(username, role, "ADMIN_SYSTEM_CHECK", "system", "health", "Admin kiểm tra trạng thái hệ thống", DB_PATH)
        st.success("Đã ghi audit log.")


def render_audit(role: str, username: str):
    if role != "Admin":
        st.error("Bạn không có quyền xem Audit log.")
        set_page("Tổng quan")
        return
    hero("Audit log", "Theo dõi các sự kiện quan trọng như đăng nhập, đăng nhập thất bại, dự đoán giao dịch, batch scoring, feedback và thao tác quản trị hệ thống.", "Security Logging")
    logs = load_audit_cached(DB_PATH)
    if role != "Admin" and not logs.empty and "username" in logs.columns:
        logs = logs[logs["username"].astype(str) == str(username)].copy()
    st.dataframe(logs, use_container_width=True, hide_index=True)
    section_title("Ghi nhận review thủ công")
    entity = st.text_input("Mã user/giao dịch", value="U0003")
    detail = st.text_area("Ghi chú", value="Đã kiểm tra hồ sơ rủi ro và đưa vào danh sách theo dõi.")
    if st.button("Ghi audit review", use_container_width=True):
        log_audit(username, role, "MANUAL_REVIEW", "user", entity, detail, DB_PATH)
        st.cache_data.clear()
        st.success("Đã ghi audit review.")


def render_assistant_fab():
    if not using_uploaded_dataset():
        st.session_state["assistant_open"] = False
        return
    st.markdown('<span id="assistant-fab-anchor"></span>', unsafe_allow_html=True)
    if st.button("Trợ lý", key="assistant_toggle", use_container_width=False):
        st.session_state["assistant_open"] = True
        st.rerun()


def render_assistant_panel(role: str, username: str):
    if not using_uploaded_dataset():
        st.session_state["assistant_open"] = False
        return
    st.markdown('<div class="assistant-panel-block"><span id="assistant-panel-anchor"></span>', unsafe_allow_html=True)
    top_cols = st.columns([5, 1], gap="small")
    with top_cols[0]:
        st.markdown("<div class='assistant-drawer-title'>Trợ lý</div>", unsafe_allow_html=True)
    with top_cols[1]:
        if st.button("×", key="assistant_close", use_container_width=True):
            st.session_state["assistant_open"] = False
            st.rerun()

    st.markdown("<div class='assistant-suggestion-line'>Hỏi về dataset CSV đã upload, rủi ro, drift hoặc cách hệ thống hoạt động.</div>", unsafe_allow_html=True)
    question = st.text_input("Câu hỏi", placeholder="Hỏi về dataset hoặc hệ thống...", key="assistant_text_input", label_visibility="collapsed")
    if st.button("Gửi", key="assistant_send", use_container_width=True):
        if question:
            drift_df = get_drift_report_cached()
            active_upload = get_uploaded_dataset()
            scored_upload = get_scored_uploaded_dataset() if not active_upload.empty else pd.DataFrame()
            answer = answer_trust_question(
                question,
                reference_rows=len(reference_df),
                feature_count=len(feature_columns),
                behavior_df=pd.DataFrame(),
                drift_df=drift_df,
                upload_df=active_upload,
                scored_df=scored_upload,
                upload_report=get_uploaded_report(),
                db_stats={},
                current_page=st.session_state.get("active_page", ""),
            )
            st.session_state.setdefault("chat_history", [])
            st.session_state["chat_history"].append({"role": "user", "content": question})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

    for msg in st.session_state.get("chat_history", [])[-6:]:
        cls = "assistant-msg user" if msg["role"] == "user" else "assistant-msg bot"
        st.markdown(f"<div class='{cls}'>{escape(str(msg['content']))}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



def main():
    # Navigation and logout now use Streamlit buttons only.
    # Avoid query-param driven page changes because they can reset the
    # Streamlit session and send the user back to the login screen.
    if not st.session_state.get("logged_in"):
        render_login()
        return
    preload_after_login()
    active_page, role, username = top_nav(st.session_state.get("role", ""), st.session_state.get("username", ""))
    page_map = {
        "Tổng quan": render_overview,
        "Dự đoán và phân tích": render_prediction,
        "Phân tích hành vi": render_behavior,
        "Hồ sơ người dùng": render_user_profile,
        "Xem theo từng user": render_user_by_user,
        "Case study": render_case_study,
        "Phát hiện bất thường": render_anomaly,
        "Monitoring & Drift": render_monitoring,
        "Batch scoring": render_batch,
        "Nhập dữ liệu": render_data_manager,
        "Trợ lý dữ liệu": render_trust_chat,
        "Feedback": render_feedback,
        "Quản trị hệ thống": render_admin,
        "Audit log": render_audit,
    }
    assistant_available = using_uploaded_dataset()
    if not assistant_available:
        st.session_state["assistant_open"] = False

    if assistant_available and st.session_state.get("assistant_open", False):
        main_col, assistant_col = st.columns([4.1, 1.35], gap="large")
        with main_col:
            page_map[active_page](role, username)
        with assistant_col:
            render_assistant_panel(role, username)
    else:
        page_map[active_page](role, username)
        if assistant_available:
            render_assistant_fab()


if __name__ == "__main__":
    main()
