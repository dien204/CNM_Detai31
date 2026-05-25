import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.behavior import compute_user_behavior_scores, monitoring_summary, read_audit_logs
from src.database import DB_PATH, init_db, log_audit, seed_demo_database
from src.model_service import TrustModelService
from src.drift_monitoring import drift_from_database
from src.trust_chat import answer_trust_question

app = FastAPI(
    title="User Trust Score API",
    version="1.0.0",
    description="Backend API for user trust scoring, behavior analytics and audit-ready ML inference.",
)

_service: Optional[TrustModelService] = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of transaction rows to score")
    top_n: int = Field(12, ge=1, le=50, description="Number of explanation features for the first row")


class ChatRequest(BaseModel):
    question: str


class AuditRequest(BaseModel):
    actor: str
    role: str
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    detail: Optional[str] = None


def service() -> TrustModelService:
    global _service
    if _service is None:
        _service = TrustModelService()
    return _service


def json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    safe_df = df.copy().replace([np.inf, -np.inf], np.nan)
    return json.loads(safe_df.to_json(orient="records", date_format="iso"))


@app.on_event("startup")
def startup_event():
    init_db(DB_PATH)
    try:
        seed_demo_database(DB_PATH, "data/demo/demo_transactions.csv")
    except Exception:
        pass
    service()


@app.get("/health")
def health():
    metadata = service().metadata()
    return {"status": "ok", "database": DB_PATH, "model": metadata}


@app.get("/metadata")
def metadata():
    return service().metadata()


@app.post("/predict")
def predict(payload: PredictRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must not be empty")
    scored, explanation = service().explain_dataframe(pd.DataFrame(payload.records), top_n=payload.top_n)
    return {
        "rows": json_safe_records(scored),
        "explanation": json_safe_records(explanation),
    }


@app.post("/batch_predict")
def batch_predict(payload: PredictRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must not be empty")
    scored, _ = service().predict_records(payload.records)
    return {"rows": json_safe_records(scored)}


@app.get("/behavior")
def behavior():
    df = compute_user_behavior_scores(DB_PATH)
    return {"rows": json_safe_records(df)}


@app.get("/monitoring")
def monitoring():
    summary = monitoring_summary(DB_PATH)
    serializable = {}
    for key, value in summary.items():
        if isinstance(value, pd.DataFrame):
            serializable[key] = json_safe_records(value)
        else:
            serializable[key] = value
    return serializable


@app.get("/drift")
def drift(limit: int = 5000):
    drift_df = drift_from_database(service().reference_df, DB_PATH, limit=limit)
    return {"rows": json_safe_records(drift_df)}


@app.post("/chat")
def chat(payload: ChatRequest):
    behavior_df = compute_user_behavior_scores(DB_PATH)
    drift_df = drift_from_database(service().reference_df, DB_PATH, limit=5000)
    answer = answer_trust_question(
        payload.question,
        reference_rows=len(service().reference_df),
        feature_count=len(service().feature_columns),
        behavior_df=behavior_df,
        drift_df=drift_df,
    )
    return {"answer": answer}


@app.get("/audit_logs")
def audit_logs(limit: int = 200):
    df = read_audit_logs(DB_PATH, limit=limit)
    return {"rows": json_safe_records(df)}


@app.post("/audit")
def audit(payload: AuditRequest):
    log_audit(payload.actor, payload.role, payload.action, payload.entity_type, payload.entity_id, payload.detail, DB_PATH)
    return {"status": "ok"}
