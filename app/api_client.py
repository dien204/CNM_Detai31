import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
USE_BACKEND = os.environ.get("USE_BACKEND", "").lower() in {"1", "true", "yes"}


def backend_enabled() -> bool:
    return bool(BACKEND_URL) or USE_BACKEND


def _url(path: str) -> str:
    base = BACKEND_URL or "http://localhost:8000"
    return f"{base}{path}"


def backend_health(timeout: float = 2.0) -> Optional[dict]:
    if not backend_enabled():
        return None
    try:
        response = requests.get(_url("/health"), timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame rows to JSON-safe records for FastAPI.

    Docker uses the real frontend/backend HTTP path. Uploaded CSV files may
    contain NaN, inf, -inf, numpy scalar values or pandas timestamps; the
    standard JSON encoder can reject those. Pandas' to_json converts non-finite
    values to null and normalizes numpy/pandas scalar types.
    """
    if df is None or df.empty:
        return []
    safe_df = df.copy().replace([np.inf, -np.inf], np.nan)
    return json.loads(safe_df.to_json(orient="records", date_format="iso"))


def predict_dataframe_backend(df: pd.DataFrame, top_n: int = 12, timeout: float = 20.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = _json_safe_records(df)
    response = requests.post(_url("/predict"), json={"records": records, "top_n": top_n}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return pd.DataFrame(payload.get("rows", [])), pd.DataFrame(payload.get("explanation", []))


def batch_predict_dataframe_backend(df: pd.DataFrame, timeout: float = 60.0) -> pd.DataFrame:
    records = _json_safe_records(df)
    response = requests.post(_url("/batch_predict"), json={"records": records}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return pd.DataFrame(payload.get("rows", []))
