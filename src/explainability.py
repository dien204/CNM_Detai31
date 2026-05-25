from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _model_importances(model: Any, n_features: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) == n_features:
            return importances
    return np.ones(n_features, dtype=float) / max(1, n_features)


def _fallback_explanation(model: Any, feature_vector: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    columns = list(feature_vector.columns)
    importances = _model_importances(model, len(columns))
    values = feature_vector.iloc[0].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_values = np.log1p(np.abs(values.to_numpy(dtype=float)))
    explanation_score = importances * (0.2 + abs_values)
    total = float(np.sum(explanation_score)) or 1.0
    relative = (explanation_score / total) * 100.0

    def interpret(feature: str, value: float, importance: float, rel: float) -> str:
        if importance <= 0:
            return "Feature có ảnh hưởng rất thấp trong model hiện tại"
        if rel >= 15:
            level = "ảnh hưởng rất mạnh"
        elif rel >= 7:
            level = "ảnh hưởng đáng chú ý"
        else:
            level = "ảnh hưởng vừa/nhỏ"
        return f"{feature} có {level}; giá trị đầu vào hiện tại là {value:.4g}"

    df = pd.DataFrame(
        {
            "feature": columns,
            "value": values.values,
            "model_importance": importances,
            "explanation_score": explanation_score,
            "relative_impact_pct": np.round(relative, 2),
            "direction": "unknown",
            "method": "Feature importance fallback",
        }
    )
    df["interpretation"] = [
        interpret(str(row.feature), float(row.value), float(row.model_importance), float(row.relative_impact_pct))
        for row in df.itertuples(index=False)
    ]
    return df.sort_values("explanation_score", ascending=False).head(top_n).reset_index(drop=True)


def _try_shap_explanation(model: Any, feature_vector: pd.DataFrame, top_n: int = 12) -> Tuple[pd.DataFrame, bool, str]:
    if os.environ.get("ENABLE_SHAP", "1") != "1":
        return pd.DataFrame(), False, "SHAP đang bị tắt bằng ENABLE_SHAP=0; app dùng fallback feature importance."
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return pd.DataFrame(), False, "Model không hỗ trợ predict/predict_proba cho SHAP, dùng fallback."
    try:
        import shap  # type: ignore
    except Exception as exc:
        return pd.DataFrame(), False, f"SHAP chưa được cài hoặc chưa khả dụng: {exc}"

    try:
        x = feature_vector.head(1).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(x)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[-1], dtype=float)
        else:
            values = np.asarray(raw_values, dtype=float)
        if values.ndim == 2:
            shap_values = values[0]
        elif values.ndim == 1:
            shap_values = values
        else:
            shap_values = values.reshape(-1)[: x.shape[1]]
        if len(shap_values) != x.shape[1]:
            return pd.DataFrame(), False, "Kích thước SHAP không khớp số feature"

        importances = _model_importances(model, x.shape[1])
        abs_shap = np.abs(shap_values)
        total = float(np.sum(abs_shap)) or 1.0
        relative = (abs_shap / total) * 100.0
        values_input = x.iloc[0]

        def direction_text(v: float) -> str:
            if v > 0:
                return "Tăng rủi ro"
            if v < 0:
                return "Giảm rủi ro"
            return "Trung tính"

        def interpret(feature: str, value: float, shap_value: float, rel: float) -> str:
            direction = direction_text(shap_value).lower()
            if rel >= 15:
                level = "rất mạnh"
            elif rel >= 7:
                level = "đáng chú ý"
            else:
                level = "vừa/nhỏ"
            return f"{feature} có tác động {level} và đang {direction}; giá trị đầu vào là {value:.4g}"

        df = pd.DataFrame(
            {
                "feature": list(x.columns),
                "value": values_input.values,
                "model_importance": importances,
                "shap_value": shap_values,
                "abs_shap_value": abs_shap,
                "explanation_score": abs_shap,
                "relative_impact_pct": np.round(relative, 2),
                "direction": [direction_text(v) for v in shap_values],
                "method": "SHAP TreeExplainer",
            }
        )
        df["interpretation"] = [
            interpret(str(row.feature), float(row.value), float(row.shap_value), float(row.relative_impact_pct))
            for row in df.itertuples(index=False)
        ]
        return df.sort_values("abs_shap_value", ascending=False).head(top_n).reset_index(drop=True), True, "ok"
    except Exception as exc:
        return pd.DataFrame(), False, f"Không tính được SHAP cho model hiện tại: {exc}"


def enhanced_explain_prediction(model: Any, feature_vector: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Return a robust explanation table.

    The function tries real SHAP first. If SHAP is not installed or fails on the
    current environment, it falls back to a lightweight feature-importance based
    explanation so the demo never breaks.
    """
    if feature_vector.empty:
        return pd.DataFrame()
    shap_df, ok, message = _try_shap_explanation(model, feature_vector, top_n=top_n)
    if ok and not shap_df.empty:
        return shap_df
    fallback = _fallback_explanation(model, feature_vector, top_n=top_n)
    if not fallback.empty:
        fallback["method_note"] = message
    return fallback


def shap_status() -> Dict[str, Any]:
    try:
        import shap  # type: ignore
        return {"available": True, "version": getattr(shap, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "version": None, "reason": str(exc)}
