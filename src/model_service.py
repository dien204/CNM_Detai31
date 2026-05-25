import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.behavior import explain_model_prediction
from src.inference import load_json_if_exists, score_transactions
from src.utils import load_model

REAL_MODEL_PATH = "models/trust_xgb_model.pkl"
REAL_FEATURE_COLUMNS_PATH = "models/feature_columns.json"
REAL_TEST_DATA_PATH = "data/processed/processed_test.csv"

DEMO_MODEL_PATH = "models/trust_xgb_demo_model.pkl"
DEMO_FEATURE_COLUMNS_PATH = "models/demo_feature_columns.json"
DEMO_TEST_DATA_PATH = "data/demo/demo_transactions.csv"

PREPROCESSING_METADATA_PATH = "data/processed/preprocessing_metadata.json"
LABEL_ENCODERS_PATH = "data/processed/label_encoders.json"


class TrustModelService:
    """Model service layer for XGBoost inference.

    This class is shared by the FastAPI backend and can also be imported by
    Streamlit when running in local fallback mode. It owns model loading,
    feature alignment, prediction and lightweight explanation.
    """

    def __init__(self):
        self.assets = self._select_assets()
        self.model = load_model(self.assets["model_path"])
        self.feature_columns = load_json_if_exists(self.assets["feature_path"], [])
        self.reference_df = pd.read_csv(self.assets["data_path"])
        metadata = load_json_if_exists(PREPROCESSING_METADATA_PATH, {})
        self.label_encoders = load_json_if_exists(LABEL_ENCODERS_PATH, {})
        self.fill_values = metadata.get("fill_values", {}) if isinstance(metadata, dict) else {}

    def _select_assets(self) -> Dict[str, Any]:
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

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.assets["mode"],
            "model_path": self.assets["model_path"],
            "data_path": self.assets["data_path"],
            "feature_count": len(self.feature_columns),
            "reference_rows": len(self.reference_df),
        }

    def predict_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return score_transactions(
            self.model,
            df,
            self.feature_columns,
            reference_df=self.reference_df,
            fill_values=self.fill_values,
            label_encoders=self.label_encoders,
        )

    def predict_records(self, records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(records)
        return self.predict_dataframe(df)

    def explain_dataframe(self, df: pd.DataFrame, top_n: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scored, X = self.predict_dataframe(df)
        explanation = explain_model_prediction(self.model, X, top_n=top_n)
        return scored, explanation
