import argparse
import json
import os

import pandas as pd

from src.inference import aggregate_user_scores, load_json_if_exists, score_transactions
from src.utils import load_model

MODEL_PATH = "models/trust_xgb_model.pkl"
DEMO_MODEL_PATH = "models/trust_xgb_demo_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"
DEMO_FEATURE_COLUMNS_PATH = "models/demo_feature_columns.json"
TEST_PATH = "data/processed/processed_test.csv"
DEMO_TEST_PATH = "data/demo/demo_transactions.csv"
OUTPUT_PATH = "data/predictions.csv"
USER_OUTPUT_PATH = "data/user_trust_scores.csv"
PREPROCESSING_METADATA_PATH = "data/processed/preprocessing_metadata.json"
LABEL_ENCODERS_PATH = "data/processed/label_encoders.json"


def load_feature_columns(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_default_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_PATH):
        return MODEL_PATH, FEATURE_COLUMNS_PATH, TEST_PATH
    return DEMO_MODEL_PATH, DEMO_FEATURE_COLUMNS_PATH, DEMO_TEST_PATH


def main():
    parser = argparse.ArgumentParser(description="Predict user trust score for transactions.")
    parser.add_argument("--input", default=None, help="CSV đầu vào. Nếu bỏ trống sẽ dùng processed_test hoặc demo data.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="File CSV lưu kết quả giao dịch.")
    parser.add_argument("--user-output", default=USER_OUTPUT_PATH, help="File CSV lưu điểm tổng hợp theo người dùng.")
    args = parser.parse_args()

    model_path, feature_path, default_input = select_default_assets()
    input_path = args.input or default_input

    if not os.path.exists(model_path):
        raise FileNotFoundError("Không tìm thấy model. Hãy chạy python -m src.train hoặc scripts/create_demo_assets.py")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_path}")

    model = load_model(model_path)
    feature_columns = load_feature_columns(feature_path)
    df = pd.read_csv(input_path)

    metadata = load_json_if_exists(PREPROCESSING_METADATA_PATH, {})
    label_encoders = load_json_if_exists(LABEL_ENCODERS_PATH, {})
    fill_values = metadata.get("fill_values", {}) if isinstance(metadata, dict) else {}

    scored_df, _ = score_transactions(
        model,
        df,
        feature_columns,
        reference_df=df,
        fill_values=fill_values,
        label_encoders=label_encoders,
    )
    user_df, user_col = aggregate_user_scores(scored_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scored_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    user_df.to_csv(args.user_output, index=False, encoding="utf-8-sig")

    print("PREDICTION COMPLETED")
    print(f"Transaction predictions: {args.output}")
    print(f"User trust scores: {args.user_output}")
    print(f"User identifier used: {user_col}")
    print(scored_df[[c for c in ["TransactionID", "Fraud_Probability", "Trust_Score", "Risk_Level"] if c in scored_df.columns]].head(10))


if __name__ == "__main__":
    main()
