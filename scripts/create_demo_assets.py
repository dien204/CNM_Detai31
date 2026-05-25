"""Create lightweight demo assets for Streamlit without requiring the full Kaggle dataset.

The generated model/data are only for UI demonstration and smoke testing. For the
official report metrics, train the model on IEEE-CIS Fraud Detection using:

    python -m src.preprocess
    python -m src.train
    python -m src.evaluate
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import save_json, save_model

FEATURE_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
DEMO_FEATURE_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "models", "demo_feature_columns.json")
DEMO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trust_xgb_demo_model.pkl")
DEMO_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "demo", "demo_transactions.csv")
DEMO_METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "demo_training_metrics.json")

RANDOM_STATE = 42
N_SAMPLES = 600


def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_demo_dataframe(feature_columns):
    rng = np.random.default_rng(RANDOM_STATE)
    values = {}

    for col in feature_columns:
        if col == "TransactionDT":
            values[col] = rng.integers(10000, 9000000, N_SAMPLES)
        elif col == "TransactionAmt":
            values[col] = np.round(rng.gamma(shape=2.0, scale=80.0, size=N_SAMPLES), 2)
        elif col.startswith("C") and col[1:].isdigit():
            values[col] = rng.poisson(lam=2.0, size=N_SAMPLES)
        elif col.startswith("D") and col[1:].isdigit():
            values[col] = rng.normal(loc=120, scale=60, size=N_SAMPLES).clip(0)
        elif col.startswith("V") and col[1:].isdigit():
            values[col] = rng.normal(loc=0, scale=1, size=N_SAMPLES)
        elif col.startswith("id_"):
            values[col] = rng.integers(0, 12, N_SAMPLES)
        elif col in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]:
            values[col] = rng.integers(0, 6, N_SAMPLES)
        else:
            values[col] = rng.normal(loc=0, scale=1, size=N_SAMPLES)

    df = pd.DataFrame(values)

    amount = df.get("TransactionAmt", pd.Series(0, index=df.index))
    c1 = df.get("C1", pd.Series(0, index=df.index))
    c13 = df.get("C13", pd.Series(0, index=df.index))
    v258 = df.get("V258", pd.Series(0, index=df.index))
    card6 = df.get("card6", pd.Series(0, index=df.index))
    device_type = df.get("DeviceType", pd.Series(0, index=df.index))

    raw_score = (
        0.009 * amount
        + 0.35 * c1
        + 0.05 * c13
        + 0.8 * (card6 == 2).astype(int)
        + 0.6 * (device_type == 1).astype(int)
        + 0.25 * v258
        - 3.2
    )
    fraud_prob = 1 / (1 + np.exp(-raw_score))
    y = rng.binomial(1, fraud_prob.clip(0.02, 0.95))

    df.insert(0, "TransactionID", np.arange(3300000, 3300000 + N_SAMPLES))
    df.insert(1, "isFraud", y.astype(int))
    return df


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "demo"), exist_ok=True)

    feature_columns = load_feature_columns()
    demo_df = make_demo_dataframe(feature_columns)

    train_df, test_df = train_test_split(
        demo_df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=demo_df["isFraud"],
    )

    X_train = train_df[feature_columns]
    y_train = train_df["isFraud"]
    X_test = test_df[feature_columns]
    y_test = test_df["isFraud"]

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "project": "31. Ứng dụng Dự đoán độ tin cậy người dùng",
        "model": "XGBoost demo model",
        "demo_only": True,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": 0.5,
        "trust_score_formula": "Trust Score = (1 - fraud_probability) * 100",
        "note": "Demo assets are generated from synthetic data and are only for Streamlit smoke testing.",
    }

    demo_df.to_csv(DEMO_DATA_PATH, index=False, encoding="utf-8-sig")
    save_model(model, DEMO_MODEL_PATH)
    save_json(feature_columns, DEMO_FEATURE_COLUMNS_PATH)
    save_json(metrics, DEMO_METRICS_PATH)

    print("Demo data saved:", DEMO_DATA_PATH)
    print("Demo model saved:", DEMO_MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
