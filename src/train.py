import os
import gc

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

from src.utils import ensure_dir, save_model, save_json

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_train.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_test.csv")
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_val.csv")

TARGET = "isFraud"
ID_COL = "TransactionID"
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(f1_values.argmax())
    return float(thresholds[best_index])


def main():
    print("=" * 60)
    print("TRAIN MODEL - 31. Ứng dụng Dự đoán độ tin cậy người dùng")
    print("=" * 60)

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            "Chưa tìm thấy dữ liệu đã xử lý. Hãy chạy: python -m src.preprocess"
        )

    print("\n[1] Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    val_df = pd.read_csv(VAL_PATH) if os.path.exists(VAL_PATH) else None

    print("Train:", train_df.shape)
    if val_df is not None:
        print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    drop_cols = [TARGET]
    if ID_COL in train_df.columns:
        drop_cols.append(ID_COL)

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET]

    if val_df is not None:
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[TARGET]
    else:
        X_val, y_val = X_test, y_test

    del train_df, test_df, val_df
    gc.collect()

    print("\n[2] Class imbalance...")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("Non-fraud:", neg)
    print("Fraud:", pos)
    print("scale_pos_weight:", round(scale_pos_weight, 2))

    print("\n[3] Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    model.fit(X_train, y_train)

    print("\n[4] Evaluating model...")

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, val_prob)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    val_pr_auc = average_precision_score(y_val, val_prob)
    val_roc_auc = roc_auc_score(y_val, val_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    report_text = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report_text)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Best threshold:", best_threshold)

    print("\n[5] Saving model, metadata and reports...")

    ensure_dir(MODEL_DIR)
    ensure_dir(REPORT_DIR)

    model_path = os.path.join(MODEL_DIR, "trust_xgb_model.pkl")
    save_model(model, model_path)

    feature_columns = X_train.columns.tolist()

    save_json(feature_columns, os.path.join(MODEL_DIR, "feature_columns.json"))

    metrics = {
        "project": "31. Ứng dụng Dự đoán độ tin cậy người dùng",
        "model": "XGBoost",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "validation_roc_auc": float(val_roc_auc),
        "validation_pr_auc": float(val_pr_auc),
        "confusion_matrix": cm.tolist(),
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": float(best_threshold),
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold_selection": "Best F1-score on validation split; test split is used only for final evaluation.",
        "trust_score_formula": "Trust Score = (1 - fraud_probability) * 100",
        "note": "Threshold is selected on processed_val.csv when available; reported metrics are computed on processed_test.csv."
    }

    save_json(metrics, os.path.join(MODEL_DIR, "training_metrics.json"))
    save_json(metrics, os.path.join(REPORT_DIR, "evaluation_metrics.json"))

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(os.path.join(REPORT_DIR, "confusion_matrix.csv"), encoding="utf-8-sig")

    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("Gợi ý: chạy thêm `python -m src.evaluate` để xuất biểu đồ vào reports/figures.")


if __name__ == "__main__":
    main()
