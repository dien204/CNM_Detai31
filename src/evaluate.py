import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_dir, load_model, save_json

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

MODEL_PATH = os.path.join(MODEL_DIR, "trust_xgb_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_test.csv")

TARGET = "isFraud"
ID_COL = "TransactionID"
THRESHOLD = 0.5
TRAINING_METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")


def load_saved_threshold(default=THRESHOLD):
    if os.path.exists(TRAINING_METRICS_PATH):
        with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return float(metrics.get("threshold", default))
    return default


def load_feature_columns(path=FEATURE_COLUMNS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model_path=MODEL_PATH, test_path=TEST_PATH, threshold=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model tại {model_path}. Hãy chạy: python -m src.train"
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu test tại {test_path}. Hãy chạy: python -m src.preprocess"
        )

    ensure_dir(REPORT_DIR)
    ensure_dir(FIGURE_DIR)

    if threshold is None:
        threshold = load_saved_threshold()

    model = load_model(model_path)
    feature_columns = load_feature_columns()
    df = pd.read_csv(test_path)

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Dữ liệu test thiếu feature so với lúc train: "
            + ", ".join(missing_features[:10])
        )

    X_test = df[feature_columns]
    y_test = df[TARGET]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    save_json(metrics, os.path.join(REPORT_DIR, "evaluation_metrics.json"))

    report_text = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    cm_df.to_csv(os.path.join(REPORT_DIR, "confusion_matrix.csv"), encoding="utf-8-sig")

    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_feature_importance(model, feature_columns)

    print("\nClassification Report:\n")
    print(report_text)
    print("\nEvaluation metrics saved to reports/")
    return metrics


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-fraud", "Fraud"])
    ax.set_yticklabels(["Non-fraud", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "roc_curve.png"), dpi=160)
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "precision_recall_curve.png"), dpi=160)
    plt.close(fig)


def plot_feature_importance(model, feature_columns, top_k=20):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(top_k)

    importance_df.to_csv(
        os.path.join(REPORT_DIR, "feature_importance_top20.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title("Top 20 Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "feature_importance_top20.png"), dpi=160)
    plt.close(fig)


def main():
    print("=" * 60)
    print("EVALUATE MODEL - USER TRUST SCORE")
    print("=" * 60)
    evaluate_model()


if __name__ == "__main__":
    main()
