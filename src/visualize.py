import os

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir

PROCESSED_TRAIN_PATH = "data/processed/processed_train.csv"
FIGURE_DIR = "reports/figures"
TARGET = "isFraud"


def main():
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        raise FileNotFoundError(
            "Không tìm thấy data/processed/processed_train.csv. Hãy chạy python -m src.preprocess trước."
        )

    ensure_dir(FIGURE_DIR)
    df = pd.read_csv(PROCESSED_TRAIN_PATH)

    target_counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Non-fraud", "Fraud"], target_counts.values)
    ax.set_title("Phân bố nhãn isFraud")
    ax.set_ylabel("Số lượng")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "target_distribution.png"), dpi=160)
    plt.close(fig)

    if "TransactionAmt" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["TransactionAmt"].clip(upper=df["TransactionAmt"].quantile(0.99)), bins=40)
        ax.set_title("Phân phối TransactionAmt")
        ax.set_xlabel("TransactionAmt")
        ax.set_ylabel("Số lượng")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "transaction_amount_distribution.png"), dpi=160)
        plt.close(fig)

    print("EDA figures saved to reports/figures")


if __name__ == "__main__":
    main()
