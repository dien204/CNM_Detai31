import os
import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.utils import reduce_mem_usage, save_json, ensure_dir
except ImportError:
    from utils import reduce_mem_usage, save_json, ensure_dir

PROJECT_NAME = "31. Ứng dụng Dự đoán độ tin cậy người dùng"
COURSE_NAME = "CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

TRAIN_TRANSACTION_PATH = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY_PATH = os.path.join(RAW_DATA_DIR, "train_identity.csv")

TARGET = "isFraud"
ID_COL = "TransactionID"

MISSING_THRESHOLD = 0.90
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.20
RANDOM_STATE = 42
UNKNOWN_CATEGORY_CODE = -1


def assert_raw_data_exists():
    missing = [
        path for path in [TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH]
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Thiếu dữ liệu raw. Hãy tải IEEE-CIS Fraud Detection từ Kaggle và đặt vào data/raw/:\n"
            + "\n".join(missing)
        )


def _split_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test before fitting preprocessing statistics.

    Splitting first avoids data leakage: column dropping, category mappings and
    missing-value imputations are learned from the training split only.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    relative_val_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_df[TARGET],
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def _fit_category_maps(train_df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, List[str]]:
    category_maps: Dict[str, List[str]] = {}
    for col in categorical_cols:
        values = train_df[col].astype("string").fillna("__MISSING__").unique().tolist()
        category_maps[col] = sorted(str(v) for v in values)
    return category_maps


def _transform_categories(df: pd.DataFrame, category_maps: Dict[str, List[str]]) -> pd.DataFrame:
    result = df.copy()
    for col, classes in category_maps.items():
        if col not in result.columns:
            result[col] = UNKNOWN_CATEGORY_CODE
            continue
        mapping = {value: idx for idx, value in enumerate(classes)}
        result[col] = (
            result[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(UNKNOWN_CATEGORY_CODE)
            .astype(np.int32)
        )
    return result


def _fit_fill_values(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    fill_values: Dict[str, float] = {}
    for col in feature_cols:
        numeric = pd.to_numeric(train_df[col], errors="coerce")
        median_value = numeric.median()
        if pd.isna(median_value) or np.isinf(median_value):
            median_value = 0.0
        fill_values[col] = float(median_value)
    return fill_values


def _apply_fill_values(df: pd.DataFrame, fill_values: Dict[str, float]) -> pd.DataFrame:
    result = df.copy()
    for col, value in fill_values.items():
        if col not in result.columns:
            result[col] = value
        result[col] = pd.to_numeric(result[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(value)
    return result


def main():
    print("=" * 60)
    print(PROJECT_NAME)
    print(COURSE_NAME)
    print("=" * 60)

    assert_raw_data_exists()
    ensure_dir(PROCESSED_DATA_DIR)

    print("\n[1] Loading raw data...")
    train_transaction = pd.read_csv(TRAIN_TRANSACTION_PATH)
    train_identity = pd.read_csv(TRAIN_IDENTITY_PATH)

    print("train_transaction:", train_transaction.shape)
    print("train_identity:", train_identity.shape)

    print("\n[2] Reducing memory before merge...")
    train_transaction = reduce_mem_usage(train_transaction)
    train_identity = reduce_mem_usage(train_identity)

    print("\n[3] Merging data...")
    df = train_transaction.merge(train_identity, on=ID_COL, how="left")

    del train_transaction, train_identity
    gc.collect()

    print("Merged dataset:", df.shape)

    print("\n[4] Splitting raw data into train/validation/test...")
    train_df, val_df, test_df = _split_raw(df)
    del df
    gc.collect()
    print("Train raw:", train_df.shape)
    print("Validation raw:", val_df.shape)
    print("Test raw:", test_df.shape)

    print("\n[5] Dropping columns missing > 90% based on TRAIN only...")
    missing_ratio = train_df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    drop_cols = [col for col in drop_cols if col not in [TARGET, ID_COL]]
    train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    val_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    test_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    print("Dropped columns:", len(drop_cols))
    print("After dropping:", train_df.shape, val_df.shape, test_df.shape)

    print("\n[6] Fitting categorical mappings on TRAIN only...")
    categorical_cols = train_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    category_maps = _fit_category_maps(train_df, categorical_cols)
    train_df = _transform_categories(train_df, category_maps)
    val_df = _transform_categories(val_df, category_maps)
    test_df = _transform_categories(test_df, category_maps)
    print("Categorical columns:", len(categorical_cols))

    print("\n[7] Fitting missing-value medians on TRAIN only...")
    feature_cols = [col for col in train_df.columns if col != TARGET]
    fill_values = _fit_fill_values(train_df, feature_cols)
    train_df = _apply_fill_values(train_df, fill_values)
    val_df = _apply_fill_values(val_df, fill_values)
    test_df = _apply_fill_values(test_df, fill_values)
    print("Remaining missing:", int(train_df.isnull().sum().sum() + val_df.isnull().sum().sum() + test_df.isnull().sum().sum()))

    print("\n[8] Aligning columns and reducing memory...")
    ordered_cols = feature_cols + [TARGET]
    train_df = reduce_mem_usage(train_df[ordered_cols])
    val_df = reduce_mem_usage(val_df[ordered_cols])
    test_df = reduce_mem_usage(test_df[ordered_cols])

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\n[9] Saving processed files...")
    paths = {
        "train": os.path.join(PROCESSED_DATA_DIR, "processed_train.csv"),
        "validation": os.path.join(PROCESSED_DATA_DIR, "processed_val.csv"),
        "test": os.path.join(PROCESSED_DATA_DIR, "processed_test.csv"),
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)
    for path in paths.values():
        print("Saved:", path)

    print("\n[10] Saving metadata...")
    metadata = {
        "project_name": PROJECT_NAME,
        "course_name": COURSE_NAME,
        "dataset": "IEEE-CIS Fraud Detection",
        "target": TARGET,
        "id_column": ID_COL,
        "missing_threshold": MISSING_THRESHOLD,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "unknown_category_code": UNKNOWN_CATEGORY_CODE,
        "dropped_columns": drop_cols,
        "categorical_columns": categorical_cols,
        "fill_values": fill_values,
        "train_shape": list(train_df.shape),
        "validation_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "preprocessing_guardrails": [
            "Raw data is split before fitting preprocessing statistics.",
            "Missing-column rule, categorical mappings and medians are fitted on train only.",
            "Validation split is reserved for threshold selection; test split is final evaluation only.",
        ],
        "note": "Optimized for local machine with 8GB RAM. processed_full.csv is not saved.",
    }

    save_json(metadata, os.path.join(PROCESSED_DATA_DIR, "preprocessing_metadata.json"))
    save_json(category_maps, os.path.join(PROCESSED_DATA_DIR, "label_encoders.json"))

    print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
