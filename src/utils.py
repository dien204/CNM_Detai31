import os
import json
import joblib
import numpy as np
import pandas as pd


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def reduce_mem_usage(df):
    for col in df.columns:
        # Chỉ xử lý cột numeric, bỏ qua object/string/category
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()

        if pd.api.types.is_integer_dtype(col_type):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)

    return df


def save_json(data, path):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"✅ JSON saved: {path}")


def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    print(f"✅ Pickle saved: {path}")


def load_pickle(path):
    obj = joblib.load(path)
    print(f"✅ Pickle loaded: {path}")
    return obj


def save_model(model, path):
    save_pickle(model, path)


def load_model(path):
    return load_pickle(path)


def calculate_trust_score(fraud_probability):
    trust_score = (1 - fraud_probability) * 100
    return round(trust_score, 2)


def get_risk_level(trust_score):
    if trust_score >= 80:
        return "Độ tin cậy cao"
    elif trust_score >= 50:
        return "Độ tin cậy trung bình"
    else:
        return "Độ tin cậy thấp"