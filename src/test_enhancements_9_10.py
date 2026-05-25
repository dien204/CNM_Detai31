import pandas as pd

from src.drift_monitoring import compute_drift_report
from src.explainability import enhanced_explain_prediction
from src.trust_chat import answer_trust_question


class DummyModel:
    feature_importances_ = [0.7, 0.3]


def test_enhanced_explain_prediction_returns_rows():
    df = pd.DataFrame({"a": [10.0], "b": [1.0]})
    result = enhanced_explain_prediction(DummyModel(), df, top_n=2)
    assert not result.empty
    assert "feature" in result.columns
    assert "method" in result.columns


def test_compute_drift_report_flags_feature_change():
    reference = pd.DataFrame({"TransactionAmt": [100, 110, 120, 130]})
    current = pd.DataFrame({"TransactionAmt": [300, 320, 340, 360]})
    report = compute_drift_report(reference, current)
    assert not report.empty
    assert "drift_level" in report.columns
    assert report.iloc[0]["drift_level"] in {"High", "Medium", "Low"}


def test_trust_chat_answers_dataset_question():
    behavior = pd.DataFrame({"user_id": ["U1"], "long_term_trust_score": [45], "explanation": ["test risk"]})
    answer = answer_trust_question("dataset hiện tại có gì?", 100, 20, behavior)
    assert "Dataset" in answer or "dataset" in answer
