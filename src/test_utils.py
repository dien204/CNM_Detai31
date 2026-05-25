import pandas as pd

try:
    from src.inference import aggregate_user_scores, prepare_features_for_inference
    from src.utils import calculate_trust_score, get_risk_level
except ImportError:
    from inference import aggregate_user_scores, prepare_features_for_inference
    from utils import calculate_trust_score, get_risk_level


def test_trust_score_and_risk_level():
    fraud_prob = 0.18
    trust_score = calculate_trust_score(fraud_prob)
    risk = get_risk_level(trust_score)

    assert trust_score == 82.0
    assert risk == "Độ tin cậy cao"


def test_prepare_features_for_inference_fills_missing_columns():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    feature_cols = ["a", "b", "c"]
    X = prepare_features_for_inference(
        df,
        feature_cols,
        fill_values={"c": 5},
        label_encoders={"b": ["x", "y"]},
    )
    assert list(X.columns) == feature_cols
    assert X.loc[0, "b"] == 0
    assert X.loc[0, "c"] == 5


def test_aggregate_user_scores():
    df = pd.DataFrame(
        {
            "UserID": ["u1", "u1", "u2"],
            "Trust_Score": [90.0, 70.0, 30.0],
            "Fraud_Probability": [0.1, 0.3, 0.7],
        }
    )
    user_df, user_col = aggregate_user_scores(df)
    assert user_col == "UserID"
    assert len(user_df) == 2
    assert "User_Trust_Score" in user_df.columns


if __name__ == "__main__":
    fraud_prob = 0.18
    trust_score = calculate_trust_score(fraud_prob)
    risk = get_risk_level(trust_score)
    print("Trust Score:", trust_score)
    print("Risk Level:", risk)
