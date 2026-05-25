import os
import tempfile

from src.behavior import detect_anomalies, monitoring_summary, user_profile
from src.database import add_feedback, seed_demo_database


def test_extended_database_features_work():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "demo.db")
        seed_demo_database(db_path=db_path, demo_csv_path="data/demo/demo_transactions.csv", reset=True, n_users=12, n_transactions=120)
        profile = user_profile(db_path, "U0001")
        assert "transactions" in profile
        assert not profile["user"].empty
        add_feedback("U0001", None, "tester", "need_review", "unit test", db_path)
        summary = monitoring_summary(db_path)
        assert summary["metrics"]["prediction_count"] > 0
        assert summary["metrics"]["feedback_count"] > 0
        anomalies = detect_anomalies(db_path)
        assert "user_id" in anomalies.columns or anomalies.empty
