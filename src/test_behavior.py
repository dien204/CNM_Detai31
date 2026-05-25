import os
import tempfile

from src.behavior import compute_user_behavior_scores
from src.database import seed_demo_database


def test_behavior_scores_are_created():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "demo.db")
        seed_demo_database(db_path=db_path, demo_csv_path="data/demo/demo_transactions.csv", reset=True, n_users=5)
        df = compute_user_behavior_scores(db_path)
        assert not df.empty
        assert "long_term_trust_score" in df.columns
        assert df["long_term_trust_score"].between(0, 100).all()
