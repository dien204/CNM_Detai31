import pandas as pd

from src.data_validation import validate_numeric_ranges, validate_required_columns


def test_validate_required_columns_reports_missing_column():
    ok, issues = validate_required_columns(pd.DataFrame({"a": [1]}), ["a", "b"])
    assert not ok
    assert issues[0]["column"] == "b"


def test_validate_numeric_ranges_reports_invalid_values():
    ok, issues = validate_numeric_ranges(pd.DataFrame({"amount": [10, -1, "x"]}), {"amount": (0, 100)})
    assert not ok
    assert "2 value" in issues[0]["issue"]
