from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class ValidationIssue:
    column: str
    issue: str
    severity: str = "error"


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> Tuple[bool, List[Dict[str, str]]]:
    """Validate that a dataframe contains the required schema columns."""
    issues: List[ValidationIssue] = []
    for col in required_columns:
        if col not in df.columns:
            issues.append(ValidationIssue(column=str(col), issue="missing_required_column"))
    return len(issues) == 0, [asdict(issue) for issue in issues]


def validate_numeric_ranges(df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]]) -> Tuple[bool, List[Dict[str, str]]]:
    """Validate simple numeric min/max ranges for imported data."""
    issues: List[ValidationIssue] = []
    for col, (min_value, max_value) in ranges.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = numeric.isna() | (numeric < min_value) | (numeric > max_value)
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            issues.append(
                ValidationIssue(
                    column=col,
                    issue=f"{invalid_count} value(s) outside range [{min_value}, {max_value}] or not numeric",
                )
            )
    return len(issues) == 0, [asdict(issue) for issue in issues]
