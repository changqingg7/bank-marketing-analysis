"""Data loading and cleaning helpers for the bank marketing analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_SOURCE_COLUMN = "y"
TARGET_COLUMN = "target_subscription"
VALID_TARGET_VALUES = {"no": 0, "yes": 1}


def load_bank_marketing_data(path: str | Path) -> pd.DataFrame:
    """Load the original semicolon-delimited bank marketing CSV file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")

    return pd.read_csv(data_path, sep=";")


def clean_bank_marketing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a de-duplicated modelling dataset and a cleaning audit summary.

    The source dataset has no null values. Values labelled ``unknown`` are kept
    as explicit categories because they convey missing customer information.
    """
    if TARGET_SOURCE_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_SOURCE_COLUMN}' is missing.")

    cleaned = df.copy()
    initial_rows = len(cleaned)

    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    cleaned[string_columns] = cleaned[string_columns].apply(lambda column: column.str.strip())

    unexpected_target_values = set(cleaned[TARGET_SOURCE_COLUMN].dropna().unique()) - set(
        VALID_TARGET_VALUES
    )
    if unexpected_target_values:
        raise ValueError(f"Unexpected target values: {sorted(unexpected_target_values)}")

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_SOURCE_COLUMN].map(VALID_TARGET_VALUES).astype("int8")

    unknown_cells = int((cleaned[string_columns] == "unknown").sum().sum())
    audit = pd.DataFrame(
        {
            "metric": [
                "rows_before_cleaning",
                "duplicate_rows_removed",
                "rows_after_cleaning",
                "missing_cells_after_cleaning",
                "unknown_category_cells_retained",
            ],
            "value": [
                initial_rows,
                duplicate_rows,
                len(cleaned),
                int(cleaned.isna().sum().sum()),
                unknown_cells,
            ],
        }
    )
    return cleaned, audit


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe as CSV, creating its parent directory when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
