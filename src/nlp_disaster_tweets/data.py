from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"text", "target"}


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the dataset before splitting."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["text", "target"]).copy()

    df["text"] = df["text"].astype(str)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[Any, Any, Any, Any]:
    """Split the dataset into stratified train and validation sets."""

    X = df["text"]
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
