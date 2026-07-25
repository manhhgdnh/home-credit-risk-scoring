"""
Data loading and protected development/holdout splitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IDENTIFIER_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

NON_PREDICTOR_COLUMNS = (
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
)


def load_application_train(
    data_path: str | Path,
) -> pd.DataFrame:
    """
    Load the labeled application table and validate its required columns.
    """
    data_path = Path(data_path)
    application_train = pd.read_csv(data_path)

    missing_columns = {
        IDENTIFIER_COLUMN,
        TARGET_COLUMN,
    } - set(application_train.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return application_train


def make_protected_split(
    application_train: pd.DataFrame,
    holdout_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the stratified development/holdout split.
    """
    required_columns = {
        IDENTIFIER_COLUMN,
        TARGET_COLUMN,
    }
    missing_columns = (
        required_columns
        - set(application_train.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    development_df, holdout_df = train_test_split(
        application_train,
        test_size=holdout_size,
        random_state=random_state,
        stratify=application_train[TARGET_COLUMN],
    )

    return (
        development_df.copy().sort_index(),
        holdout_df.copy().sort_index(),
    )


def make_model_matrices(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate predictors, binary target and applicant identifiers.
    """
    missing_columns = (
        set(NON_PREDICTOR_COLUMNS)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    predictors = data.drop(
        columns=list(NON_PREDICTOR_COLUMNS)
    ).copy()
    target = data[TARGET_COLUMN].astype("int8").copy()
    identifiers = data[IDENTIFIER_COLUMN].copy()

    return predictors, target, identifiers
