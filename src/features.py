"""
Feature schema and deterministic application-level feature engineering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


ENGINEERED_FEATURES = (
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "DAYS_EMPLOYED_ANOMALOUS",
    "REGISTRATION_YEARS",
    "ID_PUBLISH_YEARS",
    "EMPLOYMENT_START_AGE",
    "EXT_SOURCE_AVAILABLE_COUNT",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_TERM_PROXY",
    "INCOME_PER_PERSON",
    "EMPLOYED_AGE_RATIO",
)

RATIO_FEATURES = (
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_TERM_PROXY",
    "INCOME_PER_PERSON",
    "EMPLOYED_AGE_RATIO",
)

COMMON_EXCLUDED_FEATURES = (
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CREDIT_TERM_PROXY",
)

LOGISTIC_SPECIFIC_EXCLUDED_FEATURES = (
    "EMPLOYMENT_START_AGE",
)

TREE_SPECIFIC_EXCLUDED_FEATURES: tuple[str, ...] = ()

LOGISTIC_EXCLUDED_FEATURES = (
    *COMMON_EXCLUDED_FEATURES,
    *LOGISTIC_SPECIFIC_EXCLUDED_FEATURES,
)

TREE_EXCLUDED_FEATURES = (
    *COMMON_EXCLUDED_FEATURES,
    *TREE_SPECIFIC_EXCLUDED_FEATURES,
)


@dataclass(frozen=True)
class FeatureSchema:
    """
    Ordered raw and model-specific feature definitions.
    """

    raw_predictor_features: tuple[str, ...]
    raw_numerical_features: tuple[str, ...]
    raw_categorical_features: tuple[str, ...]
    unsupported_features: tuple[str, ...]

    logistic_features: tuple[str, ...]
    logistic_numerical_features: tuple[str, ...]
    logistic_categorical_features: tuple[str, ...]

    tree_features: tuple[str, ...]
    tree_numerical_features: tuple[str, ...]
    tree_categorical_features: tuple[str, ...]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide numerical Series while mapping undefined results to NaN.
    """
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype("float64")
    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype("float64")

    valid_denominator = (
        denominator.notna()
        & denominator.gt(0)
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )
    result.loc[valid_denominator] = (
        numerator.loc[valid_denominator]
        / denominator.loc[valid_denominator]
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


class ApplicationFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create deterministic application-level features.

    No target-dependent or data-dependent statistics are learned.
    """

    REQUIRED_COLUMNS = (
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS",
    )

    EXT_SOURCE_COLUMNS = (
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    )

    def __init__(
        self,
        days_per_year: float = 365.25,
        employed_sentinel: int = 365243,
    ):
        self.days_per_year = days_per_year
        self.employed_sentinel = employed_sentinel

    @staticmethod
    def _validate_dataframe_type(
        X: pd.DataFrame,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ApplicationFeatureEngineer expects "
                "a pandas DataFrame."
            )

    def _validate_dataframe(
        self,
        X: pd.DataFrame,
    ) -> None:
        self._validate_dataframe_type(X)

        missing_columns = (
            set(self.REQUIRED_COLUMNS)
            - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        non_numerical_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if not pd.api.types.is_numeric_dtype(
                X[column]
            )
        ]

        if non_numerical_columns:
            raise TypeError(
                "The following required columns must be "
                f"numerical: {non_numerical_columns}"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        self._validate_dataframe(X)

        feature_collisions = (
            set(ENGINEERED_FEATURES)
            & set(X.columns)
        )

        if feature_collisions:
            raise ValueError(
                "Engineered-feature names already exist "
                "in the input: "
                f"{sorted(feature_collisions)}"
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = len(
            self.feature_names_in_
        )
        self.feature_names_out_ = np.asarray(
            [
                *self.feature_names_in_,
                *ENGINEERED_FEATURES,
            ],
            dtype=object,
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_",
                "feature_names_out_",
            ],
        )
        self._validate_dataframe(X)

        if not X.columns.equals(
            pd.Index(self.feature_names_in_)
        ):
            raise ValueError(
                "Input columns or their order differ "
                "from the fitted schema."
            )

        transformed = X.copy()

        days_employed = transformed[
            "DAYS_EMPLOYED"
        ].astype("float64")
        employed_anomalous = days_employed.eq(
            self.employed_sentinel
        )

        transformed[
            "DAYS_EMPLOYED_ANOMALOUS"
        ] = employed_anomalous.astype("int8")
        transformed["DAYS_EMPLOYED"] = (
            days_employed.mask(employed_anomalous)
        )

        transformed["AGE_YEARS"] = (
            -transformed["DAYS_BIRTH"]
            / self.days_per_year
        )
        transformed["EMPLOYMENT_YEARS"] = (
            -transformed["DAYS_EMPLOYED"]
            / self.days_per_year
        )
        transformed["REGISTRATION_YEARS"] = (
            -transformed["DAYS_REGISTRATION"]
            / self.days_per_year
        )
        transformed["ID_PUBLISH_YEARS"] = (
            -transformed["DAYS_ID_PUBLISH"]
            / self.days_per_year
        )
        transformed["EMPLOYMENT_START_AGE"] = (
            transformed["AGE_YEARS"]
            - transformed["EMPLOYMENT_YEARS"]
        )

        transformed[
            "EXT_SOURCE_AVAILABLE_COUNT"
        ] = (
            transformed[
                list(self.EXT_SOURCE_COLUMNS)
            ]
            .notna()
            .sum(axis=1)
            .astype("int8")
        )

        transformed["CREDIT_INCOME_RATIO"] = (
            safe_divide(
                transformed["AMT_CREDIT"],
                transformed["AMT_INCOME_TOTAL"],
            )
        )
        transformed["ANNUITY_INCOME_RATIO"] = (
            safe_divide(
                transformed["AMT_ANNUITY"],
                transformed["AMT_INCOME_TOTAL"],
            )
        )
        transformed["CREDIT_GOODS_RATIO"] = (
            safe_divide(
                transformed["AMT_CREDIT"],
                transformed["AMT_GOODS_PRICE"],
            )
        )
        transformed["ANNUITY_CREDIT_RATIO"] = (
            safe_divide(
                transformed["AMT_ANNUITY"],
                transformed["AMT_CREDIT"],
            )
        )
        transformed["CREDIT_TERM_PROXY"] = (
            safe_divide(
                transformed["AMT_CREDIT"],
                transformed["AMT_ANNUITY"],
            )
        )
        transformed["INCOME_PER_PERSON"] = (
            safe_divide(
                transformed["AMT_INCOME_TOTAL"],
                transformed["CNT_FAM_MEMBERS"],
            )
        )
        transformed["EMPLOYED_AGE_RATIO"] = (
            safe_divide(
                transformed["EMPLOYMENT_YEARS"],
                transformed["AGE_YEARS"],
            )
        )

        return transformed.loc[
            :,
            self.feature_names_out_,
        ]

    def get_feature_names_out(
        self,
        input_features=None,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=["feature_names_out_"],
        )

        if input_features is not None:
            input_features = np.asarray(
                input_features,
                dtype=object,
            )

            if not np.array_equal(
                input_features,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "input_features does not match "
                    "feature_names_in_."
                )

        return self.feature_names_out_.copy()


class FeatureSubsetSelector(
    BaseEstimator,
    TransformerMixin,
):
    """
    Select a fixed ordered subset of DataFrame columns.
    """

    def __init__(
        self,
        selected_features: Sequence[str],
    ):
        self.selected_features = selected_features

    @staticmethod
    def _validate_dataframe(
        X: pd.DataFrame,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureSubsetSelector expects "
                "a pandas DataFrame."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        self._validate_dataframe(X)
        selected_features = tuple(
            self.selected_features
        )

        if not selected_features:
            raise ValueError(
                "selected_features cannot be empty."
            )

        if (
            len(selected_features)
            != len(set(selected_features))
        ):
            raise ValueError(
                "selected_features contains duplicates."
            )

        missing_features = (
            set(selected_features)
            - set(X.columns)
        )

        if missing_features:
            raise ValueError(
                "Selected features are missing from "
                f"the input: {sorted(missing_features)}"
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = len(
            self.feature_names_in_
        )
        self.selected_features_ = np.asarray(
            selected_features,
            dtype=object,
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_",
                "selected_features_",
            ],
        )
        self._validate_dataframe(X)

        if not X.columns.equals(
            pd.Index(self.feature_names_in_)
        ):
            raise ValueError(
                "Input columns or their order differ "
                "from the fitted schema."
            )

        return X.loc[
            :,
            self.selected_features_,
        ].copy()

    def get_feature_names_out(
        self,
        input_features=None,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=["selected_features_"],
        )

        if input_features is not None:
            input_features = np.asarray(
                input_features,
                dtype=object,
            )

            if not np.array_equal(
                input_features,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "input_features does not match "
                    "feature_names_in_."
                )

        return self.selected_features_.copy()


def build_feature_schema(
    X: pd.DataFrame,
) -> FeatureSchema:
    """
    Build deterministic model schemas from raw column names and dtypes.

    Feature values, the target and the holdout set are not used.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "build_feature_schema expects "
            "a pandas DataFrame."
        )

    if not X.columns.is_unique:
        raise ValueError(
            "Input feature names must be unique."
        )

    feature_collisions = (
        set(ENGINEERED_FEATURES)
        & set(X.columns)
    )

    if feature_collisions:
        raise ValueError(
            "Engineered-feature names already exist "
            "in the input: "
            f"{sorted(feature_collisions)}"
        )

    raw_predictor_features = tuple(X.columns)
    raw_categorical_features = tuple(
        feature
        for feature in X.columns
        if (
            pd.api.types.is_object_dtype(X[feature])
            or pd.api.types.is_string_dtype(X[feature])
            or isinstance(
                X[feature].dtype,
                pd.CategoricalDtype,
            )
        )
    )
    raw_numerical_features = tuple(
        feature
        for feature in X.columns
        if pd.api.types.is_numeric_dtype(
            X[feature]
        )
    )

    classified_feature_set = set(
        raw_numerical_features
        + raw_categorical_features
    )
    unsupported_features = tuple(
        feature
        for feature in X.columns
        if feature not in classified_feature_set
    )

    engineered_feature_order = (
        raw_predictor_features
        + ENGINEERED_FEATURES
    )
    logistic_excluded_set = set(
        LOGISTIC_EXCLUDED_FEATURES
    )
    tree_excluded_set = set(
        TREE_EXCLUDED_FEATURES
    )

    logistic_features = tuple(
        feature
        for feature in engineered_feature_order
        if feature not in logistic_excluded_set
    )
    tree_features = tuple(
        feature
        for feature in engineered_feature_order
        if feature not in tree_excluded_set
    )

    categorical_feature_set = set(
        raw_categorical_features
    )
    numerical_feature_set = (
        set(raw_numerical_features)
        | set(ENGINEERED_FEATURES)
    )

    logistic_categorical_features = tuple(
        feature
        for feature in logistic_features
        if feature in categorical_feature_set
    )
    logistic_numerical_features = tuple(
        feature
        for feature in logistic_features
        if feature in numerical_feature_set
    )
    tree_categorical_features = tuple(
        feature
        for feature in tree_features
        if feature in categorical_feature_set
    )
    tree_numerical_features = tuple(
        feature
        for feature in tree_features
        if feature in numerical_feature_set
    )

    return FeatureSchema(
        raw_predictor_features=(
            raw_predictor_features
        ),
        raw_numerical_features=(
            raw_numerical_features
        ),
        raw_categorical_features=(
            raw_categorical_features
        ),
        unsupported_features=(
            unsupported_features
        ),
        logistic_features=logistic_features,
        logistic_numerical_features=(
            logistic_numerical_features
        ),
        logistic_categorical_features=(
            logistic_categorical_features
        ),
        tree_features=tree_features,
        tree_numerical_features=(
            tree_numerical_features
        ),
        tree_categorical_features=(
            tree_categorical_features
        ),
    )


def build_raw_feature_schema(
    df: pd.DataFrame,
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """
    Build an audit table describing raw predictor columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "build_raw_feature_schema expects "
            "a pandas DataFrame."
        )

    categorical_feature_set = set(
        categorical_features
    )
    schema_records = []

    for feature in df.columns:
        series = df[feature]
        feature_kind = (
            "categorical"
            if feature in categorical_feature_set
            else "numerical"
        )
        value_frequencies = series.value_counts(
            dropna=False,
            normalize=True,
        )
        most_frequent_share_pct = (
            value_frequencies.iloc[0] * 100
            if not value_frequencies.empty
            else np.nan
        )

        schema_records.append(
            {
                "feature": feature,
                "feature_kind": feature_kind,
                "dtype": str(series.dtype),
                "missing_count": (
                    series.isna().sum()
                ),
                "missing_pct": (
                    series.isna().mean() * 100
                ),
                "n_unique_observed": (
                    series.nunique(dropna=True)
                ),
                "n_unique_with_missing": (
                    series.nunique(dropna=False)
                ),
                "most_frequent_share_pct": (
                    most_frequent_share_pct
                ),
                "is_constant": (
                    series.nunique(dropna=True)
                    <= 1
                ),
                "is_near_constant": (
                    most_frequent_share_pct
                    >= 99.5
                ),
            }
        )

    return pd.DataFrame(schema_records)
