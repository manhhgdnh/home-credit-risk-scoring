"""
Leakage-safe, model-specific preprocessing pipelines.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.features import (
    ApplicationFeatureEngineer,
    FeatureSchema,
    FeatureSubsetSelector,
)


def make_numerical_preprocessor(
    scale: bool,
) -> Pipeline:
    """
    Median-impute numerical variables and optionally standardize them.
    """
    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps=steps)


def make_logistic_categorical_preprocessor(
) -> Pipeline:
    """
    Impute and one-hot encode logistic-regression categories.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=None,
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def make_tree_categorical_preprocessor(
) -> Pipeline:
    """
    Impute and compactly encode tree-model categories.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def _validate_schema(
    schema: FeatureSchema,
) -> None:
    if not isinstance(schema, FeatureSchema):
        raise TypeError(
            "schema must be a FeatureSchema."
        )

    if schema.unsupported_features:
        raise ValueError(
            "Unsupported feature dtypes: "
            f"{schema.unsupported_features}"
        )


def make_logistic_preprocessor(
    schema: FeatureSchema,
) -> ColumnTransformer:
    """
    Build the logistic-regression preprocessing stage.
    """
    _validate_schema(schema)

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                make_numerical_preprocessor(
                    scale=True,
                ),
                list(
                    schema.logistic_numerical_features
                ),
            ),
            (
                "categorical",
                make_logistic_categorical_preprocessor(),
                list(
                    schema.logistic_categorical_features
                ),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_tree_preprocessor(
    schema: FeatureSchema,
) -> ColumnTransformer:
    _validate_schema(schema)

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                make_tree_categorical_preprocessor(),
                list(
                    schema.tree_categorical_features
                ),
            ),
            (
                "numerical",
                make_numerical_preprocessor(
                    scale=False,
                ),
                list(
                    schema.tree_numerical_features
                ),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def make_logistic_preprocessing_pipeline(
    schema: FeatureSchema,
) -> Pipeline:
    """
    Build feature engineering, selection and logistic preprocessing.
    """
    _validate_schema(schema)

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                ApplicationFeatureEngineer(),
            ),
            (
                "feature_selection",
                FeatureSubsetSelector(
                    selected_features=(
                        schema.logistic_features
                    ),
                ),
            ),
            (
                "preprocessor",
                make_logistic_preprocessor(schema),
            ),
        ]
    )


def make_tree_preprocessing_pipeline(
    schema: FeatureSchema,
) -> Pipeline:
    """
    Build feature engineering, selection and tree preprocessing.
    """
    _validate_schema(schema)

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                ApplicationFeatureEngineer(),
            ),
            (
                "feature_selection",
                FeatureSubsetSelector(
                    selected_features=(
                        schema.tree_features
                    ),
                ),
            ),
            (
                "preprocessor",
                make_tree_preprocessor(schema),
            ),
        ]
    )
