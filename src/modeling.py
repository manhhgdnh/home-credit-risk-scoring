"""
Baseline model factories.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline

from src.features import FeatureSchema
from src.preprocessing import (
    make_logistic_preprocessing_pipeline,
    make_tree_preprocessing_pipeline,
)


def make_dummy_model(
) -> DummyClassifier:
    return DummyClassifier(
        strategy="prior",
    )


def make_logistic_model(
    schema: FeatureSchema,
    class_weight=None,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessing",
                make_logistic_preprocessing_pipeline(
                    schema
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_tree_model(
    schema: FeatureSchema,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessing",
                make_tree_preprocessing_pipeline(
                    schema
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )