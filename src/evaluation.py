"""
Cross-validation scoring and result utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_brier_score": "neg_brier_score",
}


def extract_fold_scores(
    model_name: str,
    cv_results: dict,
) -> pd.DataFrame:
    n_folds = len(
        cv_results["test_roc_auc"]
    )

    return pd.DataFrame(
        {
            "model": model_name,
            "fold": np.arange(1, n_folds + 1),
            "roc_auc": cv_results[
                "test_roc_auc"
            ],
            "average_precision": cv_results[
                "test_average_precision"
            ],
            "brier_score": -cv_results[
                "test_neg_brier_score"
            ],
            "fit_time_seconds": cv_results[
                "fit_time"
            ],
            "score_time_seconds": cv_results[
                "score_time"
            ],
        }
    )


def summarize_fold_scores(
    fold_scores: pd.DataFrame,
) -> pd.DataFrame:
    return (
        fold_scores
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            average_precision_mean=(
                "average_precision",
                "mean",
            ),
            average_precision_std=(
                "average_precision",
                "std",
            ),
            brier_score_mean=(
                "brier_score",
                "mean",
            ),
            fit_time_mean=(
                "fit_time_seconds",
                "mean",
            ),
        )
        .sort_values(
            "roc_auc_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )