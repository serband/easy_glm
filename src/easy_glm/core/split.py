"""Train / holdout split conventions shared by the pipeline and the workbench."""

from __future__ import annotations

import polars as pl

TRAIN_FLAG = 1
HOLDOUT_FLAG = 0


def validate_train_test_column(
    data: pl.DataFrame,
    train_test_col: str,
    *,
    require_training_rows: bool = True,
) -> None:
    """Ensure ``train_test_col`` exists and uses 1 = train, 0 = holdout."""
    if train_test_col not in data.columns:
        raise ValueError(
            f"Column '{train_test_col}' not found in data. Add a train/holdout "
            f"indicator column: {TRAIN_FLAG} = train (used for fitting), "
            f"{HOLDOUT_FLAG} = holdout (validation only). Pass its name as "
            "train_test_col."
        )
    flags = data[train_test_col].drop_nulls().unique().sort().to_list()
    invalid = [v for v in flags if v not in (TRAIN_FLAG, HOLDOUT_FLAG)]
    if invalid:
        raise ValueError(
            f"Column '{train_test_col}' must contain only {TRAIN_FLAG} (train) "
            f"and {HOLDOUT_FLAG} (holdout); found: {invalid}"
        )
    if (
        require_training_rows
        and data.filter(pl.col(train_test_col) == TRAIN_FLAG).is_empty()
    ):
        raise ValueError(
            f"No training rows in '{train_test_col}' (expected value {TRAIN_FLAG}). "
            "Check your train/holdout split."
        )
