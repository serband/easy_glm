"""Bounded three-pair CPU spike; run from the repository root with PYTHONPATH=src."""

from __future__ import annotations

import json
import resource
import time

import numpy as np
import polars as pl

from easy_glm.engine.models import FromToRow, ModelMetadata, VariableConfig
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.pair_distillation import (
    distill_pair_cells,
    fit_catboost_pair_raw,
)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (term - y + mu)))


def main() -> None:
    rng = np.random.default_rng(42)
    n = 2800
    x = rng.normal(size=(n, 3))
    external_offset = rng.normal(0, 0.45, n)
    frame = pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "offset": external_offset}
    )
    main_table = VariableConfig(
        type="numeric",
        table=[FromToRow(None, 0.0, 0.78), FromToRow(0.0, None, 1.24)],
    )
    RateModel._precompute_variables({"x0": main_table})
    main_model = RateModel(
        base_rate=2.0,
        variables={"x0": main_table},
        metadata=ModelMetadata(offset_col="offset", link="log"),
    )
    main_baseline = main_model.predict(frame, exposure_col=None)
    expected_main = 2.0 * np.where(x[:, 0] < 0, 0.78, 1.24) * np.exp(external_offset)
    offset_once_error = float(np.max(np.abs(main_baseline / expected_main - 1)))
    signal = (
        0.4 * np.sin(1.8 * x[:, 0]) * np.tanh(x[:, 1])
        - 0.28 * (x[:, 1] > 0) * (x[:, 2] > 0)
        + 0.22 * np.sin(x[:, 0] + x[:, 2])
    )
    y = rng.poisson(main_baseline * np.exp(signal)).astype(float)
    train = np.arange(n) % 5 != 0
    valid = ~train
    weights = rng.uniform(0.7, 1.4, n)
    # Fit each axis from training rows only.  Edges are shared by all stages;
    # the main table above remains frozen and is distinct from pair axes.
    edges = [np.quantile(x[train, j], [0.2, 0.4, 0.6, 0.8]) for j in range(3)]
    codes = [
        np.searchsorted(edge, x[:, j], side="right") for j, edge in enumerate(edges)
    ]
    baseline = main_baseline.copy()
    stages = [(0, 1), (1, 2), (0, 2)]
    records = []
    start = time.perf_counter()
    for stage, (a, b) in enumerate(stages, 1):
        before = baseline.copy()
        teacher = fit_catboost_pair_raw(
            x[train][:, [a, b]],
            y[train],
            before[train],
            sample_weight=weights[train],
            iterations=70,
            depth=3,
            learning_rate=0.07,
            l2_leaf_reg=3.0,
            thread_count=2,
            seed=stage,
        )
        teacher_train = teacher.predict_mean(x[train][:, [a, b]], before[train])
        cell_ids = codes[a] * 5 + codes[b]
        table = distill_pair_cells(
            cell_ids[train],
            before[train],
            teacher_train,
            sample_weight=weights[train],
            n_cells=25,
            min_weight_share=0.005,
        )
        baseline *= table.relativities[cell_ids]
        records.append(
            {
                "stage": stage,
                "pair": [a, b],
                "prefix_validation_deviance": poisson_deviance(y[valid], before[valid]),
                "table_validation_deviance": poisson_deviance(
                    y[valid], baseline[valid]
                ),
                "teacher_training_deviance": poisson_deviance(y[train], teacher_train),
                "table_training_deviance": poisson_deviance(y[train], baseline[train]),
                "unsupported_cells": int(
                    sum(reason is not None for reason in table.fallback_reason)
                ),
                "mean_multiplier": float(np.mean(table.relativities[cell_ids[train]])),
            }
        )
    print(
        json.dumps(
            {
                "rows": n,
                "training_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "stages": records,
                "wall_seconds": round(time.perf_counter() - start, 3),
                "peak_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1
                ),
                "offset_once_error": offset_once_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
