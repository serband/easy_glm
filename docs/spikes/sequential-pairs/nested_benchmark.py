"""Measure the bounded nested-CV trainer with the proposed default search."""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import polars as pl

from easy_glm.workflow import pair_stages
from easy_glm.workflow.project import PairStageConfig, Project
from easy_glm.workflow.run import run_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--stages", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    rng = np.random.default_rng(241)
    n = args.rows
    train_rows = int(0.8 * n)
    values = rng.normal(size=(n, 4))
    main, a, b, c = values.T
    offset = rng.normal(0, 0.35, n)
    mean = np.exp(
        0.25 + 0.2 * main + 0.35 * a * b - 0.25 * b * c + 0.2 * a * c + offset
    )
    frame = pl.DataFrame(
        {
            "main": main,
            "A": a,
            "B": b,
            "C": c,
            "offset": offset,
            "y": rng.poisson(mean).astype(float),
            "split": [1] * train_rows + [0] * (n - train_rows),
        }
    )
    project = Project()
    project.data.roles = {
        "main": "predictor",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
        "offset": "offset",
        "y": "target",
        "split": "split",
    }
    project.data.split.column = "split"
    config = project.new_model("benchmark")
    config.predictors = ["main"]
    config.penalty.alpha = 0.01
    config.pair_stages = [
        PairStageConfig("ab", "A", "B"),
        PairStageConfig("bc", "B", "C"),
        PairStageConfig("ac", "A", "C"),
    ][: args.stages]
    teacher_fits = 0
    main_fits = 0
    real_teacher = pair_stages.fit_catboost_pair_raw
    real_main = pair_stages._fit_main_effects

    def counted_teacher(*args, **kwargs):
        nonlocal teacher_fits
        teacher_fits += 1
        return real_teacher(*args, **kwargs)

    def counted_main(*args, **kwargs):
        nonlocal main_fits
        main_fits += 1
        return real_main(*args, **kwargs)

    pair_stages.fit_catboost_pair_raw = counted_teacher
    pair_stages._fit_main_effects = counted_main
    cache: dict = {}
    start = time.perf_counter()
    run = run_model(project, frame, "benchmark", pair_stages_cache=cache)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "rows": n,
                "train_rows": train_rows,
                "holdout_rows": n - train_rows,
                "stages": len(run.pair_stages),
                "candidate_counts": [
                    len(stage.cv_candidates) for stage in run.pair_stages
                ],
                "selected": [
                    (
                        None
                        if stage.chosen_candidate is None
                        else stage.chosen_candidate.__dict__
                    )
                    for stage in run.pair_stages
                ],
                "cv_table_losses": [stage.table_cv_loss for stage in run.pair_stages],
                "teacher_fits": teacher_fits,
                "fold_main_fits": main_fits,
                "fold_prefix_cache_entries": len(cache.get("fold_prefix", {})),
                "wall_seconds": round(elapsed, 3),
                "peak_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
