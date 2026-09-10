"""The intercept-only benchmark must be identifiable for weighted logit fits."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from easy_glm.workflow import Project, VariableDesign, null_model_predict, run_model


@pytest.mark.parametrize("divide", [False, True])
def test_weighted_logit_offset_null_matches_scalar_likelihood_and_full_workflow(divide):
    rng = np.random.default_rng(318)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    weight = rng.integers(2, 12, n).astype(float)
    offset = rng.uniform(-0.5, 0.5, n)
    successes = rng.binomial(weight.astype(int), expit(0.6 + 0.7 * (x > 0) + offset))
    proportion = successes / weight
    frame = pl.DataFrame(
        {
            "x": x,
            "weight": weight,
            "offset": offset,
            "target": successes if divide else proportion,
            "traintest": (np.arange(n) < 450).astype(int),
        }
    )
    project = Project(name="Weighted logit")
    project.data.roles = {
        "x": "predictor",
        "target": "target",
        "weight": "weight",
        "offset": "offset",
        "traintest": "split",
    }
    project.design.variables["x"] = VariableDesign(knots=[-0.5, 0.0, 0.5])
    cfg = project.new_model(
        "logit", family="binomial", divide_target_by_weight=divide, predictors=["x"]
    )
    cfg.penalty.alpha = 0.01
    train, holdout = frame[:450], frame[450:]
    intercept = brentq(
        lambda value: np.sum(
            weight[:450] * (proportion[:450] - expit(value + offset[:450]))
        ),
        -20.0,
        20.0,
    )
    expected = expit(intercept + offset[450:])
    prediction = null_model_predict(project, cfg, train, holdout)
    np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-7)
    # Holdout outcomes never enter the fitted benchmark.
    changed = holdout.with_columns(pl.lit(0.0).alias("target"))
    np.testing.assert_array_equal(
        null_model_predict(project, cfg, train, changed), prediction
    )
    run = run_model(project, frame, "logit")
    assert np.isfinite(run.metrics["holdout"]["deviance_explained"])
