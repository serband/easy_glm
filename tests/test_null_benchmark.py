"""Intercept-only benchmarks retain accurate predictions without solver stalls."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from easy_glm.core.fit import resolve_family
from easy_glm.workflow import Project, VariableDesign, null_model_predict, run_model
from easy_glm.workflow.diagnostics import deviance_stats


@pytest.mark.parametrize("threads", [1, 10])
@pytest.mark.parametrize("divide", [False, True])
def test_tweedie_cost_null_matches_weighted_mean_and_deviance(threads, divide):
    rng = np.random.default_rng(42)
    n = 43_602
    weight = rng.uniform(0.005, 3.0, n)
    count = rng.poisson(0.045 * weight)
    amount = count * rng.lognormal(8.0, 1.4, n)
    rate = amount / weight
    frame = pl.DataFrame({"target": amount if divide else rate, "exposure": weight})
    project = Project(name="Claims cost benchmark")
    project.data.roles = {"target": "target", "exposure": "weight"}
    cfg = project.new_model(
        "cost", family="tweedie", tweedie_power=1.5, divide_target_by_weight=divide
    )
    # No offset: the intercept-only Tweedie MLE is the exposure-weighted mean.
    oracle = np.full(n, amount.sum() / weight.sum())
    with (
        threadpool_limits(limits=threads),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        prediction = null_model_predict(project, cfg, frame, frame)
    assert not caught, [str(w.message) for w in caught]
    np.testing.assert_allclose(prediction, oracle, rtol=1e-8, atol=0)
    # Independent p=1.5 deviance formula, including the zero-claim observations.
    expected_deviance = float(
        np.sum(
            4 * weight * (rate / np.sqrt(oracle) - 2 * np.sqrt(rate) + np.sqrt(oracle))
        )
    )
    family, _, _ = resolve_family("tweedie", 1.5)
    actual = deviance_stats(family, rate, prediction, weight, mu0_unit=prediction)
    assert actual["null_deviance"] == pytest.approx(expected_deviance, rel=1e-10)


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
