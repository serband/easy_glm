import math

import numpy as np
import pytest

from easy_glm.workflow.diagnostics import (
    family_metrics,
    gini,
    metric_definitions,
)


def test_gaussian_metrics_use_weighted_unit_errors_and_eval_subset_mean():
    observed = np.array([1.0, -1.0])
    predicted = np.array([2.0, 1.0])
    weight = np.array([1.0, 3.0])

    metrics = family_metrics("gaussian", observed, predicted, weight)

    assert metrics["rmse"] == pytest.approx(math.sqrt(13.0 / 4.0))
    assert metrics["mae"] == pytest.approx(7.0 / 4.0)
    assert metrics["r2"] == pytest.approx(1.0 - 13.0 / 3.0)
    assert metrics["log_loss"] is None


def test_gaussian_r2_is_none_for_constant_observed_response():
    metrics = family_metrics(
        "gaussian",
        np.array([-2.0, -2.0]),
        np.array([-2.0, -1.0]),
        np.array([1.0, 2.0]),
    )

    assert metrics["r2"] is None
    assert "constant observed response" in metrics["metric_reasons"]["r2"]


def test_binomial_metrics_match_independent_weighted_formulas():
    observed = np.array([0.0, 1.0, 1.0])
    predicted = np.array([0.1, 0.7, 0.8])
    weight = np.array([1.0, 2.0, 1.0])

    metrics = family_metrics("binomial", observed, predicted, weight)
    expected_log_loss = -(math.log(0.9) + 2 * math.log(0.7) + math.log(0.8)) / 4
    expected_brier = (0.1**2 + 2 * 0.3**2 + 0.2**2) / 4

    assert metrics["log_loss"] == pytest.approx(expected_log_loss)
    assert metrics["brier"] == pytest.approx(expected_brier)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["response_kind"] == "binary"


def test_binomial_fractions_have_squared_error_label_and_no_auc():
    metrics = family_metrics(
        "binomial",
        np.array([0.2, 0.8]),
        np.array([0.3, 0.6]),
        np.array([2.0, 1.0]),
    )
    definitions = metric_definitions("binomial", metrics=metrics)

    assert metrics["brier"] == pytest.approx((2 * 0.1**2 + 0.2**2) / 3)
    assert metrics["roc_auc"] is None
    assert metrics["response_kind"] == "fractional"
    assert (
        next(item for item in definitions if item["key"] == "brier")["label"]
        == "Mean squared error (observed proportion)"
    )


def test_binomial_auc_needs_positive_weight_in_both_classes():
    metrics = family_metrics(
        "binomial",
        np.array([0.0, 1.0]),
        np.array([0.2, 0.8]),
        np.array([1.0, 0.0]),
    )

    assert metrics["roc_auc"] is None
    assert "both classes" in metrics["metric_reasons"]["roc_auc"]


def test_binomial_endpoint_predictions_are_finite_without_warning():
    with np.errstate(all="raise"):
        metrics = family_metrics(
            "binomial",
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        )

    assert metrics["log_loss"] is not None
    assert math.isfinite(metrics["log_loss"])
    assert metrics["log_loss"] > 30


def test_binomial_rejects_values_outside_the_probability_domain():
    with pytest.raises(ValueError, match="predictions between 0 and 1"):
        family_metrics(
            "binomial",
            np.array([0.0, 1.0]),
            np.array([-0.1, 1.1]),
        )


def test_gini_is_positive_scale_invariant_and_rejects_undefined_inputs():
    actual = np.array([0.0, 2.0, 8.0])
    expected = np.array([1.0, 3.0, 9.0])
    weight = np.array([1.0, 2.0, 1.0])

    value = gini(actual, expected, weight)
    assert value == pytest.approx(gini(actual, expected * 17.0, weight))
    assert np.isnan(gini(np.array([-1.0, 2.0]), expected[:2], weight[:2]))
    assert np.isnan(gini(np.array([1.0, 2.0]), np.array([1.0, -2.0])))
    assert np.isnan(gini(np.array([1.0, 2.0]), np.array([1.0, np.inf])))
    assert np.isnan(gini(np.array([1.0, 2.0]), np.ones(2), np.array([1.0, -1.0])))
    assert np.isnan(gini(np.array([1.0, 2.0]), np.ones(2), np.array([1.0, 2.0])))


def test_metric_definitions_name_tweedie_power_and_separate_r2_from_deviance():
    tweedie = metric_definitions("tweedie", 1.7)
    gaussian = metric_definitions("gaussian")

    assert any(item["label"] == "Tweedie mean deviance (p=1.7)" for item in tweedie)
    assert [item["key"] for item in gaussian if item["section"] == "primary"] == [
        "rmse",
        "mae",
        "r2",
    ]
    assert all("accuracy" not in item["label"].lower() for item in tweedie + gaussian)


def test_core_normal_family_alias_uses_gaussian_metrics():
    metrics = family_metrics("normal", np.array([1.0, 3.0]), np.array([2.0, 2.0]))

    assert metrics["rmse"] == pytest.approx(1.0)
    assert [item["key"] for item in metric_definitions("normal")[:3]] == [
        "rmse",
        "mae",
        "r2",
    ]
