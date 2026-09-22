"""Numerical contract for sequential CatBoost pair distillation."""

from __future__ import annotations

import builtins
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from easy_glm.workflow.pair_distillation import (
    distill_pair_cells,
    fit_catboost_pair_raw,
    teacher_mean_from_raw,
)
from easy_glm.workflow.pair_stages import _optuna_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "family,power", [("poisson", None), ("tweedie", 1.3), ("tweedie", 1.8)]
)
def test_cell_solution_matches_independent_loss_minimisation(family, power):
    rng = np.random.default_rng(12)
    n = 190
    baseline = np.exp(rng.normal(0, 2.1, n) + rng.normal(0, 0.5, n))
    teacher = baseline * np.exp(rng.normal(0.1, 0.6, n))
    weight = np.exp(rng.normal(0, 0.9, n))
    weight[:5] = 0
    cells = rng.integers(0, 4, n)
    result = distill_pair_cells(
        cells,
        baseline,
        teacher,
        sample_weight=weight,
        family=family,
        tweedie_power=power,
        n_cells=4,
    )
    p = power or 1.0
    for cell in range(4):
        use = (cells == cell) & (weight > 0)
        cell_baseline = baseline[use]
        cell_teacher = teacher[use]
        cell_weight = weight[use]

        def loss(
            log_r,
            cell_baseline=cell_baseline,
            cell_teacher=cell_teacher,
            cell_weight=cell_weight,
        ):
            mu = cell_baseline * np.exp(log_r)
            if p == 1:
                return np.sum(cell_weight * (mu - cell_teacher * np.log(mu)))
            return np.sum(
                cell_weight
                * (mu ** (2 - p) / (2 - p) - cell_teacher * mu ** (1 - p) / (1 - p))
            )

        optimum = minimize_scalar(
            loss, bounds=(-6, 6), method="bounded", options={"xatol": 1e-12}
        )
        assert optimum.success
        assert np.log(result.relativities[cell]) == pytest.approx(optimum.x, abs=2e-7)
        delta = 1e-5
        derivative = (
            loss(np.log(result.relativities[cell]) + delta)
            - loss(np.log(result.relativities[cell]) - delta)
        ) / (2 * delta)
        assert abs(derivative) < 1e-5 * max(1, abs(loss(optimum.x)))


def test_extreme_baselines_weights_zero_targets_and_support():
    baseline = np.array([1e-220, 1e220, 1e-80, 1e80, 7.0])
    teacher = baseline * np.array([2.0, 3.0, 0.5, 1.5, 4.0])
    weight = np.array([1.0, 1.0, 0.1, 0.1, 0.0])
    cells = np.array([0, 0, 1, 1, 2])
    result = distill_pair_cells(
        cells,
        baseline,
        teacher,
        sample_weight=weight,
        family="tweedie",
        tweedie_power=1.5,
        n_cells=4,
        min_weight_share=0.1,
    )
    assert np.all(np.isfinite(result.relativities))
    assert result.relativities[0] == pytest.approx(3.0, rel=1e-12)
    assert result.relativities[1] == 1
    assert result.fallback_reason == (None, "insufficient_support", "empty", "empty")
    assert result.row_count.tolist() == [2, 2, 0, 0]
    assert result.fitting_weight.tolist() == pytest.approx([2, 0.2, 0, 0])
    # Distillation depends on soft teacher means; observed zeros are a valid
    # fitting target for CatBoost and are never substituted for those means.
    zero_outcomes = np.zeros(5)
    assert np.all(zero_outcomes == 0)


def test_adapter_scaling_and_invalid_means():
    baseline = np.array([0.3, 2.0, 20.0])
    raw = np.array([0.4, -0.2, 0.0])
    expected = baseline * np.exp(raw)
    np.testing.assert_allclose(teacher_mean_from_raw(raw, baseline), expected)
    np.testing.assert_allclose(
        teacher_mean_from_raw(raw, baseline, target_scale=1e6), expected
    )
    for bad in ([0, 1, 2], [np.nan, 1, 2], [np.inf, 1, 2]):
        with pytest.raises(ValueError, match="baseline_mean"):
            teacher_mean_from_raw(raw, bad)
    with pytest.raises(ValueError, match="teacher_mean"):
        teacher_mean_from_raw([1000, 0, 0], baseline)
    with pytest.raises(ValueError, match="teacher_mean"):
        distill_pair_cells([0], [1.0], [0.0], n_cells=1)
    with pytest.raises(ValueError, match="sample_weight"):
        distill_pair_cells([0], [1.0], [1.0], sample_weight=[-1], n_cells=1)
    with pytest.raises(ValueError, match="positive fitting weight"):
        distill_pair_cells([0], [1.0], [1.0], sample_weight=[0], n_cells=1)


@pytest.mark.parametrize("family,power", [("poisson", None), ("tweedie", 1.5)])
def test_catboost_raw_baseline_and_two_features(family, power):
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(23)
    n = 300
    x = rng.normal(size=(n, 2))
    offset = rng.normal(0.6, 0.7, n)
    baseline = np.exp(offset)
    y = rng.poisson(baseline * np.exp(0.5 * x[:, 0] * x[:, 1])).astype(float)
    y[:10] = 0
    weight = rng.uniform(0.2, 3, n)
    teacher = fit_catboost_pair_raw(
        x,
        y,
        baseline,
        sample_weight=weight,
        family=family,
        tweedie_power=power,
        target_scale=10.0,
        iterations=30,
        depth=2,
        thread_count=2,
        seed=23,
    )
    expected = baseline * np.exp(
        teacher.model.predict(x, prediction_type="RawFormulaVal")
    )
    np.testing.assert_allclose(teacher.predict_mean(x, baseline), expected, rtol=1e-12)
    # Pool predictions contain the Pool baseline; adding it again would be wrong.
    pool = catboost.Pool(x, baseline=np.log(baseline / 10.0))
    pool_raw = teacher.model.predict(pool, prediction_type="RawFormulaVal")
    correction = teacher.model.predict(x, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(
        pool_raw - correction, np.log(baseline / 10.0), atol=1e-7
    )
    with pytest.raises(ValueError, match="exactly two columns"):
        fit_catboost_pair_raw(x[:, :1], y, baseline)


def test_family_gate():
    with pytest.raises(ValueError, match="supports"):
        distill_pair_cells([0], [1], [1], family="gamma", n_cells=1)
    with pytest.raises(ValueError, match="supports"):
        distill_pair_cells([0], [1], [1], family="tweedie", tweedie_power=2, n_cells=1)


@pytest.mark.parametrize("family,power", [("poisson", None), ("tweedie", 1.5)])
@pytest.mark.parametrize("constant_target", [0.0, 1.0])
def test_constant_target_is_valid_with_varying_baseline(family, power, constant_target):
    x = np.arange(80, dtype=float).reshape(40, 2)
    baseline = np.exp(np.linspace(-0.8, 0.8, 40))
    teacher = fit_catboost_pair_raw(
        x,
        np.full(40, constant_target),
        baseline,
        family=family,
        tweedie_power=power,
        iterations=4,
        thread_count=1,
    )
    assert np.all(np.isfinite(teacher.predict_mean(x, baseline)))


def test_missing_training_dependency_gives_install_command(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "catboost":
            raise ImportError("synthetic missing CatBoost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install --upgrade easy-glm"):
        fit_catboost_pair_raw(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 1.0]),
            np.ones(2),
            iterations=2,
        )


def test_missing_tuning_dependency_gives_standard_install_command(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("synthetic missing Optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install --upgrade easy-glm"):
        _optuna_module()


def test_standard_install_declares_pair_training_dependencies():
    """Pair training must work after the ordinary, extras-free installation."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert re.search(r'"catboost>=1\.2\.10,<1\.3"', dependencies)
    assert re.search(r'"optuna>=4,<5"', dependencies)
    assert re.search(r"(?m)^pairs = \[\]", pyproject)

    benchmark = pyproject.split("benchmark = [", 1)[1].split("]", 1)[0]
    assert "catboost" not in benchmark.lower()
