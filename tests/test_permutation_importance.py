"""Permutation importance uses original source columns and original GLM scoring."""

import pickle

import numpy as np
import polars as pl
import pytest

from easy_glm.core.design import (
    CategoricalEncoder,
    DesignSpec,
    InteractionEncoder,
    StepEncoder,
)
from easy_glm.core.fit import TwoStageFit, fit_glm, fit_two_stage
from easy_glm.workflow.diagnostics import permutation_importance, unit_values


def fixture(family="poisson", divide=True):
    rng = np.random.default_rng(9)
    n = 320
    x = rng.integers(0, 3, n).astype(float)
    x[::17] = np.nan
    noise = rng.integers(0, 3, n).astype(float)
    noise[::19] = np.nan
    w = rng.uniform(0.3, 2, n)
    offset = rng.normal(0, 0.2, n)
    mu = np.exp(0.7 * np.nan_to_num(x) + offset)
    if family == "binomial":
        w = np.ones(n)
        y = rng.binomial(1, mu / (1 + mu), n)
        divide = False
    elif family == "gamma":
        y = rng.gamma(3, mu / 3) * w if divide else rng.gamma(3, mu / 3)
    elif family == "normal":
        y = rng.normal(mu, 0.1) * w if divide else rng.normal(mu, 0.1)
    else:
        y = rng.poisson(mu * w) if divide else rng.poisson(mu)
    frame = pl.DataFrame({"x": x, "noise": noise, "y": y, "w": w, "offset": offset})
    spec = DesignSpec({name: StepEncoder(name, [0.5, 1.5]) for name in ["x", "noise"]})
    fit = fit_glm(
        frame,
        spec,
        "y",
        family=family,
        alpha=0.001,
        weight_col=None if family == "binomial" else "w",
        offset_col="offset",
        divide_target_by_weight=divide,
    )
    return frame, fit


def manual(frame, fit, name):
    y, w = unit_values(frame, fit)
    offset = frame["offset"].to_numpy()
    weight = w if fit.weight_col else None

    def loss(data):
        return (
            fit.model.family_instance.deviance(
                y, fit.predict(data, offset=offset), sample_weight=weight
            )
            / w.sum()
        )

    baseline = loss(frame)
    rng = np.random.default_rng(42)
    losses = np.array(
        [
            loss(frame.with_columns(frame[name].gather(rng.permutation(frame.height))))
            for _ in range(5)
        ]
    )
    return baseline, losses


@pytest.mark.parametrize(
    "family,divide",
    [
        ("poisson", True),
        ("poisson", False),
        ("gamma", True),
        ("normal", False),
        ("binomial", False),
    ],
)
def test_matches_seeded_original_column_reference_and_fixed_scale(
    family, divide, monkeypatch
):
    frame, fit = fixture(family, divide)
    expected = {name: manual(frame, fit, name) for name in ["x", "noise"]}
    before_frame = frame.clone()
    before_fit = pickle.dumps(fit)

    def forbidden(*args, **kwargs):
        raise AssertionError("importance must not fit or build a design matrix")

    monkeypatch.setattr(DesignSpec, "build", forbidden)
    monkeypatch.setattr(type(fit.model), "fit", forbidden)
    rows = permutation_importance(fit, frame).to_dicts()
    for row in rows:
        baseline, losses = expected[row["variable"]]
        assert row["baseline_deviance"] == pytest.approx(baseline, abs=1e-13)
        assert row["shuffled_deviance"] == pytest.approx(losses.mean(), abs=1e-13)
        assert row["importance"] == pytest.approx((losses - baseline).mean(), abs=1e-13)
        assert row["std"] == pytest.approx((losses - baseline).std(), abs=1e-13)
    assert frame.equals(before_frame)
    assert pickle.dumps(fit) == before_fit
    assert rows == permutation_importance(fit, frame).to_dicts()


def test_informative_rank_and_zero_coefficient_predictor():
    frame, fit = fixture()
    fit.model.coef_[fit.spec.slices()["noise"]] = 0
    rows = permutation_importance(fit, frame).to_dicts()
    assert rows[0]["variable"] == "x" and rows[0]["importance"] > 0.2
    assert rows[1]["variable"] == "noise"
    assert rows[1]["importance"] == 0 and rows[1]["std"] == 0


def test_negative_importance_is_not_clipped():
    frame, fit = fixture()
    # A deliberately wrong fitted effect improves when its input is scrambled.
    fit.model.coef_[:] = 0
    fit.model.coef_[fit.spec.slices()["x"]] = -0.5
    rows = permutation_importance(fit, frame).to_dicts()
    assert rows[-1]["variable"] == "x"
    assert rows[-1]["importance"] < 0


def test_one_permuted_parent_drives_main_and_interaction_encoders():
    frame, original = fixture()
    spec = original.spec
    spec.add_interaction(
        InteractionEncoder.from_data(
            spec["x"], spec["noise"], frame, weights=frame["w"], min_cell_exposure=0
        )
    )
    fit = fit_two_stage(
        frame,
        spec,
        "y",
        alpha=0.002,
        stage2_alpha=0.002,
        weight_col="w",
        offset_col="offset",
        divide_target_by_weight=True,
    )
    assert isinstance(fit, TwoStageFit)
    # A parent still belongs to importance if its main coefficients are zero.
    fit.model.coef_[spec.slices()["noise"]] = 0
    results = permutation_importance(fit, frame).to_dicts()
    assert {r["variable"] for r in results} == {"x", "noise"}
    for row in results:
        baseline, losses = manual(frame, fit, row["variable"])
        assert row["importance"] == pytest.approx((losses - baseline).mean(), abs=1e-13)


def test_protected_columns_and_validation():
    frame, fit = fixture()
    assert permutation_importance(fit, frame, protected_columns=("x",))[
        "variable"
    ].to_list() == ["noise"]
    with pytest.raises(ValueError, match="two training rows"):
        permutation_importance(fit, frame.head(1))
    with pytest.raises(ValueError, match="positive integer"):
        permutation_importance(fit, frame, repeats=0)


def test_categorical_null_and_other_values_keep_their_type():
    frame, _ = fixture()
    labels = [
        None if i % 13 == 0 else ("unseen" if i % 7 == 0 else str(i % 3))
        for i in range(frame.height)
    ]
    frame = frame.with_columns(pl.Series("noise", labels))
    spec = DesignSpec(
        {
            "x": StepEncoder("x", [0.5, 1.5]),
            "noise": CategoricalEncoder("noise", ["0", "1", "2"]),
        }
    )
    fit = fit_glm(
        frame,
        spec,
        "y",
        alpha=0.01,
        weight_col="w",
        offset_col="offset",
        divide_target_by_weight=True,
    )
    rows = permutation_importance(fit, frame).to_dicts()
    for row in rows:
        baseline, losses = manual(frame, fit, row["variable"])
        assert row["importance"] == pytest.approx((losses - baseline).mean(), abs=1e-13)
    assert frame["noise"].to_list() == labels
    assert frame["noise"].dtype == pl.Utf8
