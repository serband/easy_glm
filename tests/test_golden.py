"""Golden French-motor test.

Fits a fixed model on a checked-in, deterministic 50,000-row subsample of the
French motor set (``tests/fixtures/french_motor_50k.parquet``) and compares
against recorded numbers. It runs in CI on every push. The fixture is exactly
``load_external_dataframe().select(COLUMNS).sort("IDpol").sample(n=50_000,
seed=20260902).sort("IDpol")`` with the id and categorical columns cast to text —
see ``tests/fixtures/make_french_motor_50k.py``, which regenerates or checks it.

**Any change to a number in ``GOLDEN`` is a blocking review item** and needs a
written reason in the pull request (see docs/RELEASE_0.4_PLAN.md, §R7).

Tolerances: ``gini`` pools tied scores, so it is independent of row order and
of ``e / w`` rounding noise (before that fix it moved at the 1e-5 level between
identical runs). The fit is deterministic (glum's coordinate descent from a fixed
start; a refit on the same machine reproduces coefficients to 1e-15), so the
only expected variation is BLAS/platform rounding, which is far below 1e-6
relative on aggregate metrics and relativities. Integer counts are exact.
Note the fit is *not* bitwise reproducible (BLAS threading moves the last digit,
about 5e-16 relative), so ``RTOL`` must not be tightened towards machine epsilon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, base_rate, fit_glm, rate_tables, to_rate_model
from easy_glm.engine import RateModel
from easy_glm.workflow import ModelConfig, deviance_stats, gini, totals, unit_values

FIXTURE = Path(__file__).parent / "fixtures" / "french_motor_50k.parquet"
PREDICTORS = [
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "Density",
    "VehPower",
    "Region",
    "VehBrand",
    "VehGas",
    "Area",
]
ALPHA = 0.001
SPLIT_SEED = 42

GOLDEN = {
    "train_rows": 34887,
    "holdout_rows": 15113,
    "n_features": 112,
    "n_nonzero": 61,
    "base_rate": 0.045973033712248534,
    "holdout_ae": 0.9956834429674676,
    "holdout_gini": 0.33729515212920436,
    "holdout_dev_explained": 0.05382065989126372,
    # first five DrivAge bands: < 25, [25, 28), [28, 30), [30, 32), [32, 34)
    "drivage_first5": [
        0.6590053328,
        0.5515189057,
        0.5515189057,
        0.5632373546,
        0.6170785699,
    ],
}
RTOL = 1e-6


@pytest.fixture(scope="module")
def golden_fit():
    df = pl.read_parquet(FIXTURE)
    rng = np.random.default_rng(SPLIT_SEED)
    df = df.with_columns(
        pl.Series("traintest", (rng.random(df.height) < 0.7).astype(np.int64))
    )
    train = df.filter(pl.col("traintest") == 1)
    holdout = df.filter(pl.col("traintest") == 0)
    spec = DesignSpec.from_data(train, PREDICTORS, weight_col="Exposure")
    fit = fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=ALPHA,
    )
    rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
    return fit, rm, train, holdout


def test_fixture_is_the_recorded_subsample():
    df = pl.read_parquet(FIXTURE)
    assert df.height == 50_000
    assert set(PREDICTORS + ["IDpol", "ClaimNb", "Exposure"]) <= set(df.columns)


def test_golden_shape(golden_fit):
    fit, rm, train, holdout = golden_fit
    assert train.height == GOLDEN["train_rows"]
    assert holdout.height == GOLDEN["holdout_rows"]
    assert len(fit.coef) == GOLDEN["n_features"]
    assert int((fit.coef != 0).sum()) == GOLDEN["n_nonzero"]


def test_golden_metrics(golden_fit):
    fit, rm, train, holdout = golden_fit
    cfg = ModelConfig(
        family="poisson",
        target="ClaimNb",
        weight="Exposure",
        divide_target_by_weight=True,
        predictors=PREDICTORS,
    )
    pred = rm.predict(holdout, exposure_col=None)
    actual, expected, w = totals(holdout, cfg, pred)
    y, wu = unit_values(holdout, cfg)
    dev = deviance_stats(fit.model.family_instance, y, pred, wu)
    assert rm.base_rate == pytest.approx(GOLDEN["base_rate"], rel=RTOL)
    assert actual.sum() / expected.sum() == pytest.approx(
        GOLDEN["holdout_ae"], rel=RTOL
    )
    assert gini(actual, expected, w) == pytest.approx(GOLDEN["holdout_gini"], rel=RTOL)
    assert dev["deviance_explained"] == pytest.approx(
        GOLDEN["holdout_dev_explained"], rel=RTOL
    )


def test_golden_drivage_relativities(golden_fit):
    _fit, rm, _train, _holdout = golden_fit
    rows = rm.variables["DrivAge"].table
    assert rows[0].to_ == 25.0 and rows[1].from_ == 25.0
    got = [r.relativity for r in rows[:5]]
    np.testing.assert_allclose(got, GOLDEN["drivage_first5"], rtol=RTOL)


def test_golden_rate_model_matches_glm(golden_fit):
    fit, rm, _train, holdout = golden_fit
    np.testing.assert_allclose(
        rm.predict(holdout, exposure_col=None), fit.predict(holdout), rtol=1e-10
    )


def test_golden_hand_built_tables_match_exact_tables(golden_fit):
    """The C2 promise on the golden data: tables loaded through
    ``from_rate_tables`` score identically to the exact tables from the fit."""
    fit, rm, _train, holdout = golden_fit
    rebuilt = RateModel.from_rate_tables(rate_tables(fit), base_rate(fit))
    np.testing.assert_allclose(
        rebuilt.predict(holdout, exposure_col=None),
        rm.predict(holdout, exposure_col=None),
        rtol=1e-12,
    )


def test_fixture_matches_its_recipe():
    """Re-derive the fixture from the cached full dataset when it is available."""
    import importlib.util

    from easy_glm.core.data import _cache_path

    spec = importlib.util.spec_from_file_location(
        "make_fixture", Path(__file__).parent / "fixtures" / "make_french_motor_50k.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    url = "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"
    cache = _cache_path(url)
    if not cache.exists():
        pytest.skip("full French motor dataset not cached on this machine")
    regenerated = module.regenerate(pl.read_parquet(cache))
    assert regenerated.equals(pl.read_parquet(FIXTURE))
