"""The product's central promises, checked on every design the core supports.

Every case is scored on data containing nulls in numeric *and* categorical
predictors and an unseen categorical level, because those are exactly the rows
where a table-based scorer and the GLM can quietly disagree.

Invariants
----------
1. ``RateModel.predict(df, exposure_col=None) == fit.predict(df)`` (rtol 1e-10).
2. A JSON round-trip of the RateModel scores identically.
3. ``RateModel.to_excel`` writes the relativities the scorer uses, including
   manual adjustments.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, fit_glm, to_rate_model
from easy_glm.engine import RateModel

RTOL = 1e-10


def _data(seed: int = 3, n: int = 4000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    power = rng.integers(4, 12, n)  # integer-typed categorical candidate
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.5, 0.3, 0.15, 0.05]).astype(
        object
    )
    expo = rng.uniform(0.2, 1.0, n)
    current = rng.uniform(150.0, 900.0, n)  # e.g. current premium
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + 0.05 * (power - 7)
        + np.where(region == "R1", 0.0, 0.25)
    )
    claims = rng.poisson(mu * expo).astype(float)
    age[rng.random(n) < 0.05] = np.nan
    region[rng.random(n) < 0.04] = None
    df = pl.DataFrame(
        {
            "ClaimNb": claims,
            "Exposure": expo,
            "DrivAge": age,
            "VehPower": power,
            "Region": region,
            "logprem": np.log(current),
            "prem": current,
        }
    ).with_columns(pl.col("DrivAge").fill_nan(None))
    return df


def _scoring_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Holdout-like frame with an unseen categorical level injected."""
    out = df.tail(800)
    return out.with_columns(
        pl.when(pl.arange(0, out.height) % 50 == 0)
        .then(pl.lit("UNSEEN"))
        .otherwise(pl.col("Region"))
        .alias("Region")
    )


CASES = {
    "step_only": {"predictors": ["DrivAge"], "categorical": None, "offset": None},
    "categorical_string": {
        "predictors": ["Region"],
        "categorical": None,
        "offset": None,
    },
    "categorical_integer": {
        "predictors": ["VehPower"],
        "categorical": ["VehPower"],
        "offset": None,
    },
    "mixed": {
        "predictors": ["DrivAge", "VehPower", "Region"],
        "categorical": ["VehPower"],
        "offset": None,
    },
    "mixed_with_offset": {
        "predictors": ["DrivAge", "VehPower", "Region"],
        "categorical": ["VehPower"],
        "offset": "logprem",
    },
}


@pytest.fixture(params=list(CASES), scope="module")
def fitted(request):
    case = CASES[request.param]
    df = _data()
    train = df.head(3200)
    spec = DesignSpec.from_data(
        train, case["predictors"], categorical=case["categorical"], min_level_share=0.02
    )
    fit = fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        offset_col=case["offset"],
        alpha=0.002,
    )
    rm = to_rate_model(fit, exposure_col="Exposure")
    return request.param, fit, rm, _scoring_frame(df)


def test_rate_model_reproduces_glm(fitted):
    name, fit, rm, score_df = fitted
    assert score_df["DrivAge"].null_count() > 0 and score_df["Region"].null_count() > 0
    assert (score_df["Region"] == "UNSEEN").sum() > 0
    np.testing.assert_allclose(
        rm.predict(score_df, exposure_col=None),
        fit.predict(score_df),
        rtol=RTOL,
        atol=0,
    )


def test_json_roundtrip_scores_identically(fitted, tmp_path):
    name, fit, rm, score_df = fitted
    rm.to_json(tmp_path / f"{name}.easyglm")
    back = RateModel.from_json(tmp_path / f"{name}.easyglm")
    np.testing.assert_allclose(
        back.predict(score_df, exposure_col=None),
        rm.predict(score_df, exposure_col=None),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        back.predict(score_df, exposure_col=None), fit.predict(score_df), rtol=RTOL
    )


def _excel_relativities(path) -> dict[str, list[float]]:
    """Every sheet that has a ``relativity`` column, read back with polars."""
    sheets = pl.read_excel(path, sheet_id=0)
    return {
        name: frame["relativity"].to_list()
        for name, frame in sheets.items()
        if "relativity" in frame.columns
    }


def test_excel_matches_scorer_including_adjustments(fitted, tmp_path):
    name, fit, rm, score_df = fitted
    var = next(iter(rm.variables))
    row = rm.variables[var].table[1]
    rm.update_relativity(var, row.from_, row.to_, 3.0)
    assert rm.variables[var].table[1].relativity == 3.0
    path = rm.to_excel(tmp_path / f"{name}.xlsx")
    sheets = _excel_relativities(path)
    assert var in sheets, sheets.keys()
    expected = [r.relativity for r in rm.variables[var].table]
    np.testing.assert_allclose(sheets[var], expected, rtol=1e-9)
