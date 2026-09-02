"""The product's central promises, checked on every design the core supports.

Every case is scored on data containing nulls in numeric *and* categorical
predictors and an unseen categorical level, because those are exactly the rows
where a table-based scorer and the GLM can quietly disagree.

Invariants
----------
1. ``RateModel.predict(df, exposure_col=None) == fit.predict(df)`` (rtol 1e-10).
2. A JSON round-trip of the RateModel scores identically.
3. ``RateModel.to_excel`` writes the relativities the scorer uses, including
   manual adjustments. (The 0.3 defect was in the workbench download and the
   exported script, which used the fitted view; those paths are guarded in
   ``tests/test_c1_foundations.py``.)
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
    dens = rng.integers(1, 3000, n).astype(float)
    young_r2 = (age < 25) & (region == "R2")  # a genuine interaction
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + 0.05 * (power - 7)
        + 0.0001 * dens
        + np.where(region == "R1", 0.0, 0.25)
        + np.where(young_r2, 0.6, 0.0)
    )
    claims = rng.poisson(mu * expo).astype(float)
    age[rng.random(n) < 0.05] = np.nan
    region[rng.random(n) < 0.04] = None
    power = power.astype(object)
    power[rng.random(n) < 0.03] = None  # nulls in the integer-typed categorical too
    df = pl.DataFrame(
        {
            "ClaimNb": claims,
            "Exposure": expo,
            "DrivAge": age,
            "VehPower": pl.Series(
                [None if v is None else int(v) for v in power], dtype=pl.Int64
            ),
            "Region": region,
            "Density": dens,
            "logprem": np.log(current),
            "prem": current,
        }
    ).with_columns(pl.col("DrivAge").fill_nan(None))
    return df


def _scoring_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Holdout-like frame with an unseen categorical level injected and Density
    values pushed beyond the training range (below and above the clamp)."""
    out = df.tail(800)
    i = pl.arange(0, out.height)
    return out.with_columns(
        pl.when(i % 50 == 0)
        .then(pl.lit("UNSEEN"))
        .otherwise(pl.col("Region"))
        .alias("Region"),
        pl.when(i % 37 == 0)
        .then(pl.col("Density") + 10_000.0)
        .when(i % 37 == 1)
        .then(pl.col("Density") - 10_000.0)
        .otherwise(pl.col("Density"))
        .alias("Density"),
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
    # two-way interactions on top of the mains: cells are tied to the parents'
    # rate-table rows, so nulls (numeric null row) and unseen levels (Other row)
    # must land in the same cell for the GLM and the scorer
    "interaction_num_cat": {
        "predictors": ["DrivAge", "Region"],
        "categorical": None,
        "offset": None,
        "interactions": [("DrivAge", "Region")],
    },
    "interaction_num_num": {
        "predictors": ["DrivAge", "Density"],
        "categorical": None,
        "offset": None,
        "interactions": [("DrivAge", "Density")],
        # coarse knots so the 3,200 training rows fill the cells
        "knots": {"DrivAge": [25, 40, 60], "Density": [500, 1500]},
    },
    "interaction_cat_cat": {
        "predictors": ["VehPower", "Region"],
        "categorical": ["VehPower"],
        "offset": None,
        "interactions": [("VehPower", "Region")],
    },
    "interaction_with_offset": {
        "predictors": ["DrivAge", "VehPower", "Region"],
        "categorical": ["VehPower"],
        "offset": "logprem",
        "interactions": [("DrivAge", "Region"), ("VehPower", "Region")],
    },
    # piecewise-linear terms: log-linear inside each band, flat outside the
    # clamp, nulls on their own row — scored on data beyond the training range
    "linear_only": {
        "predictors": ["Density"],
        "categorical": None,
        "offset": None,
        "linear": ["Density"],
        "knots": {"Density": [300, 1000, 2000]},
    },
    "linear_mixed_all": {
        "predictors": ["DrivAge", "Density", "VehPower", "Region"],
        "categorical": ["VehPower"],
        "offset": "logprem",
        "linear": ["Density"],
        "knots": {"DrivAge": [25, 40, 60], "Density": [500, 1500]},
        "interactions": [("Density", "Region"), ("DrivAge", "VehPower")],
    },
}


@pytest.fixture(params=list(CASES), scope="module")
def fitted(request):
    case = CASES[request.param]
    df = _data()
    train = df.head(3200)
    spec = DesignSpec.from_data(
        train,
        case["predictors"],
        categorical=case["categorical"],
        min_level_share=0.02,
        knots=case.get("knots"),
        interactions=case.get("interactions"),
        min_cell_exposure=0.005,
        linear=case.get("linear"),
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
    if name.startswith("linear"):
        assert any(c.type == "linear" for c in rm.variables.values())
        assert (score_df["Density"] > 5_000).sum() > 0  # beyond the clamp
        assert (score_df["Density"] < 0).sum() > 0
    if "interaction" in name or name == "linear_mixed_all":
        # the fit must actually use cells, otherwise the case proves nothing
        assert any(c.type == "interaction" for c in rm.variables.values())
        cells = [
            c
            for k, c in zip(fit.spec.features, fit.coef, strict=True)
            if k.kind == "cell"
        ]
        assert cells and any(c != 0 for c in cells), "no interaction cell was fitted"
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
