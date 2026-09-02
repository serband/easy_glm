"""Planted-truth tests: a synthetic book with a known effect must be recovered.

Interactions (piece A): a strong ``Age × Region`` cell is planted; the fitted
model must reproduce it, thin non-signal cells must stay at 1.0, and the
A/E-by-pair diagnostic on a model *without* the interaction must expose it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, fit_glm, rate_tables
from easy_glm.workflow import ModelConfig, ae_by_pair, totals

PLANTED_LOG_EFFECT = 0.9  # young drivers in region R2 claim e^0.9 ≈ 2.46× more
KNOTS = [25, 30, 40, 50, 60]
FIT = {"family": "poisson", "weight_col": "Exposure", "divide_target_by_weight": True}


@pytest.fixture(scope="module")
def planted() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 40_000
    age = rng.integers(18, 80, n).astype(float)
    # R5 is deliberately rare (~0.2%): its cells hold 10–50 rows each and carry
    # no signal, so they must not pick up spurious adjustments
    region = rng.choice(
        ["R1", "R2", "R3", "R4", "R5"], n, p=[0.5, 0.3, 0.148, 0.05, 0.002]
    ).astype(object)
    expo = rng.uniform(0.2, 1.0, n)
    young_r2 = (age < 25) & (region == "R2")
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + np.where(region == "R1", 0.0, 0.25)
        + np.where(young_r2, PLANTED_LOG_EFFECT, 0.0)
    )
    return pl.DataFrame(
        {
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "Region": region,
        }
    )


def _double_difference(fit) -> float:
    """log[p(<25,R2) p(≥25,R1) / (p(<25,R1) p(≥25,R2))] — the interaction as the
    data identifies it, whichever way the lasso splits it between mains and cell."""
    probe = pl.DataFrame(
        {
            "DrivAge": [20.0, 20.0, 40.0, 40.0],
            "Region": ["R2", "R1", "R2", "R1"],
            "Exposure": [1.0] * 4,
        }
    )
    p = fit.predict(probe)
    return float(np.log(p[0] * p[3] / (p[1] * p[2])))


class TestPlantedInteraction:
    def test_cell_recovered_and_thin_cells_stay_flat(self, planted):
        spec = DesignSpec.from_data(
            planted,
            ["DrivAge", "Region"],
            knots={"DrivAge": KNOTS},
            min_level_share=0.001,  # keep R5 as a level
            interactions=[("DrivAge", "Region")],
            min_cell_exposure=0.0,  # keep every cell, including the thin ones
        )
        assert "R5" in spec["Region"].levels
        fit = fit_glm(planted, spec, "ClaimNb", alpha=2e-4, **FIT)
        # the planted effect, within 10% on the relativity scale (|Δlog| ≤ 0.1)
        assert abs(_double_difference(fit) - PLANTED_LOG_EFFECT) < 0.1
        tab = rate_tables(fit)["DrivAge×Region"]
        planted_cell = tab.filter((pl.col("to_a") == 25.0) & (pl.col("from_b") == "R2"))
        assert planted_cell["relativity"][0] > 1.2  # the cell itself carries part of it
        # thin non-signal cells (10–50 rows) stay within [0.98, 1.02]
        thin = tab.filter((pl.col("from_b") == "R5") & (pl.col("exposure") > 0))
        rows_per_cell = planted.filter(pl.col("Region") == "R5").height / thin.height
        assert 5 <= rows_per_cell <= 60
        assert thin["relativity"].min() >= 0.98 and thin["relativity"].max() <= 1.02
        # the planted cell is the strongest adjustment in the table; other kept
        # cells carry only sampling noise (fat cells are allowed some, thin ones none)
        clean = tab.filter(
            pl.col("kept") & ~((pl.col("to_a") == 25.0) & (pl.col("from_b") == "R2"))
        )
        assert planted_cell["relativity"][0] > clean["relativity"].max()
        assert (clean["relativity"] <= 1.25).all()

    def test_recovery_at_cv_chosen_alpha(self, planted):
        """The product fits by cross-validation. At a CV-chosen alpha ordinary
        lasso shrinkage recovers the planted interaction at roughly 65–85% of
        its size (the independent reviewer measured 0.59–0.75 of 0.90 over three
        seeds with cv=5 / 25 alphas); the thin cells must still be exactly flat
        and the planted cell must still be the strongest adjustment. Bounds:
        recovered in [0.45, 0.95] — wide enough for CV's fold randomness, tight
        enough to fail if the P1 rule or the cell indexing regress."""
        spec = DesignSpec.from_data(
            planted,
            ["DrivAge", "Region"],
            knots={"DrivAge": KNOTS},
            min_level_share=0.001,
            interactions=[("DrivAge", "Region")],
            min_cell_exposure=0.0,
        )
        fit = fit_glm(planted, spec, "ClaimNb", cv=5, n_alphas=20, **FIT)
        assert 2e-4 < fit.alpha < 3e-3  # a CV alpha, not the hand-picked one
        recovered = _double_difference(fit)
        assert 0.45 <= recovered <= 0.95, recovered
        tab = rate_tables(fit)["DrivAge×Region"]
        thin = tab.filter((pl.col("from_b") == "R5") & (pl.col("exposure") > 0))
        assert thin["relativity"].min() >= 0.98 and thin["relativity"].max() <= 1.02
        planted_cell = tab.filter((pl.col("to_a") == 25.0) & (pl.col("from_b") == "R2"))
        others = tab.filter(
            pl.col("kept") & ~((pl.col("to_a") == 25.0) & (pl.col("from_b") == "R2"))
        )
        assert planted_cell["relativity"][0] > 1.2
        assert planted_cell["relativity"][0] > others["relativity"].max()

    def test_ae_by_pair_exposes_the_missing_interaction(self, planted):
        spec = DesignSpec.from_data(
            planted,
            ["DrivAge", "Region"],
            knots={"DrivAge": KNOTS},
            min_level_share=0.001,
        )
        fit = fit_glm(planted, spec, "ClaimNb", alpha=5e-4, **FIT)
        cfg = ModelConfig(
            target="ClaimNb", weight="Exposure", divide_target_by_weight=True
        )
        actual, expected, w = totals(planted, cfg, fit.predict(planted))
        pair = ae_by_pair(
            planted, "DrivAge", "Region", actual, expected, w, knots_a=KNOTS
        )
        assert set(pair.columns) >= {
            "label_a",
            "label_b",
            "exposure",
            "actual",
            "expected",
            "ae",
        }
        assert pair["exposure"].sum() == pytest.approx(planted["Exposure"].sum())
        assert pair["actual"].sum() == pytest.approx(actual.sum())
        assert pair["expected"].sum() == pytest.approx(expected.sum())
        cell = pair.filter(
            (pl.col("label_a") == "< 25.0") & (pl.col("label_b") == "R2")
        )
        assert cell.height == 1
        assert abs(np.log(cell["ae"][0])) > 0.2
        # and everything else is roughly calibrated
        others = pair.filter(
            ~((pl.col("label_a") == "< 25.0") & (pl.col("label_b") == "R2"))
            & (pl.col("exposure") > 200)
        )
        assert others["ae"].abs().max() < 1.5

    def test_ae_by_pair_handles_categorical_pairs_and_nulls(self, planted):
        df = planted.with_columns(
            pl.when(pl.arange(0, planted.height) % 97 == 0)
            .then(None)
            .otherwise(pl.col("Region"))
            .alias("Region"),
            pl.when(pl.col("DrivAge") < 30)
            .then(pl.lit("young"))
            .otherwise(pl.lit("old"))
            .alias("AgeBand"),
        )
        w = df["Exposure"].to_numpy()
        actual = df["ClaimNb"].to_numpy()
        pair = ae_by_pair(df, "AgeBand", "Region", actual, actual * 1.1, w)
        assert "Other / Unknown" in pair["label_b"].to_list()
        assert pair["exposure"].sum() == pytest.approx(w.sum())
        with_claims = pair.filter(pl.col("expected") > 0)
        np.testing.assert_allclose(with_claims["ae"].to_numpy(), 1 / 1.1)
