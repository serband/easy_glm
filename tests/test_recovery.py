"""Planted-truth tests: a synthetic book with a known effect must be recovered.

Interactions (piece A): a strong ``Age × Region`` cell is planted; the fitted
model must reproduce it, thin non-signal cells must stay at 1.0, and the
A/E-by-pair diagnostic on a model *without* the interaction must expose it.

Piecewise-linear terms (piece B): a log-linear mileage effect with two slope
changes is planted; the fitted slopes must be recovered, only knots near the
true bends may carry a slope change, and the curve must be flat beyond the
training range.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model
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


# --------------------------------------------------------------------------
# piece B: planted piecewise-linear effect
# --------------------------------------------------------------------------
TRUE_SLOPES = (0.00010, 0.00003, -0.00004)  # per mile, on the log scale
TRUE_BENDS = (8_000.0, 20_000.0)


@pytest.fixture(scope="module")
def planted_linear() -> pl.DataFrame:
    rng = np.random.default_rng(23)
    n = 120_000
    mileage = rng.uniform(0, 30_000, n)
    expo = rng.uniform(0.3, 1.0, n)
    s1, s2, s3 = TRUE_SLOPES
    b1, b2 = TRUE_BENDS
    lp = (
        -2.2
        + s1 * np.minimum(mileage, b1)
        + s2 * np.clip(mileage - b1, 0, b2 - b1)
        + s3 * np.maximum(mileage - b2, 0)
    )
    return pl.DataFrame(
        {
            "ClaimNb": rng.poisson(np.exp(lp) * expo).astype(float),
            "Exposure": expo,
            "Mileage": mileage,
        }
    )


class TestPlantedLinear:
    KNOTS = [2_000.0 * k for k in range(1, 15)]  # every 2,000 miles
    ALPHA = 3e-5

    def test_slopes_recovered_and_bends_are_sparse(self, planted_linear):
        """At a moderate penalty the *average* slope of each true segment is
        recovered within 10% (the local slope of a single 2,000-mile band is a
        noisy statistic and is not asserted), the whole fitted curve is within
        0.06 of the true log relativity everywhere on [0, 30000] (the property
        an actuary cares about, and robust to where the lasso puts the bends),
        and the slope changes concentrate at the two true bends: within two knot
        spacings of each bend the fitted change has the right sign and 60–140%
        of the true size, and the changes everywhere else sum to less than those
        at the bends. The two-spacing window is an observed property of the
        lasso on this data (it smears a bend over its neighbours), not a
        tolerance the feature promises."""
        spec = DesignSpec.from_data(
            planted_linear,
            ["Mileage"],
            linear=["Mileage"],
            knots={"Mileage": self.KNOTS},
            clamp={"Mileage": (0.0, 30_000.0)},
        )
        fit = fit_glm(planted_linear, spec, "ClaimNb", alpha=self.ALPHA, **FIT)
        probe = pl.DataFrame(
            {"Mileage": [0.0, *TRUE_BENDS, 30_000.0], "Exposure": [1.0] * 4}
        )
        log_rel = np.log(fit.predict(probe))
        edges = [0.0, *TRUE_BENDS, 30_000.0]
        for j, true in enumerate(TRUE_SLOPES):
            avg = (log_rel[j + 1] - log_rel[j]) / (edges[j + 1] - edges[j])
            assert abs(avg - true) <= 0.1 * abs(true), (edges[j], avg, true)
        # the curve itself: max |log rel_fitted - log rel_true| on a 100-point grid
        m = np.linspace(0.0, 30_000.0, 100)
        grid = pl.DataFrame({"Mileage": m, "Exposure": np.ones(100)})
        fitted_log = np.log(fit.predict(grid))
        fitted_log -= fitted_log[0]
        s1, s2, s3 = TRUE_SLOPES
        b1, b2 = TRUE_BENDS
        true_log = (
            s1 * np.minimum(m, b1)
            + s2 * np.clip(m - b1, 0, b2 - b1)
            + s3 * np.maximum(m - b2, 0)
        )
        # Measured 0.0525 with this seed; a near-unpenalised fit (alpha 1e-6)
        # gives 0.058, so the gap is sampling noise of the planted book (the
        # first segment's slope comes out 5% low at every penalty), not
        # shrinkage. Tolerance set from that evidence, not from the method.
        assert np.abs(fitted_log - true_log).max() <= 0.06
        enc = spec["Mileage"]
        beta = fit.coef[fit.spec.slices()["Mileage"]][: len(enc.hinges)][1:]  # skip lo
        knots = np.asarray(enc.knots)
        near = np.zeros(len(knots), dtype=bool)
        for bend, before, after in zip(
            TRUE_BENDS, TRUE_SLOPES[:-1], TRUE_SLOPES[1:], strict=True
        ):
            window = np.abs(knots - bend) <= 4_000.0
            near |= window
            change, true_change = beta[window].sum(), after - before
            assert np.sign(change) == np.sign(true_change), (bend, change)
            assert 0.6 <= change / true_change <= 1.4, (bend, change, true_change)
        assert np.abs(beta[~near]).sum() <= np.abs(beta[near]).sum()
        # and the table agrees with the coefficients: slopes are cumulative sums
        tab = rate_tables(fit)["Mileage"]
        bands = tab.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        full = fit.coef[fit.spec.slices()["Mileage"]][: len(enc.hinges)]
        np.testing.assert_allclose(
            bands["slope"].to_numpy(), np.cumsum(full), atol=1e-15
        )

    def test_curve_is_flat_beyond_the_training_range(self, planted_linear):
        spec = DesignSpec.from_data(
            planted_linear,
            ["Mileage"],
            linear=["Mileage"],
            knots={"Mileage": self.KNOTS},
        )
        fit = fit_glm(planted_linear, spec, "ClaimNb", alpha=self.ALPHA, **FIT)
        rm = to_rate_model(fit)
        lo, hi = spec["Mileage"].clamp
        beyond = pl.DataFrame(
            {
                "Mileage": [lo - 5_000.0, lo, hi, hi + 5_000.0, hi + 1e6],
                "Exposure": [1.0] * 5,
            }
        )
        p = rm.predict(beyond, exposure_col=None)
        assert p[0] == pytest.approx(p[1], rel=1e-12)
        assert p[2] == pytest.approx(p[3], rel=1e-12) == pytest.approx(p[4], rel=1e-12)
        np.testing.assert_allclose(p, fit.predict(beyond), rtol=1e-10)
