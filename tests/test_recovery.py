"""Planted-truth tests: a synthetic book with a known effect must be recovered.

Interactions (piece A): a strong ``Age × Region`` cell is planted; the fitted
model must reproduce it, thin non-signal cells must stay at 1.0, and the
A/E-by-pair diagnostic on a model *without* the interaction must expose it.

Piecewise-linear terms (pieces B and B2): a mileage effect that is flat, then
rises, then is flat again is planted; the rate table's band slopes must be
exactly zero on both flat stretches, the rise must be recovered within 10%, the
whole curve must be close to the truth, and it must be flat beyond the training
range. A monotone constraint in the wrong direction must flatten the term
rather than bend it.
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
# pieces B / B2: planted piecewise-linear effect
# --------------------------------------------------------------------------
#: per mile, on the log scale: **flat**, then a rise, then **flat** again. The
#: two flat stretches are what B2 exists for — the basis penalises slopes, so
#: the lasso must return exactly 0 on every band inside them.
TRUE_SLOPES = (0.0, 0.00020, 0.0)
TRUE_BENDS = (8_000.0, 20_000.0)


@pytest.fixture(scope="module")
def planted_linear() -> pl.DataFrame:
    """A book with the planted curve above: flat below 8,000 miles, rising at
    2e-4 per mile to 20,000 (a total rise of 2.4 in log, ~11x), flat after that.

    The size (120k rows) and the base rate (``exp(-0.5)``, ~0.6 claims per
    exposure year) were raised with the B2 basis change so that a single band's
    slope is measurable: with the old book the sampling noise on one 2,000-mile
    band was as large as the planted slope, so no penalty could zero a flat
    stretch without shrinking the sloped one by far more than 10%.

    The flat stretches sit at the two **ends** of the range and the slope in the
    middle, which is where the per-unit-of-rise penalty (see
    ``easy_glm.core.fit.penalty_weights``) is easiest on a fitted slope: an end
    band is a nearly constant column, so it now pays the most for its rise. The
    opposite shape is harder and the numbers are worth recording — with the
    slope at the ends and the flat stretch in the middle, the penalty that
    flattens the middle costs the end slopes 7-13% of their size. Measured over
    eight seeds of *this* book at the penalty below: every one of the nine flat
    bands is exactly 0 and exactly the six bands of the sloped stretch are
    non-zero, on every seed; the recovered slope is 95.4-97.0% of the truth and
    the worst gap to the true curve is 0.141.
    """
    rng = np.random.default_rng(23)
    n = 120_000
    mileage = rng.uniform(0, 30_000, n)
    expo = rng.uniform(0.3, 1.0, n)
    s1, s2, s3 = TRUE_SLOPES
    b1, b2 = TRUE_BENDS
    lp = (
        -0.5
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


@pytest.fixture(scope="module")
def planted_increasing() -> pl.DataFrame:
    """A book whose mileage effect only ever goes up (slopes 2e-4 then 1e-4)."""
    rng = np.random.default_rng(31)
    n = 60_000
    mileage = rng.uniform(0, 30_000, n)
    expo = rng.uniform(0.3, 1.0, n)
    lp = (
        -0.5
        + 2e-4 * np.minimum(mileage, 15_000)
        + 1e-4 * np.maximum(mileage - 15_000, 0)
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
    ALPHA = 0.03

    def test_flat_stretches_are_exactly_flat_and_the_slope_is_recovered(
        self, planted_linear
    ):
        """The point of the B2 basis (actuary's answer to Q3): each coefficient
        is the slope *inside* one band, so the lasso makes a band **exactly
        flat** rather than merely bend-free.

        Asserted here: the rate table's ``slope`` is exactly 0.0 for every one
        of the nine bands inside the planted flat stretches, and non-zero for
        exactly the six bands of the sloped stretch; no band has a slope of the
        wrong sign; the average slope of the sloped stretch is within 10% of the
        truth and the flat stretches are flat end to end; the whole fitted curve
        is within 0.18 of the true log relativity on a 100-point grid (the true
        curve spans 2.40 in log, so that is the ~4% lasso shrinkage the segment
        check also sees — worst measured over eight seeds 0.141); and the
        table's slopes are the model's own coefficients, bit for bit.

        Assertion 1 reads the **table**, not the coefficients, so it is a
        statement about what the actuary sees and it is basis-independent: with
        the old hinge basis the table's slope is a cumulative sum of noisy
        change-of-slope terms and is essentially never exactly zero.

        The old test's "slope changes concentrate at the two true bends" check
        is gone: with this basis there are no change-of-slope coefficients to
        count. Sparsity now lives in the slopes themselves and is asserted
        directly by the exactly-zero flat bands.
        """
        spec = DesignSpec.from_data(
            planted_linear,
            ["Mileage"],
            linear=["Mileage"],
            knots={"Mileage": self.KNOTS},
            clamp={"Mileage": (0.0, 30_000.0)},
        )
        fit = fit_glm(planted_linear, spec, "ClaimNb", alpha=self.ALPHA, **FIT)
        enc = spec["Mileage"]
        beta = fit.coef[fit.spec.slices()["Mileage"]][: enc.n_bands]
        starts = np.asarray(enc.band_starts())
        ends = np.asarray(enc.band_edges()[1:])
        b1, b2 = TRUE_BENDS
        tab = rate_tables(fit)["Mileage"]
        bands = tab.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        slope = bands["slope"].to_numpy()

        # 1. the planted flat stretches come back exactly flat, and only the
        #    bands of the sloped stretch carry a slope
        flat = (ends <= b1) | (starts >= b2)
        sloped = (starts >= b1) & (ends <= b2)
        assert flat.sum() == 9 and sloped.sum() == 6
        assert list(slope[flat]) == [0.0] * 9, slope[flat]
        assert (slope[sloped] != 0.0).all(), slope[sloped]

        # 2. no band has a slope of the wrong sign
        assert (slope >= 0.0).all()

        # 3. the average slope of each stretch
        probe = pl.DataFrame(
            {"Mileage": [0.0, *TRUE_BENDS, 30_000.0], "Exposure": [1.0] * 4}
        )
        log_rel = np.log(fit.predict(probe))
        edges = [0.0, *TRUE_BENDS, 30_000.0]
        for j, true in enumerate(TRUE_SLOPES):
            avg = (log_rel[j + 1] - log_rel[j]) / (edges[j + 1] - edges[j])
            if true == 0.0:
                assert abs(avg) <= 1e-12, avg  # flat bands ⇒ a flat stretch
            else:
                assert abs(avg - true) <= 0.1 * abs(true), (edges[j], avg, true)

        # 4. the curve itself on a 100-point grid
        m = np.linspace(0.0, 30_000.0, 100)
        grid = pl.DataFrame({"Mileage": m, "Exposure": np.ones(100)})
        fitted_log = np.log(fit.predict(grid))
        fitted_log -= fitted_log[0]
        s1, s2, s3 = TRUE_SLOPES
        true_log = (
            s1 * np.minimum(m, b1)
            + s2 * np.clip(m - b1, 0, b2 - b1)
            + s3 * np.maximum(m - b2, 0)
        )
        assert np.abs(fitted_log - true_log).max() <= 0.18

        # 5. beta_j *is* band j's slope — bit for bit, not to a tolerance
        assert list(slope) == list(beta)
        # and each flat stretch is one relativity, repeated
        flat_rows = bands.filter(pl.col("to") <= b1)
        assert flat_rows.height == 4
        assert (
            flat_rows["relativity_to"].to_numpy() == flat_rows["relativity"].to_numpy()
        ).all()

    def test_monotone_in_the_wrong_direction_flattens_the_term(
        self, planted_increasing
    ):
        """B2 re-enables monotone constraints on linear terms as sign bounds on
        the band slopes. On a curve that truly only rises, a *decreasing*
        constraint cannot produce a single positive slope, so the lasso's only
        admissible answer is a flat term; the matching *increasing* constraint
        leaves the recovery untouched."""
        knots = [2_500.0 * k for k in range(1, 12)]

        def _fit(monotone):
            spec = DesignSpec.from_data(
                planted_increasing,
                ["Mileage"],
                linear=["Mileage"],
                knots={"Mileage": knots},
                clamp={"Mileage": (0.0, 30_000.0)},
            )
            fit = fit_glm(
                planted_increasing,
                spec,
                "ClaimNb",
                alpha=0.005,
                monotone=monotone,
                **FIT,
            )
            enc = spec["Mileage"]
            return fit, fit.coef[fit.spec.slices()["Mileage"]][: enc.n_bands]

        _, free = _fit(None)
        assert (free > 0).all()  # unconstrained, the planted rise is recovered

        fit_down, down = _fit({"Mileage": "decreasing"})
        assert not (down > 0).any()  # never a positive slope
        assert list(down) == [0.0] * len(down)  # ... so the term goes flat
        probe = pl.DataFrame(
            {"Mileage": [0.0, 10_000.0, 30_000.0], "Exposure": [1.0] * 3}
        )
        p = fit_down.predict(probe)
        np.testing.assert_allclose(p, p[0], rtol=1e-12)
        tab = rate_tables(fit_down)["Mileage"]
        assert (tab["slope"] == 0.0).all()
        assert np.allclose(tab["relativity"].to_numpy()[:-1], 1.0)

        fit_up, up = _fit({"Mileage": "increasing"})
        assert (up >= 0).all()
        # the bound is not binding, so the fit is the free one (to solver noise)
        np.testing.assert_allclose(up, free, rtol=1e-2, atol=1e-12)
        log_rel = np.log(fit_up.predict(probe))
        assert (
            abs((log_rel[2] - log_rel[0]) / (2e-4 * 15_000 + 1e-4 * 15_000) - 1) < 0.1
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
