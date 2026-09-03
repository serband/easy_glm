"""Piecewise-linear (L-dummy) terms — pieces B and B2.

Contract (docs/RELEASE_0.4_PLAN.md §R2 as revised by R10/Q3): ``LinearEncoder``
clips ``x`` to ``[lo, hi]`` and has one column per band, ``clip(x - k_j, 0,
width_j)``, so each coefficient is the *slope inside that band* and the lasso
zeroes slopes (flat sections). The term is exactly flat outside the clamp,
treats nulls as the value at ``lo`` times a null factor, and its rate table is
log-linear inside each band with relativity 1.00 at ``x_base``. Monotone
constraints are sign bounds on the band slopes and are available for linear
terms. ``kind="continuous"`` is the same encoder with no interior knots.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import polars as pl
import pytest

from easy_glm import (
    DesignSpec,
    LinearEncoder,
    StepEncoder,
    base_rate,
    fit_glm,
    rate_tables,
    to_rate_model,
)
from easy_glm.core.design import linear_encoder_from_data, round_range_outward
from easy_glm.core.excel import rate_model_tables
from easy_glm.core.fit import monotone_bounds
from easy_glm.engine import RateModel
from easy_glm.engine._scoring import row_index as engine_row_index
from easy_glm.engine.models import BandRow, level_label
from easy_glm.ui.metrics import compute_actual_expected

FIT = {"family": "poisson", "weight_col": "Exposure", "divide_target_by_weight": True}


def _book(seed: int = 21, n: int = 12_000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    mileage = rng.uniform(0, 30_000, n)
    age = rng.integers(18, 80, n).astype(float)
    region = rng.choice(["R1", "R2", "R3"], n, p=[0.6, 0.3, 0.1]).astype(object)
    expo = rng.uniform(0.2, 1.0, n)
    # log-linear in mileage with two slope changes at 8,000 and 20,000
    lin = (
        0.00005 * np.minimum(mileage, 8_000)
        + 0.00002 * np.clip(mileage - 8_000, 0, 12_000)
        - 0.00001 * np.maximum(mileage - 20_000, 0)
    )
    mu = np.exp(
        -2.4 + lin - 0.02 * np.maximum(45 - age, 0) + np.where(region == "R1", 0, 0.2)
    )
    claims = rng.poisson(mu * expo).astype(float)
    mileage[rng.random(n) < 0.04] = np.nan
    region[rng.random(n) < 0.03] = None
    return pl.DataFrame(
        {
            "ClaimNb": claims,
            "Exposure": expo,
            "Mileage": mileage,
            "DrivAge": age,
            "Region": pl.Series(
                [None if r is None else str(r) for r in region], dtype=pl.Utf8
            ),
            "logprem": np.log(rng.uniform(200, 900, n)),
        }
    ).with_columns(pl.col("Mileage").fill_nan(None))


@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    return _book()


@pytest.fixture(scope="module")
def spec(book) -> DesignSpec:
    return DesignSpec.from_data(
        book.head(9000),
        ["Mileage", "DrivAge", "Region"],
        linear=["Mileage"],
        knots={"Mileage": [5_000, 8_000, 12_000, 16_000, 20_000, 25_000]},
        min_level_share=0.02,
    )


@pytest.fixture(scope="module")
def fitted(book, spec):
    fit = fit_glm(book.head(9000), spec, "ClaimNb", alpha=0.0005, **FIT)
    return fit, to_rate_model(fit, exposure_col="Exposure")


def _probe(enc: LinearEncoder) -> pl.DataFrame:
    """Values below, at and above both clamps, on knots, inside bands and null."""
    lo, hi = enc.clamp
    xs = [
        lo - 1e6,
        lo - 1.0,
        lo,
        np.nextafter(lo, hi),
        *enc.knots,
        12_345.678,
        hi - 1.0,
        hi,
        hi + 1.0,
        hi + 1e6,
        None,
    ]
    return pl.DataFrame(
        {
            "Mileage": pl.Series(xs, dtype=pl.Float64),
            "DrivAge": [40.0] * len(xs),
            "Region": ["R1"] * len(xs),
            "Exposure": [1.0] * len(xs),
        }
    )


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class TestLinearEncoder:
    def test_features_rows_and_clamp(self):
        enc = LinearEncoder("x", [7.0, 3.0], (1.0, 9.0))
        assert enc.knots == [3.0, 7.0]
        assert enc.band_starts() == [1.0, 3.0, 7.0]
        assert enc.band_widths() == [2.0, 4.0, 2.0] and enc.n_bands == 3
        assert [f.kind for f in enc.features()] == ["band", "band", "band", "null"]
        assert [f.name for f in enc.features()] == [
            "x in [1, 3)",
            "x in [3, 7)",
            "x in [7, 9)",
            "x is null",
        ]
        assert enc.rows() == [
            (None, 1.0),
            (1.0, 3.0),
            (3.0, 7.0),
            (7.0, 9.0),
            (9.0, None),
            (None, None),
        ]
        assert enc.band_edges() == [1.0, 3.0, 7.0, 9.0]

    def test_band_penalty_is_equal_per_unit_of_rise(self, book):
        """B2 review S1: a band's fitted number is a slope, so without a weight a
        wide band that few rows reach buys its *rise* far more cheaply than the
        body of the curve. ``penalty_weights`` equalises the cost per unit of
        rise; the columns are untouched, so the coefficient stays the slope."""
        from easy_glm.core.fit import penalty_weights

        # a skewed factor: a dense body and a wide tail few rows reach, the
        # shape that produced the 89x bonus-malus relativity the review measured
        rng = np.random.default_rng(4)
        x = np.clip(rng.gamma(2.0, 900.0, 20_000), 0.0, 30_000.0)
        skew = pl.DataFrame(
            {"X": x, "Exposure": rng.uniform(0.2, 1.0, x.size), "ClaimNb": 0.0}
        )
        spec = DesignSpec.from_data(
            skew,
            ["X"],
            linear=["X"],
            knots={"X": [2_000.0, 5_000.0]},
            clamp={"X": (0.0, 30_000.0)},
        )
        enc = spec["X"]
        design = spec.build(skew)
        w = skew["Exposure"].to_numpy()
        p1 = penalty_weights(spec, design, w, scale_predictors=True)
        assert p1 is not None
        bands = p1[: enc.n_bands]
        assert p1[enc.n_bands] == 1.0  # the "is null" column is never weighted
        # the wide 5,000-30,000 tail is penalised hardest, by a wide margin
        assert bands.argmax() == enc.n_bands - 1
        assert bands[-1] > 5 * bands[0]
        # cost of one unit of rise = P1 * sd(column) / width: equal everywhere
        widths = np.asarray(enc.band_widths())
        ww = w / w.sum()
        cols = design[:, : enc.n_bands]
        sd = np.sqrt(ww @ (cols**2) - (ww @ cols) ** 2)
        cost = bands * sd / widths
        np.testing.assert_allclose(cost, cost[0], rtol=1e-12)
        # without standardisation the same equality needs widths, not 1/sd
        raw = penalty_weights(spec, design, w, scale_predictors=False)
        np.testing.assert_allclose(
            raw[: enc.n_bands] / widths, raw[0] / widths[0], rtol=1e-12
        )
        assert raw[: enc.n_bands].mean() == pytest.approx(1.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="too close together"):
            LinearEncoder("x", [3.0, 3.0 + 1e-12], (1.0, 9.0))
        assert LinearEncoder("x", [3.0, 3.0], (1.0, 9.0)).knots == [3.0]  # deduped
        with pytest.raises(ValueError, match="strictly inside"):
            LinearEncoder("x", [1.0], (1.0, 9.0))
        with pytest.raises(ValueError, match="lo < hi"):
            LinearEncoder("x", [], (9.0, 1.0))
        with pytest.raises(ValueError, match="finite"):
            LinearEncoder("x", [], (0.0, float("inf")))
        with pytest.raises(ValueError, match="clamp must be"):
            LinearEncoder("x", [], (1.0,))  # type: ignore[arg-type]

    def test_transform_is_the_amount_of_x_inside_each_band(self):
        """Each column is how much of ``x`` falls in that band: 0 below it, the
        band width once ``x`` is past it. Coefficient = slope within the band."""
        enc = LinearEncoder("x", [3.0, 7.0], (1.0, 9.0))
        s = pl.Series([-5.0, 1.0, 2.0, None, 3.0, 8.5, 9.0, 12.0])
        mat = enc.transform(s)
        assert np.isfinite(mat).all()
        np.testing.assert_array_equal(mat[0], mat[1])  # below lo == at lo
        np.testing.assert_array_equal(mat[6], mat[7])  # above hi == at hi
        np.testing.assert_array_equal(mat[3], [0, 0, 0, 1])  # null: bands 0, flag 1
        np.testing.assert_allclose(mat[1], [0.0, 0.0, 0.0, 0])  # at lo: nothing yet
        np.testing.assert_allclose(mat[2], [1.0, 0.0, 0.0, 0])  # 1 unit into band 1
        np.testing.assert_allclose(mat[4], [2.0, 0.0, 0.0, 0])  # band 1 full at x = 3
        np.testing.assert_allclose(mat[5], [2.0, 4.0, 1.5, 0])  # 8.5: 1.5 into band 3
        # every column is capped at its band width, so the row sum is x - lo
        np.testing.assert_allclose(mat[5, :3].sum(), 8.5 - 1.0)
        idx = enc.row_index(s)
        np.testing.assert_array_equal(idx, [0, 1, 1, 5, 2, 3, 4, 4])

    def test_one_band_when_there_are_no_knots(self):
        """``kind="continuous"``: a single slope on the raw clamped value."""
        enc = LinearEncoder("x", [], (0.0, 10.0))
        assert enc.n_bands == 1 and enc.band_edges() == [0.0, 10.0]
        assert [f.name for f in enc.features()] == ["x in [0, 10)", "x is null"]
        mat = enc.transform(pl.Series([-1.0, 0.0, 2.5, 10.0, 99.0, None]))
        np.testing.assert_allclose(mat[:, 0], [0.0, 0.0, 2.5, 10.0, 10.0, 0.0])
        assert enc.rows() == [(None, 0.0), (0.0, 10.0), (10.0, None), (None, None)]

    def test_from_data_rounds_clamp_outward_and_drops_outside_knots(self, book):
        enc = linear_encoder_from_data(
            "Mileage", book["Mileage"], knots=[-5.0, 100.0, 29_999.0, 40_000.0]
        )
        lo, hi = enc.clamp
        assert lo <= book["Mileage"].min() and hi >= book["Mileage"].max()
        assert lo == 0.0 and hi == 30_000.0
        assert enc.knots == [100.0, 29_999.0]
        assert round_range_outward(18.0, 80.0) == (18.0, 80.0)
        assert round_range_outward(0.0037955, 99.9971) == (0.0, 100.0)
        with pytest.raises(ValueError, match="not numeric"):
            DesignSpec.from_data(book, ["Region"], linear=["Region"])
        with pytest.raises(ValueError, match="empty"):
            linear_encoder_from_data("c", pl.Series([2.0, 2.0, 2.0]))

    def test_json_roundtrip_and_repr(self, spec):
        back = DesignSpec.from_dict(spec.to_dict())
        assert back.to_dict() == spec.to_dict()
        assert isinstance(back["Mileage"], LinearEncoder)
        assert "linear(6 knots" in repr(spec)


# --------------------------------------------------------------------------
# fit: slopes, continuity, monotone
# --------------------------------------------------------------------------
class TestFit:
    def test_band_slopes_are_the_coefficients_themselves(self, fitted, spec):
        fit, rm = fitted
        enc = spec["Mileage"]
        coef = fit.coef[fit.spec.slices()["Mileage"]][: enc.n_bands]
        tab = rate_tables(fit)["Mileage"]
        sloped = tab.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        # the basis penalises slopes, so beta_j *is* the slope of band j
        np.testing.assert_allclose(sloped["slope"].to_numpy(), coef, atol=1e-15)
        # flat end rows and the null row have slope 0
        assert (
            tab.filter(pl.col("from").is_null() | pl.col("to").is_null())["slope"] == 0
        ).all()
        # continuity at every interior edge, on the table and in the engine
        rel = sloped["relativity"].to_numpy()
        rel_to = sloped["relativity_to"].to_numpy()
        np.testing.assert_allclose(rel_to[:-1], rel[1:], rtol=1e-12)
        rows = [r for r in rm.variables["Mileage"].table if isinstance(r, BandRow)]
        bands = [r for r in rows if r.from_ is not None and r.to_ is not None]
        for a, b in zip(bands[:-1], bands[1:], strict=True):
            assert a.relativity_to == pytest.approx(b.relativity, rel=1e-12)
        # the below-lo flat row equals the value at lo; the above-hi row the value at hi
        assert rows[0].relativity == pytest.approx(bands[0].relativity, rel=1e-12)
        assert rows[-2].relativity == pytest.approx(bands[-1].relativity_to, rel=1e-12)

    def test_exactness_at_edges_beyond_clamp_and_nulls(self, fitted, spec, book):
        fit, rm = fitted
        probe = _probe(spec["Mileage"])
        assert probe["Mileage"].null_count() == 1
        np.testing.assert_allclose(
            rm.predict(probe, exposure_col=None), fit.predict(probe), rtol=1e-10, atol=0
        )
        # flat outside the clamp: below-lo equals at-lo, above-hi equals at-hi
        p = rm.predict(probe, exposure_col=None)
        assert p[0] == pytest.approx(p[2], rel=1e-12) and p[1] == pytest.approx(
            p[2], rel=1e-12
        )
        assert p[-2] == pytest.approx(p[-4], rel=1e-12) and p[-3] == pytest.approx(
            p[-4], rel=1e-12
        )
        hold = book.tail(3000)
        np.testing.assert_allclose(
            rm.predict(hold, exposure_col=None), fit.predict(hold), rtol=1e-10, atol=0
        )

    def test_base_row_and_base_rate(self, fitted, spec, book):
        fit, rm = fitted
        tab = rate_tables(fit)["Mileage"]
        base = tab.filter(pl.col("is_base"))
        assert base.height == 1 and base["relativity"][0] == pytest.approx(1.0)
        cfg = rm.variables["Mileage"]
        assert cfg.x_base == base["from"][0]  # relativity 1.00 at the lower edge
        # the most exposed band (modal_bins) is the base, as for step tables
        assert fit.modal_bins["Mileage"] == int(np.flatnonzero(tab["is_base"])[0])
        ref = rate_tables(fit, base="reference")["Mileage"]
        assert ref["relativity"][0] == pytest.approx(1.0)  # below-lo row
        rm_ref = to_rate_model(fit, base="reference")
        assert rm_ref.base_rate == pytest.approx(base_rate(fit, base="reference"))
        assert rm_ref.base_rate != pytest.approx(rm.base_rate)
        hold = book.tail(1000)
        np.testing.assert_allclose(
            rm_ref.predict(hold, exposure_col=None),
            rm.predict(hold, exposure_col=None),
            rtol=1e-12,
        )

    def test_engine_row_index_agrees_with_encoder(self, fitted, spec, book):
        _, rm = fitted
        values = pl.concat(
            [book.tail(500)["Mileage"], _probe(spec["Mileage"])["Mileage"]]
        )
        np.testing.assert_array_equal(
            spec["Mileage"].row_index(values),
            engine_row_index(values, rm.variables["Mileage"]),
        )

    def test_monotone_bounds_the_band_slopes_of_a_linear_term(self, spec, book):
        """A monotone constraint on a linear term is a sign bound on every band
        slope — not on the *change* of slope, so the curve is monotone without
        being forced convex. The null column is never bounded."""
        enc = spec["Mileage"]
        sl = spec.slices()["Mileage"]
        lower, upper = monotone_bounds(spec, {"Mileage": "increasing"})
        assert np.all(lower[sl][: enc.n_bands] == 0.0)
        assert np.all(np.isinf(upper[sl]))
        assert lower[sl][enc.n_bands] == -np.inf  # the "is null" column
        lower, upper = monotone_bounds(spec, {"Mileage": "decreasing"})
        assert np.all(upper[sl][: enc.n_bands] == 0.0)
        assert np.all(np.isinf(lower[sl]))
        assert upper[sl][enc.n_bands] == np.inf
        # and it reaches the fit: an increasing constraint gives no negative slope
        fit = fit_glm(
            book.head(4000),
            spec,
            "ClaimNb",
            alpha=0.0005,
            monotone={"Mileage": "increasing"},
            **FIT,
        )
        tab = rate_tables(fit)["Mileage"]
        assert (tab["slope"] >= -1e-14).all()
        rel = tab.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        assert (rel["relativity_to"].to_numpy() >= rel["relativity"].to_numpy()).all()
        # step variables in the same spec are still fine
        lower, upper = monotone_bounds(spec, {"DrivAge": "decreasing"})
        assert np.all(
            upper[spec.slices()["DrivAge"]][: len(spec["DrivAge"].knots)] == 0
        )
        with pytest.raises(ValueError, match="categorical"):
            monotone_bounds(spec, {"Region": "increasing"})


# --------------------------------------------------------------------------
# engine: edits, snapshots, JSON, from_rate_tables, Excel, A/E
# --------------------------------------------------------------------------
class TestEngine:
    def test_band_edit_changes_two_adjacent_slopes_and_keeps_continuity(self, fitted):
        _, rm0 = fitted
        rm = rm0.clone()
        rows = rm.variables["Mileage"].table
        before = [(r.from_, r.to_, r.relativity, r.slope) for r in rows]
        target = rows[3]  # an interior sloped band
        assert target.from_ is not None and target.to_ is not None
        rm.update_relativity("Mileage", target.from_, target.to_, 1.4)
        after = [
            (r.from_, r.to_, r.relativity, r.slope)
            for r in rm.variables["Mileage"].table
        ]
        changed = [
            i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a
        ]
        assert changed == [2, 3]  # previous band's slope and this band (value + slope)
        assert after[3][2] == 1.4 and after[2][2] == before[2][2]
        bands = [
            r
            for r in rm.variables["Mileage"].table
            if r.from_ is not None and r.to_ is not None
        ]
        for a, b in zip(bands[:-1], bands[1:], strict=True):
            assert a.relativity_to == pytest.approx(b.relativity, rel=1e-12)
        # scoring uses the edited curve: value at the band start is exactly 1.4 × mains
        x = pl.DataFrame(
            {
                "Mileage": [target.from_],
                "DrivAge": [40.0],
                "Region": ["R1"],
                "Exposure": [1.0],
            }
        )
        ratio = (
            rm.predict(x, exposure_col=None)[0] / rm0.predict(x, exposure_col=None)[0]
        )
        assert ratio == pytest.approx(1.4 / before[3][2], rel=1e-12)
        # non-positive values are refused
        with pytest.raises(ValueError, match="> 0"):
            rm.update_relativity("Mileage", target.from_, target.to_, 0.0)

    def test_first_band_edit_moves_the_lower_flat_row(self, fitted):
        _, rm0 = fitted
        rm = rm0.clone()
        rows = rm.variables["Mileage"].table
        first = rows[1]
        rm.update_relativity("Mileage", first.from_, first.to_, 0.8)
        rows = rm.variables["Mileage"].table
        assert rows[0].relativity == 0.8 and rows[1].relativity == 0.8
        assert rows[0].slope == 0.0

    def test_every_row_edits_as_a_node_and_the_curve_stays_continuous(self, fitted):
        """Rows: 0 = (None, lo), 1 = first band, 2 = interior, n-3 = last band,
        n-2 = (hi, None), n-1 = null. Editing a row moves its node; only the
        slopes of the bands touching that node change (none for the null row,
        one at either end, two in the interior) and the curve has no jump at
        ``lo`` or ``hi`` afterwards."""
        _, rm0 = fitted
        table = rm0.variables["Mileage"].table
        n = len(table)
        lo, hi = table[1].from_, table[-2].from_
        just_below = [np.nextafter(lo, -np.inf), np.nextafter(hi, -np.inf)]
        probe = pl.DataFrame(
            {
                "Mileage": [just_below[0], lo, just_below[1], hi, None],
                "DrivAge": [40.0] * 5,
                "Region": ["R1"] * 5,
                "Exposure": [1.0] * 5,
            }
        )
        p0 = rm0.predict(probe, exposure_col=None)
        expected = {  # row edited -> (rows whose value changes, slopes that change)
            0: ({0, 1}, {1}),
            1: ({0, 1}, {1}),
            2: ({2}, {1, 2}),
            n - 3: ({n - 3}, {n - 4, n - 3}),
            n - 2: ({n - 2}, {n - 3}),
            n - 1: ({n - 1}, set()),
        }
        moved_at = {0: 1, 1: 1, 2: None, n - 3: None, n - 2: 3, n - 1: 4}
        for idx, (rows_changed, slopes_changed) in expected.items():
            rm = rm0.clone()
            r = table[idx]
            rm.update_relativity("Mileage", r.from_, r.to_, r.relativity * 1.25)
            new = rm.variables["Mileage"].table
            assert {
                i for i in range(n) if new[i].relativity != table[i].relativity
            } == rows_changed, idx
            assert {
                i for i in range(n) if new[i].slope != table[i].slope
            } == slopes_changed, idx
            for i in rows_changed:
                assert new[i].relativity == pytest.approx(table[idx].relativity * 1.25)
            p1 = rm.predict(probe, exposure_col=None)
            assert p1[0] == pytest.approx(p1[1], rel=1e-12), ("jump at lo", idx)
            assert p1[2] == pytest.approx(p1[3], rel=1e-12), ("jump at hi", idx)
            if moved_at[idx] is not None:  # the node's own value moved by 1.25
                j = moved_at[idx]
                assert p1[j] / p0[j] == pytest.approx(1.25, rel=1e-12), idx
            # and the table still reads back as a valid continuous curve
            RateModel.from_rate_tables(rate_model_tables(rm), rm.base_rate)

    def test_null_row_edit_does_not_touch_the_curve(self, fitted):
        _, rm0 = fitted
        rm = rm0.clone()
        rm.update_relativity("Mileage", None, None, 3.0)
        x = _probe(rm.variables["Mileage"] and fitted[0].spec["Mileage"])
        p0, p1 = rm0.predict(x, exposure_col=None), rm.predict(x, exposure_col=None)
        np.testing.assert_allclose(p1[:-1], p0[:-1], rtol=1e-15)
        assert p1[-1] / p0[-1] == pytest.approx(
            3.0 / rm0.variables["Mileage"].table[-1].relativity, rel=1e-12
        )

    def test_snapshot_json_switch_and_diff(self, fitted, tmp_path):
        _, rm0 = fitted
        rm = rm0.clone()
        rows = rm.variables["Mileage"].table
        rm.update_relativity("Mileage", rows[2].from_, rows[2].to_, 1.7)
        v2 = rm.create_snapshot("edit")
        edited = [(r.relativity, r.slope) for r in rm.variables["Mileage"].table]
        rm.to_json(tmp_path / "lin.easyglm")
        back = RateModel.from_json(tmp_path / "lin.easyglm")
        assert [
            (r.relativity, r.slope) for r in back.variables["Mileage"].table
        ] == edited
        assert back.variables["Mileage"].x_base == rm.variables["Mileage"].x_base
        back.switch_to(1)
        assert [(r.relativity, r.slope) for r in back.variables["Mileage"].table] == [
            (r.relativity, r.slope) for r in rm0.variables["Mileage"].table
        ]
        back.switch_to(v2)
        assert [
            (r.relativity, r.slope) for r in back.variables["Mileage"].table
        ] == edited
        changes = rm.diff(1, v2)
        assert [c.variable for c in changes] == ["Mileage"]
        assert changes[0].new_relativity == 1.7

    def test_from_rate_tables_roundtrip_excel_and_errors(self, fitted, tmp_path):
        fit, rm0 = fitted
        rm = rm0.clone()
        rows = rm.variables["Mileage"].table
        rm.update_relativity("Mileage", rows[3].from_, rows[3].to_, 1.4)
        tables = rate_model_tables(rm)
        assert {"slope", "relativity_to"} <= set(tables["Mileage"].columns)
        rebuilt = RateModel.from_rate_tables(tables, rm.base_rate)
        assert rebuilt.variables["Mileage"].type == "linear"
        hold = _book(seed=99, n=2000)
        np.testing.assert_allclose(
            rebuilt.predict(hold, exposure_col=None),
            rm.predict(hold, exposure_col=None),
            rtol=1e-12,
        )
        # through Excel
        path = rm.to_excel(tmp_path / "lin.xlsx")
        sheets = pl.read_excel(path, sheet_id=0)
        back = RateModel.from_rate_tables(
            {v: sheets[v] for v in rm.variables}, rm.base_rate
        )
        np.testing.assert_allclose(
            back.predict(hold, exposure_col=None),
            rm.predict(hold, exposure_col=None),
            rtol=1e-9,
        )
        # fresh fit: rate_tables(fit) -> from_rate_tables == to_rate_model
        direct = RateModel.from_rate_tables(rate_tables(fit), base_rate(fit))
        np.testing.assert_allclose(
            direct.predict(hold, exposure_col=None),
            rm0.predict(hold, exposure_col=None),
            rtol=1e-12,
        )
        # broken tables
        t = tables["Mileage"]
        broken = t.with_columns(
            pl.when(pl.arange(0, t.height) == 3)
            .then(pl.col("relativity") * 1.5)
            .otherwise(pl.col("relativity"))
            .alias("relativity")
        )
        # an interior start value that no longer matches the slope column: the
        # values win, the slope column is reported
        with pytest.warns(UserWarning, match="slope column"):
            rb = RateModel.from_rate_tables({"Mileage": broken}, 1.0)
        rows = rb.variables["Mileage"].table
        assert rows[3].relativity == pytest.approx(t["relativity"][3] * 1.5)
        assert rows[2].relativity_to == pytest.approx(rows[3].relativity, rel=1e-12)
        assert rows[3].relativity_to == pytest.approx(rows[4].relativity, rel=1e-12)
        with pytest.raises(ValueError, match="slope 0"):
            RateModel.from_rate_tables(
                {
                    "Mileage": t.with_columns(
                        pl.when(pl.col("from").is_null() & pl.col("to").is_not_null())
                        .then(0.1)
                        .otherwise(pl.col("slope"))
                        .alias("slope")
                    )
                },
                1.0,
            )
        with pytest.raises(ValueError, match="> 0"):
            RateModel.from_rate_tables(
                {"Mileage": t.with_columns(pl.lit(-1.0).alias("relativity"))}, 1.0
            )

    def test_from_rate_tables_shapes_rounding_and_x_base(self, fitted, tmp_path):
        fit, rm0 = fitted
        t = rate_model_tables(rm0)["Mileage"]
        hold = _book(seed=7, n=1500)
        n = t.height
        # a table rounded the way a rate manual prints it loads and is continuous
        rounded = t.with_columns(
            pl.col("relativity").round(4),
            pl.col("relativity_to").round(4),
            pl.col("slope").round(6),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rb = RateModel.from_rate_tables(
                {**rate_model_tables(rm0), "Mileage": rounded}, rm0.base_rate
            )
        rows = rb.variables["Mileage"].table
        for a, b in zip(rows[1:-2], rows[2:-1], strict=True):
            assert a.relativity_to == pytest.approx(b.relativity, rel=1e-12)
        np.testing.assert_allclose(
            rb.predict(hold, exposure_col=None),
            rm0.predict(hold, exposure_col=None),
            rtol=3e-4,
        )
        assert rb.variables["Mileage"].x_base == rm0.variables["Mileage"].x_base
        # x_base: from the is_base column, else from the unique 1.0 row
        assert rm0.variables["Mileage"].x_base is not None
        assert (
            RateModel.from_rate_tables(rate_tables(fit), 1.0)
            .variables["Mileage"]
            .x_base
            == to_rate_model(fit).variables["Mileage"].x_base
        )
        no_flag = RateModel.from_rate_tables({"Mileage": t.drop("is_base")}, 1.0)
        assert no_flag.variables["Mileage"].x_base == rm0.variables["Mileage"].x_base
        path = rm0.to_excel(tmp_path / "xb.xlsx")
        sheets = pl.read_excel(path, sheet_id=0)
        via_excel = RateModel.from_rate_tables(
            {v: sheets[v] for v in rm0.variables}, 1.0
        )
        assert via_excel.variables["Mileage"].x_base == rm0.variables["Mileage"].x_base
        assert "x_base (Mileage)" in sheets["Summary"].to_series(0).to_list()
        # editing the base band keeps x_base (it names the point, not the value)
        rm = rm0.clone()
        xb = rm.variables["Mileage"].x_base
        base_row = next(r for r in rm.variables["Mileage"].table if r.from_ == xb)
        rm.update_relativity("Mileage", base_row.from_, base_row.to_, 1.3)
        assert rm.variables["Mileage"].x_base == xb
        rm.to_json(tmp_path / "xb.easyglm")
        assert (
            RateModel.from_json(tmp_path / "xb.easyglm").variables["Mileage"].x_base
            == xb
        )
        # without slope: refused when relativity_to shows it was linear, a
        # step table on purpose when both are gone
        with pytest.raises(ValueError, match="needs its slopes"):
            RateModel.from_rate_tables({"Mileage": t.drop("slope")}, 1.0)
        as_steps = RateModel.from_rate_tables(
            {"Mileage": t.drop("slope", "relativity_to", "is_base")}, 1.0
        )
        assert as_steps.variables["Mileage"].type == "numeric"
        # zero-width band
        k = float(t["to"][1])
        zero = pl.concat(
            [
                t.head(2),
                t.head(2).tail(1).with_columns(pl.lit(k).alias("from")),
                t.tail(n - 2),
            ]
        )
        with pytest.raises(ValueError, match="zero width"):
            RateModel.from_rate_tables({"Mileage": zero}, 1.0)
        # a cliff at lo: the '< lo' row and the first band must agree
        at_lo = t.with_columns(
            pl.when(pl.arange(0, n) == 0)
            .then(pl.col("relativity") * 1.5)
            .otherwise(pl.col("relativity"))
            .alias("relativity")
        )
        with pytest.raises(ValueError, match="not continuous at"):
            RateModel.from_rate_tables({"Mileage": at_lo}, 1.0)
        # a changed (hi, None) value: the last slope is re-derived towards it
        # (the stale slope column is reported), so there is no cliff at hi
        at_hi = t.with_columns(
            pl.when(pl.arange(0, n) == n - 2)
            .then(pl.col("relativity") * 1.5)
            .otherwise(pl.col("relativity"))
            .alias("relativity")
        )
        with pytest.warns(UserWarning, match="slope column"):
            rh = RateModel.from_rate_tables({"Mileage": at_hi}, 1.0)
        hi = rh.variables["Mileage"].table[-2].from_
        x = pl.DataFrame({"Mileage": [np.nextafter(hi, -np.inf), hi]})
        p = rh.predict(x, exposure_col=None)
        assert p[0] == pytest.approx(p[1], rel=1e-12)
        # missing null row: warned, nulls raise at scoring
        with pytest.warns(UserWarning, match="no null row"):
            nn = RateModel.from_rate_tables({"Mileage": t.head(n - 1)}, 1.0)
        with pytest.raises(ValueError, match="NaN"):
            nn.predict(pl.DataFrame({"Mileage": [None, 1.0]}), exposure_col=None)

    def test_inf_and_integer_input_score_like_the_glm(self, fitted):
        fit, rm = fitted
        x = pl.DataFrame(
            {
                "Mileage": [np.inf, -np.inf, 1e308, -1e308, 12_000.0],
                "DrivAge": [40.0] * 5,
                "Region": ["R1"] * 5,
                "Exposure": [1.0] * 5,
            }
        )
        p = rm.predict(x, exposure_col=None)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p, fit.predict(x), rtol=1e-10)
        assert p[0] == p[2] and p[1] == p[3]
        ints = _book(seed=3, n=500).with_columns(
            pl.col("Mileage").round(0).cast(pl.Int64)
        )
        np.testing.assert_allclose(
            rm.predict(ints, exposure_col=None), fit.predict(ints), rtol=1e-10
        )

    def test_json_orders_and_validates_linear_rows(self, fitted):
        _, rm0 = fitted
        raw = rm0._to_dict()
        tab = raw["variables"]["Mileage"]["table"]
        raw["variables"]["Mileage"]["table"] = tab[-1:] + tab[:-1][::-1]
        back = RateModel._from_dict(raw)
        assert [(r.from_, r.to_) for r in back.variables["Mileage"].table] == [
            (r.from_, r.to_) for r in rm0.variables["Mileage"].table
        ]
        hold = _book(seed=11, n=300)
        np.testing.assert_allclose(
            back.predict(hold, exposure_col=None), rm0.predict(hold, exposure_col=None)
        )
        first = back.variables["Mileage"].table[1]
        back.update_relativity(
            "Mileage", first.from_, first.to_, 0.9
        )  # neighbours right
        assert back.variables["Mileage"].table[0].relativity == 0.9
        for row in raw["variables"]["Mileage"]["table"]:
            row.pop("slope")
        with pytest.raises(ValueError, match="needs a 'slope'"):
            RateModel._from_dict(raw)

    def test_null_row_is_never_the_base_of_a_linear_term(self, book):
        mostly_null = book.head(6000).with_columns(
            pl.when(pl.arange(0, 6000) % 10 < 7)
            .then(None)
            .otherwise(pl.col("Mileage"))
            .alias("Mileage")
        )
        spec = DesignSpec.from_data(
            mostly_null, ["Mileage", "DrivAge"], linear=["Mileage"], n_bins=6
        )
        fit = fit_glm(mostly_null, spec, "ClaimNb", alpha=0.001, **FIT)
        tab = rate_tables(fit)["Mileage"]
        base = tab.filter(pl.col("is_base"))
        assert base.height == 1 and base["from"][0] is not None
        rm = to_rate_model(fit)
        assert rm.variables["Mileage"].x_base == base["from"][0]
        np.testing.assert_allclose(
            rm.predict(mostly_null, exposure_col=None),
            fit.predict(mostly_null),
            rtol=1e-10,
        )

    def test_round_outward_clamp_extends_the_end_bands_at_their_slope(self, book):
        shifted = book.head(6000).with_columns(
            pl.when(pl.arange(0, 6000) == 0)
            .then(29_857.0)
            .when(pl.arange(0, 6000) == 1)
            .then(17.65)
            .otherwise((pl.col("Mileage") * 0.99 + 17.65).clip(17.65, 29_857.0))
            .alias("Mileage")
        )
        spec = DesignSpec.from_data(
            shifted, ["Mileage"], linear=["Mileage"], knots={"Mileage": [8000, 20000]}
        )
        enc = spec["Mileage"]
        assert enc.clamp == (0.0, 29_900.0)  # < 1% of the range at either end
        fit = fit_glm(shifted, spec, "ClaimNb", alpha=0.0005, **FIT)
        rm = to_rate_model(fit)
        last = rm.variables["Mileage"].table[-3]
        assert (last.from_, last.to_) == (20_000.0, 29_900.0)
        x = pl.DataFrame({"Mileage": [29_857.0, 29_900.0, 0.0, 17.65]})
        p = rm.predict(x, exposure_col=None)
        assert p[1] / p[0] == pytest.approx(np.exp(last.slope * 43.0), rel=1e-12)
        first = rm.variables["Mileage"].table[1]
        assert p[3] / p[2] == pytest.approx(np.exp(first.slope * 17.65), rel=1e-12)

    def test_labels_and_actual_expected(self, fitted, book):
        fit, rm = fitted
        rows = rm.variables["Mileage"].table
        assert (
            level_label(rows[0]).startswith("<")
            and level_label(rows[-1]) == "Other / Unknown"
        )
        data = book.tail(3000).with_columns(
            (pl.col("ClaimNb") / pl.col("Exposure")).alias("ClaimNb")
        )
        ae = compute_actual_expected(rm, data, "Mileage", formula="sum_over_weight")[
            "subsets"
        ]["all"]
        assert len(ae) == len(rows)
        assert sum(r["exposure"] for r in ae) == pytest.approx(data["Exposure"].sum())
        assert ae[-1]["exposure"] > 0  # the null row has exposure
        # the (None, lo) and (hi, None) rows have none in-sample (clamp = data range)
        assert ae[0]["exposure"] == 0.0 and ae[-2]["exposure"] == 0.0

    def test_interaction_with_a_linear_parent(self, book):
        train = book.head(9000)
        spec = DesignSpec.from_data(
            train,
            ["Mileage", "Region"],
            linear=["Mileage"],
            knots={"Mileage": [10_000, 20_000]},
            min_level_share=0.02,
            interactions=[("Mileage", "Region")],
            min_cell_exposure=0.005,
        )
        fit = fit_glm(train, spec, "ClaimNb", alpha=0.0005, **FIT)
        rm = to_rate_model(fit)
        hold = book.tail(3000).with_columns(
            pl.when(pl.arange(0, 3000) % 40 == 0)
            .then(pl.lit("NEW"))
            .otherwise(pl.col("Region"))
            .alias("Region")
        )
        np.testing.assert_allclose(
            rm.predict(hold, exposure_col=None), fit.predict(hold), rtol=1e-10
        )
        assert "Mileage×Region" in rm.variables
        cells = rm.variables["Mileage×Region"].table
        assert all(isinstance(c.from_a, float) or c.from_a is None for c in cells)


# --------------------------------------------------------------------------
# workflow: project spec, validation, script round trip
# --------------------------------------------------------------------------
class TestWorkflow:
    @pytest.fixture
    def project(self, book, tmp_path):
        from easy_glm.workflow import Project, VariableDesign

        path = tmp_path / "book.parquet"
        book.write_parquet(path)
        p = Project(name="lin")
        p.data.source.type = "parquet"
        p.data.source.path = str(path)
        p.data.roles = {
            "ClaimNb": "target",
            "Exposure": "weight",
            "Mileage": "predictor",
            "DrivAge": "predictor",
            "Region": "predictor",
            "logprem": "ignore",
        }
        p.data.split.mode = "random"
        p.data.split.seed = 5
        p.design.variables["Mileage"] = VariableDesign(
            kind="linear", knots=[8_000.0, 20_000.0]
        )
        p.design.variables["DrivAge"] = VariableDesign(monotone="decreasing")
        p.new_model(
            "freq",
            divide_target_by_weight=True,
            predictors=["Mileage", "DrivAge", "Region"],
        )
        p.models["freq"].penalty.alpha = 0.0005
        p.models["freq"].penalty.cv = None
        return p

    def test_validate_accepts_monotone_on_linear_and_checks_the_clamp(self, project):
        from easy_glm.workflow import VariableDesign

        assert project.validate() == []
        project.design.variables["Mileage"] = VariableDesign(
            kind="linear", knots=[8_000.0], monotone="increasing"
        )
        assert project.validate() == []  # B2: allowed, as a bound on the slopes
        project.models["freq"].monotone = {"Mileage": "decreasing"}
        assert project.validate("freq") == []
        project.models["freq"].monotone = {}
        project.design.variables["Mileage"] = VariableDesign(
            kind="continuous", monotone="increasing"
        )
        assert project.validate() == []
        project.design.variables["Mileage"] = VariableDesign(kind="wobbly")
        assert any("kind must be" in p for p in project.validate())
        project.design.variables["Mileage"] = VariableDesign(
            kind="linear", clamp=[5.0, 1.0]
        )
        assert any("clamp" in p for p in project.validate())

    def test_continuous_kind_is_a_one_band_linear_term(self, project, book):
        from easy_glm.workflow import (
            VariableDesign,
            build_design,
            prepare,
            run_model,
            to_script,
        )

        project.design.variables["Mileage"] = VariableDesign(kind="continuous")
        df = prepare(project)
        train = df.filter(pl.col("traintest") == 1)
        spec = build_design(project, train, ["Mileage"], weight_col="Exposure")
        enc = spec["Mileage"]
        assert isinstance(enc, LinearEncoder)
        assert enc.knots == [] and enc.n_bands == 1
        # the quantile knots that a "linear" term would have used are ignored
        assert enc.n_features == 2  # one slope + the null column
        run = run_model(project, df, "freq")
        cfg = run.rate_model.variables["Mileage"]
        assert cfg.type == "linear"  # same table type, editor and Excel sheet
        assert len(cfg.table) == 4  # < lo, the single band, >= hi, null
        hold = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            run.rate_model.predict(hold, exposure_col=None),
            run.fit.predict(hold),
            rtol=1e-10,
        )
        # the exported script round-trips the one-band design both ways
        src = to_script(project, "freq", run=run, output_prefix="cont_v1")
        assert "LinearEncoder('Mileage', [], clamp=(" in src
        no_run = to_script(project, "freq")
        assert "linear=['Mileage']" in no_run and "knots={'Mileage': []}" in no_run

    def test_continuous_exported_scripts_rebuild_the_model(self, project, tmp_path):
        """Both scripts are *executed*, with a run (explicit encoders) and
        without one (``from_data(linear=..., knots={var: []})``); each must
        rebuild a four-row Mileage table that scores like the workbench."""
        from easy_glm.workflow import VariableDesign, prepare, run_model, to_script

        project.design.variables["Mileage"] = VariableDesign(
            kind="continuous", monotone="increasing"
        )
        df = prepare(project)
        run = run_model(project, df, "freq")
        hold = df.filter(pl.col("traintest") == 0)
        for tag, src in (
            ("with_run", to_script(project, "freq", run=run, output_prefix="cont_run")),
            ("no_run", to_script(project, "freq", output_prefix="cont_norun")),
        ):
            assert "monotone={" in src and "'Mileage': 'increasing'" in src, tag
            script = tmp_path / f"rebuild_{tag}.py"
            script.write_text(src)
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc.returncode == 0, (tag, proc.stderr[-2000:])
            prefix = "cont_run" if tag == "with_run" else "cont_norun"
            rebuilt = RateModel.from_json(tmp_path / f"{prefix}.easyglm")
            table = rebuilt.variables["Mileage"].table
            assert len(table) == 4, tag  # still one band
            assert all(r.slope >= 0 for r in table), tag  # constraint reproduced
            if tag == "with_run":  # the no-run script re-derives its own clamp
                np.testing.assert_allclose(
                    rebuilt.predict(hold, exposure_col=None),
                    run.predict(hold),
                    rtol=1e-10,
                )

    def test_continuous_table_survives_excel_and_from_rate_tables(self, project):
        """A one-band table is the shape most likely to fall through the cracks
        of the linear reader (two flat rows, one band, the null row)."""
        from easy_glm.workflow import VariableDesign, prepare, run_model

        project.design.variables["Mileage"] = VariableDesign(kind="continuous")
        df = prepare(project)
        run = run_model(project, df, "freq")
        rm = run.rate_model
        sheets = rate_model_tables(rm)
        assert set(sheets["Mileage"].columns) >= {"slope", "relativity_to", "is_base"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "missing null row" style warnings
            back = RateModel.from_rate_tables(sheets, base_rate=rm.base_rate)
        assert back.variables["Mileage"].x_base == rm.variables["Mileage"].x_base
        hold = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            back.predict(hold, exposure_col=None),
            rm.predict(hold, exposure_col=None),
            rtol=1e-12,
        )

    def test_base_point_of_a_one_band_term_follows_the_exposure(self, book):
        """Q2 for a continuous term: with a single band the 1.00 point is the
        clamp the bulk of the business is nearer to, not always the lower one."""
        from easy_glm.core.fit import weighted_median

        low = book.with_columns((pl.col("Mileage") * 0.2).alias("Mileage"))
        high = book.with_columns((30_000.0 - pl.col("Mileage") * 0.2).alias("Mileage"))
        seen = []
        for frame in (low, high):
            spec = DesignSpec.from_data(
                frame,
                ["Mileage"],
                linear=["Mileage"],
                knots={"Mileage": []},
                clamp={"Mileage": (0.0, 30_000.0)},
            )
            fit = fit_glm(frame, spec, "ClaimNb", alpha=0.0005, **FIT)
            rm = to_rate_model(fit)
            cfg = rm.variables["Mileage"]
            tab = rate_tables(fit)["Mileage"]
            base = tab.filter(pl.col("is_base"))
            assert base.height == 1
            assert base["relativity"][0] == pytest.approx(1.0)
            assert cfg.x_base == base["from"][0]
            seen.append(cfg.x_base)
        assert seen == [0.0, 30_000.0]
        # the rule itself: the weighted median against the middle of the range
        med = weighted_median(high["Mileage"].to_numpy(), high["Exposure"].to_numpy())
        assert med > 15_000.0
        assert weighted_median(np.array([np.nan, np.nan]), np.ones(2)) != med

    def test_a_flattened_term_keeps_its_base_point_and_edits_as_nodes(self, book, spec):
        """A monotone constraint in the wrong direction leaves every slope 0 and
        every relativity 1.0000, so there is no unique 1.0 row to recover the
        base from — the ``is_base`` column has to carry it. That table is a
        reachable workbench state, so pin it."""
        fit = fit_glm(
            book.head(9000),
            spec,
            "ClaimNb",
            alpha=0.02,
            monotone={"Mileage": "decreasing"},
            **FIT,
        )
        tab = rate_tables(fit)["Mileage"]
        assert (tab["slope"] == 0.0).all()
        np.testing.assert_allclose(tab["relativity"].to_numpy()[:-1], 1.0)
        rm = to_rate_model(fit)
        x_base = rm.variables["Mileage"].x_base
        assert x_base is not None
        back = RateModel.from_rate_tables(rate_model_tables(rm), base_rate=rm.base_rate)
        assert back.variables["Mileage"].x_base == x_base
        # and an edit on one of the flat bands still moves exactly two slopes
        edited = rm.clone()
        rows = edited.variables["Mileage"].table
        before = [(r.relativity, r.slope) for r in rows]
        target = rows[3]
        edited.update_relativity("Mileage", target.from_, target.to_, 1.2)
        after = edited.variables["Mileage"].table
        moved = [i for i, r in enumerate(after) if r.slope != before[i][1]]
        assert moved == [2, 3]
        eps = np.nextafter(float(target.from_), -np.inf)
        probe = pl.DataFrame(
            {
                "Mileage": [eps, float(target.from_)],
                "DrivAge": [40.0, 40.0],
                "Region": ["R1", "R1"],
                "Exposure": [1.0, 1.0],
            }
        )
        p = edited.predict(probe, exposure_col=None)
        assert p[0] == pytest.approx(p[1], rel=1e-12)

    def test_monotone_on_a_linear_interaction_parent_binds_only_the_main(self, book):
        """The bound is on the factor's own band slopes; an interaction cell on
        top of it is not constrained (true for step terms too). Recorded so the
        behaviour is a decision, not an accident."""
        spec = DesignSpec.from_data(
            book.head(9000),
            ["Mileage", "Region"],
            linear=["Mileage"],
            knots={"Mileage": [8_000.0, 20_000.0]},
            min_level_share=0.02,
            interactions=[("Mileage", "Region")],
            min_cell_exposure=0.0,
        )
        lower, upper = monotone_bounds(spec, {"Mileage": "increasing"})
        sl = spec.slices()["Mileage"]
        enc = spec["Mileage"]
        assert np.all(lower[sl][: enc.n_bands] == 0.0)
        cells = spec.slices()["Mileage×Region"]
        assert np.all(np.isinf(lower[cells])) and np.all(np.isinf(upper[cells]))
        fit = fit_glm(
            book.head(9000),
            spec,
            "ClaimNb",
            alpha=0.0005,
            monotone={"Mileage": "increasing"},
            **FIT,
        )
        assert (rate_tables(fit)["Mileage"]["slope"] >= -1e-14).all()

    def test_monotone_on_a_continuous_term_runs_end_to_end(self, project):
        from easy_glm.workflow import VariableDesign, prepare, run_model

        project.design.variables["Mileage"] = VariableDesign(
            kind="continuous", monotone="increasing"
        )
        assert project.validate() == []
        df = prepare(project)
        run = run_model(project, df, "freq")
        assert run.fit.monotone["Mileage"] == "increasing"
        slopes = run.tables["Mileage"]["slope"].to_numpy()
        assert (slopes >= -1e-14).all()

    def test_build_design_clamp_and_integer_knots(self, project, book):
        from easy_glm.workflow import VariableDesign, build_design, prepare

        df = prepare(project)
        train = df.filter(pl.col("traintest") == 1)
        spec = build_design(
            project, train, ["Mileage", "DrivAge"], weight_col="Exposure"
        )
        enc = spec["Mileage"]
        assert isinstance(enc, LinearEncoder) and enc.knots == [8_000.0, 20_000.0]
        assert isinstance(spec["DrivAge"], StepEncoder)
        project.design.variables["Mileage"] = VariableDesign(
            kind="linear", clamp=[100.0, 25_000.0]
        )
        spec = build_design(project, train, ["Mileage"], weight_col="Exposure")
        assert spec["Mileage"].clamp == (100.0, 25_000.0)
        project.design.variables["DrivAge"] = VariableDesign(
            kind="linear", knots="integer"
        )
        spec = build_design(project, train, ["DrivAge"], weight_col="Exposure")
        assert isinstance(spec["DrivAge"], LinearEncoder)
        assert spec["DrivAge"].knots[:3] == [19.0, 20.0, 21.0]

    def test_run_model_hash_and_exported_script(self, project, tmp_path):
        from easy_glm.workflow import Adjustment, prepare, run_model, to_script

        df = prepare(project)
        run = run_model(project, df, "freq")
        assert run.rate_model.variables["Mileage"].type == "linear"
        assert run.fit.monotone == {"DrivAge": "decreasing"}
        band = run.rate_model.variables["Mileage"].table[2]
        project.models["freq"].adjustments.append(
            Adjustment("Mileage", band.from_, band.to_, 1.3)
        )
        run = run_model(project, df, "freq")
        edited = next(
            r
            for r in run.rate_model.variables["Mileage"].table
            if r.from_ == band.from_
        )
        assert edited.relativity == 1.3
        src = to_script(project, "freq", run=run, output_prefix="lin_v1")
        assert "LinearEncoder('Mileage', [8000, 20000], clamp=(" in src
        assert (
            f"rm.update_relativity('Mileage', {band.from_!r}, {band.to_!r}, 1.3)" in src
        )
        script = tmp_path / "rebuild.py"
        script.write_text(src)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        rebuilt = RateModel.from_json(tmp_path / "lin_v1.easyglm")
        hold = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            rebuilt.predict(hold, exposure_col=None), run.predict(hold), rtol=1e-10
        )
        # the no-run script derives the linear design from the data
        assert "linear=['Mileage']" in to_script(project, "freq")
        project.design.variables["Mileage"].clamp = [100.0, 25_000.0]
        no_run = to_script(project, "freq")
        assert "clamp={'Mileage': (100.0, 25000.0)}" in no_run
        assert "knots={'Mileage': [8000.0, 20000.0]}" in no_run

    def test_apply_adjustments_names_the_refused_entry(self, project, book):
        from easy_glm.workflow import Adjustment, AdjustmentError
        from easy_glm.workflow.run import apply_adjustments, build_design

        spec = build_design(
            project,
            book.head(3000),
            ["Mileage", "DrivAge", "Region"],
            weight_col="Exposure",
        )
        fit = fit_glm(book.head(3000), spec, "ClaimNb", alpha=0.001, **FIT)
        rm = to_rate_model(fit)
        cfg = project.models["freq"]
        bad = Adjustment("Mileage", 8_000.0, 20_000.0, 0.0)
        cfg.adjustments = [Adjustment("DrivAge", None, None, 1.1), bad]
        with pytest.raises(AdjustmentError, match="must be > 0") as info:
            apply_adjustments(rm, cfg)
        assert info.value.adjustment is bad

    def test_app_pages_render_with_a_linear_term(self, project, tmp_path):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        ppath = tmp_path / "lin.easyglm-project.json"
        project.to_json(ppath)
        for page in [
            "pages_design",
            "pages_model",
            "pages_tables",
            "pages_diagnostics",
            "pages_export",
        ]:
            script = f"""
import importlib, streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project
S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(ppath)!r}), None); st.session_state._loaded = True
if S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app." + {page!r}).render()
"""
            at = AppTest.from_string(script, default_timeout=180)
            at.run()
            assert not at.exception, (page, [e.value for e in at.exception])

    @staticmethod
    def _page_script(ppath, page: str, tail: str = "") -> str:
        return f"""
import importlib, streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project
S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(ppath)!r}), None); st.session_state._loaded = True
if S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app." + {page!r}).render()
{tail}
"""

    def test_design_page_keeps_a_user_clamp_across_renders(self, project, tmp_path):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        project.design.variables["Mileage"].clamp = [100.0, 25_000.0]
        ppath = tmp_path / "clamp.easyglm-project.json"
        project.to_json(ppath)
        tail = (
            'st.session_state.vd_after = S.project().design.variables["Mileage"]'
            ".__dict__.copy()"
        )
        at = AppTest.from_string(
            self._page_script(ppath, "pages_design", tail), default_timeout=180
        )
        at.run()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        after = at.session_state["vd_after"]
        assert after["clamp"] == [100.0, 25_000.0]
        assert after["knots"] == [8_000.0, 20_000.0] and after["kind"] == "linear"

    def test_tables_page_survives_a_zero_adjustment_on_a_linear_band(
        self, project, tmp_path
    ):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        from easy_glm.workflow import Adjustment

        project.models["freq"].adjustments = [
            Adjustment("Mileage", 8_000.0, 20_000.0, 0.0),
            Adjustment("DrivAge", None, None, 1.1),
        ]
        ppath = tmp_path / "zero.easyglm-project.json"
        project.to_json(ppath)
        tail = 'st.session_state.adj_after = S.project().models["freq"].adjustments'
        at = AppTest.from_string(
            self._page_script(ppath, "pages_tables", tail), default_timeout=180
        )
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("must be > 0" in e.value for e in at.error)
        left = at.session_state["adj_after"]
        assert [(a.variable, a.relativity) for a in left] == [("DrivAge", 1.1)]
        at.run()  # and the page keeps rendering afterwards
        assert not at.exception
