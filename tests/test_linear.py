"""Piecewise-linear (L-dummy) terms — piece B.

Contract (docs/RELEASE_0.4_PLAN.md §R2): ``LinearEncoder`` clips ``x`` to
``[lo, hi]``, has a hinge at ``lo`` and at every interior knot, is exactly flat
outside the clamp, treats nulls as the value at ``lo`` times a null factor,
and its rate table is log-linear inside each band with relativity 1.00 at
``x_base``; monotone constraints are refused.
"""

from __future__ import annotations

import subprocess
import sys

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
            "Region": region,
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
        assert enc.knots == [3.0, 7.0] and enc.hinges == [1.0, 3.0, 7.0]
        assert [f.kind for f in enc.features()] == ["hinge", "hinge", "hinge", "null"]
        assert enc.rows() == [
            (None, 1.0),
            (1.0, 3.0),
            (3.0, 7.0),
            (7.0, 9.0),
            (9.0, None),
            (None, None),
        ]
        assert enc.band_edges() == [1.0, 3.0, 7.0, 9.0]

    def test_validation(self):
        with pytest.raises(ValueError, match="strictly inside"):
            LinearEncoder("x", [1.0], (1.0, 9.0))
        with pytest.raises(ValueError, match="lo < hi"):
            LinearEncoder("x", [], (9.0, 1.0))
        with pytest.raises(ValueError, match="finite"):
            LinearEncoder("x", [], (0.0, float("inf")))
        with pytest.raises(ValueError, match="clamp must be"):
            LinearEncoder("x", [], (1.0,))  # type: ignore[arg-type]

    def test_transform_is_clipped_finite_and_null_aware(self):
        enc = LinearEncoder("x", [3.0, 7.0], (1.0, 9.0))
        s = pl.Series([-5.0, 1.0, 2.0, None, 3.0, 8.5, 9.0, 12.0])
        mat = enc.transform(s)
        assert np.isfinite(mat).all()
        np.testing.assert_array_equal(mat[0], mat[1])  # below lo == at lo
        np.testing.assert_array_equal(mat[6], mat[7])  # above hi == at hi
        np.testing.assert_array_equal(mat[3], [0, 0, 0, 1])  # null: hinges 0, flag 1
        np.testing.assert_allclose(mat[5], [7.5, 5.5, 1.5, 0])
        idx = enc.row_index(s)
        np.testing.assert_array_equal(idx, [0, 1, 1, 5, 2, 3, 4, 4])

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
    def test_slopes_are_cumulative_hinge_coefficients(self, fitted, spec):
        fit, rm = fitted
        enc = spec["Mileage"]
        coef = fit.coef[fit.spec.slices()["Mileage"]][: len(enc.hinges)]
        tab = rate_tables(fit)["Mileage"]
        sloped = tab.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        np.testing.assert_allclose(
            sloped["slope"].to_numpy(), np.cumsum(coef), atol=1e-15
        )
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

    def test_monotone_on_linear_raises(self, spec, book):
        with pytest.raises(ValueError, match="piecewise-linear"):
            monotone_bounds(spec, {"Mileage": "increasing"})
        with pytest.raises(ValueError, match="piecewise-linear"):
            fit_glm(
                book.head(2000),
                spec,
                "ClaimNb",
                alpha=0.01,
                monotone={"Mileage": "increasing"},
                **FIT,
            )
        # step variables in the same spec are still fine
        lower, upper = monotone_bounds(spec, {"DrivAge": "decreasing"})
        assert np.all(
            upper[spec.slices()["DrivAge"]][: len(spec["DrivAge"].knots)] == 0
        )


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

    def test_flat_and_null_rows_edit_as_steps(self, fitted):
        _, rm0 = fitted
        rm = rm0.clone()
        rows = rm.variables["Mileage"].table
        before = [(r.relativity, r.slope) for r in rows]
        rm.update_relativity("Mileage", rows[-2].from_, None, 2.0)  # (hi, None)
        rm.update_relativity("Mileage", None, None, 3.0)  # null row
        after = [(r.relativity, r.slope) for r in rm.variables["Mileage"].table]
        diff = [i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a]
        assert diff == [len(rows) - 2, len(rows) - 1]
        x = pl.DataFrame(
            {
                "Mileage": [None, 1e9],
                "DrivAge": [40.0, 40.0],
                "Region": ["R1", "R1"],
                "Exposure": [1.0, 1.0],
            }
        )
        p0 = rm0.predict(x, exposure_col=None)
        p1 = rm.predict(x, exposure_col=None)
        assert p1[0] / p0[0] == pytest.approx(3.0 / before[-1][0], rel=1e-12)
        assert p1[1] / p0[1] == pytest.approx(2.0 / before[-2][0], rel=1e-12)

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
        with pytest.raises(ValueError, match="not continuous"):
            RateModel.from_rate_tables({"Mileage": broken}, 1.0)
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

    def test_validate_rejects_monotone_on_linear(self, project):
        from easy_glm.workflow import VariableDesign

        assert project.validate() == []
        project.design.variables["Mileage"] = VariableDesign(
            kind="linear", monotone="increasing"
        )
        assert any("piecewise-linear" in p for p in project.validate())
        project.design.variables["Mileage"] = VariableDesign(kind="linear")
        project.models["freq"].monotone = {"Mileage": "increasing"}
        assert any("piecewise-linear" in p for p in project.validate("freq"))
        project.models["freq"].monotone = {}
        project.design.variables["Mileage"] = VariableDesign(
            kind="linear", clamp=[5.0, 1.0]
        )
        assert any("clamp" in p for p in project.validate())

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
