"""Two-way interactions: encoder, engine table type, Excel, penalties, spec,
and the two-stage fit (A2) that freezes the mains before fitting the cells."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import (
    CategoricalEncoder,
    DesignSpec,
    InteractionEncoder,
    StepEncoder,
    TwoStageFit,
    fit_glm,
    fit_two_stage,
    rate_tables,
    to_rate_model,
)
from easy_glm.core.excel import interaction_matrices, rate_model_tables
from easy_glm.core.fit import monotone_bounds, penalty_weights
from easy_glm.core.tables import base_rate
from easy_glm.engine import RateModel
from easy_glm.engine._scoring import row_index as engine_row_index
from easy_glm.engine.models import CellRow
from easy_glm.ui.metrics import compute_actual_expected
from easy_glm.workflow import ae_by_pair
from easy_glm.workflow.explore import band_labels


def _book(seed: int = 5, n: int = 6000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    dens = rng.integers(1, 3000, n).astype(float)
    power = rng.integers(4, 12, n).astype(object)
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.5, 0.3, 0.15, 0.05]).astype(
        object
    )
    expo = rng.uniform(0.2, 1.0, n)
    young_r2 = (age < 25) & (region == "R2")
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + 0.0001 * dens
        + np.where(region == "R1", 0.0, 0.25)
        + np.where(young_r2, 0.7, 0.0)
    )
    claims = rng.poisson(mu * expo).astype(float)
    age[rng.random(n) < 0.05] = np.nan
    region[rng.random(n) < 0.04] = None
    power[rng.random(n) < 0.03] = None
    return pl.DataFrame(
        {
            "ClaimNb": claims,
            "Exposure": expo,
            "DrivAge": age,
            "Density": dens,
            "VehPower": pl.Series(
                [None if v is None else int(v) for v in power], dtype=pl.Int64
            ),
            "Region": region,
        }
    ).with_columns(pl.col("DrivAge").fill_nan(None))


@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    return _book()


@pytest.fixture(scope="module")
def spec(book) -> DesignSpec:
    return DesignSpec.from_data(
        book,
        ["DrivAge", "Density", "VehPower", "Region"],
        knots={"DrivAge": [25, 30, 40, 50, 60]},
        categorical=["VehPower"],
        min_level_share=0.02,
        weight_col="Exposure",  # cell exposure = fit weights
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.005,
    )


@pytest.fixture(scope="module")
def fitted(book, spec):
    fit = fit_glm(
        book,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.001,
    )
    return fit, to_rate_model(fit, exposure_col="Exposure")


def _scoring(book: pl.DataFrame) -> pl.DataFrame:
    out = book.tail(1000)
    return out.with_columns(
        pl.when(pl.arange(0, out.height) % 40 == 0)
        .then(pl.lit("UNSEEN"))
        .otherwise(pl.col("Region"))
        .alias("Region")
    )


# --------------------------------------------------------------------------
# shared row-index rule
# --------------------------------------------------------------------------
class TestRowIndex:
    def test_step_rows_and_index(self):
        enc = StepEncoder("x", [3.0, 7.0])
        assert enc.rows() == [(None, 3.0), (3.0, 7.0), (7.0, None), (None, None)]
        idx = enc.row_index(pl.Series([1.0, 3.0, 6.9, 7.0, 100.0, None]))
        assert idx.tolist() == [0, 1, 1, 2, 2, 3]

    def test_categorical_rows_and_index(self):
        enc = CategoricalEncoder("r", ["A", "B"])
        assert enc.rows() == [("A", "A"), ("B", "B"), (None, None)]
        idx = enc.row_index(pl.Series(["B", "A", "zzz", None]))
        assert idx.tolist() == [1, 0, 2, 2]
        # integer-typed column against string levels
        enc_i = CategoricalEncoder("p", ["4", "5"])
        assert enc_i.row_index(pl.Series([5, 4, 9, None])).tolist() == [1, 0, 2, 2]

    def test_encoder_and_engine_agree(self, book, spec, fitted):
        """The encoder's row rule and the RateModel's row rule are one rule."""
        fit, rm = fitted
        score = _scoring(book)
        for var in ("DrivAge", "Region", "VehPower"):
            enc_idx = spec[var].row_index(score[var])
            eng_idx = engine_row_index(score[var], rm.variables[var])
            np.testing.assert_array_equal(enc_idx, eng_idx)
            assert enc_idx.max() < spec[var].n_rows == len(rm.variables[var].table)

    def test_encoder_and_engine_agree_on_an_adversarial_frame(self, spec, fitted):
        """Nulls, NaN, values exactly on knots, unseen and integer-coded levels."""
        _, rm = fitted
        frame = pl.DataFrame(
            {
                "DrivAge": [None, float("nan"), 24.999, 25.0, 60.0, 1e9, -5.0],
                "Region": ["R1", None, "R2", "NEVER", "R4", "R3", ""],
                "VehPower": pl.Series([4, None, 99, 11, 4, 5, 7], dtype=pl.Int64),
            }
        )
        for var in ("DrivAge", "Region", "VehPower"):
            np.testing.assert_array_equal(
                spec[var].row_index(frame[var]),
                engine_row_index(frame[var], rm.variables[var]),
            )
        # and the rule is the documented one
        assert spec["DrivAge"].row_index(frame["DrivAge"]).tolist()[:4] == [6, 6, 0, 1]
        assert spec["Region"].row_index(frame["Region"]).tolist()[1] == len(
            spec["Region"].levels
        )

    def test_interaction_cell_index_uses_parents(self, book, spec):
        inter = spec["DrivAge×Region"]
        ia, ib = inter.cell_index(book.head(100))
        np.testing.assert_array_equal(
            ia, spec["DrivAge"].row_index(book["DrivAge"].head(100))
        )
        np.testing.assert_array_equal(
            ib, spec["Region"].row_index(book["Region"].head(100))
        )


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class TestInteractionEncoder:
    def test_kept_cells_exposure_and_columns(self, book):
        base = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [25, 40]},
            min_level_share=0.02,
        )
        enc = InteractionEncoder.from_data(
            base["DrivAge"],
            base["Region"],
            book,
            weights=book["Exposure"],
            min_cell_exposure=0.05,
        )
        exposure = np.asarray(enc.exposure)
        assert exposure.shape == (base["DrivAge"].n_rows, base["Region"].n_rows)
        assert exposure.sum() == pytest.approx(book["Exposure"].sum())
        share = exposure / exposure.sum()
        expected_cells = {
            (i, j)
            for i in range(share.shape[0])
            for j in range(share.shape[1])
            if share[i, j] >= 0.05
        }
        assert set(enc.cells) == expected_cells
        assert 0 < len(enc.cells) < exposure.size  # some cells were lumped
        assert enc.n_features == len(enc.cells) == len(enc.features())
        assert all(f.kind == "cell" and f.cell in enc.cells for f in enc.features())
        # design columns are one-hot over kept cells and zero elsewhere
        base.add_interaction(enc)
        design = base.build(book.head(500))
        cols = design[:, base.slices()["DrivAge×Region"]]
        assert set(np.unique(cols)) <= {0.0, 1.0}
        assert (cols.sum(axis=1) <= 1).all()

    def test_validation(self, book, spec):
        a, b = spec["DrivAge"], spec["Region"]
        na, nb = a.n_rows, b.n_rows
        exp = np.zeros((na, nb)).tolist()
        with pytest.raises(ValueError, match="different"):
            InteractionEncoder(a, a, [(0, 0)], np.zeros((na, na)).tolist())
        with pytest.raises(ValueError, match="outside"):
            InteractionEncoder(a, b, [(na, 0)], exp)
        with pytest.raises(ValueError, match="not unique"):
            InteractionEncoder(a, b, [(0, 0), (0, 0)], exp)
        with pytest.raises(ValueError, match="exposure must be"):
            InteractionEncoder(a, b, [(0, 0)], [[1.0]])
        with pytest.raises(ValueError, match="Interactions of interactions"):
            InteractionEncoder(spec["DrivAge×Region"], b, [(0, 0)], exp)
        with pytest.raises(ValueError, match="parent"):
            DesignSpec({"DrivAge": a}).add_interaction(
                InteractionEncoder(a, b, [(0, 0)], exp)
            )

    def test_spec_json_roundtrip(self, spec, book, tmp_path):
        spec.to_json(tmp_path / "spec.json")
        back = DesignSpec.from_json(tmp_path / "spec.json")
        assert back.to_dict() == spec.to_dict()
        assert back.variables == spec.variables
        np.testing.assert_array_equal(
            back.build(book.head(300)), spec.build(book.head(300))
        )
        assert back["DrivAge×Region"].a is back["DrivAge"]  # parents resolved by name
        bad = spec.to_dict()
        bad["encoders"][-1]["a"] = "Nope"
        with pytest.raises(ValueError, match="unknown parent"):
            DesignSpec.from_dict(bad)

    def test_symmetry(self, book):
        kw = {
            "knots": {"DrivAge": [25, 40, 60]},
            "min_level_share": 0.02,
            "min_cell_exposure": 0.01,
        }
        s_ab = DesignSpec.from_data(
            book, ["DrivAge", "Region"], interactions=[("DrivAge", "Region")], **kw
        )
        s_ba = DesignSpec.from_data(
            book, ["DrivAge", "Region"], interactions=[("Region", "DrivAge")], **kw
        )
        # the two designs differ only by column order; coordinate descent stops
        # at slightly different points unless the tolerance is tight
        fits = [
            fit_glm(
                book,
                s,
                "ClaimNb",
                family="poisson",
                weight_col="Exposure",
                divide_target_by_weight=True,
                alpha=0.001,
                gradient_tol=1e-9,
            )
            for s in (s_ab, s_ba)
        ]
        score = _scoring(book)
        np.testing.assert_allclose(
            fits[0].predict(score), fits[1].predict(score), rtol=1e-5
        )
        m_ab = interaction_matrices(to_rate_model(fits[0]), "DrivAge×Region")[2]
        m_ba = interaction_matrices(to_rate_model(fits[1]), "Region×DrivAge")[2]
        np.testing.assert_allclose(
            np.asarray(m_ab), np.asarray(m_ba).T, rtol=1e-4, atol=1e-6
        )

    def test_parent_must_be_a_predictor(self, book):
        with pytest.raises(ValueError, match="not one of the predictors"):
            DesignSpec.from_data(
                book,
                ["DrivAge"],
                knots={"DrivAge": [30]},
                interactions=[("DrivAge", "Region")],
            )

    def test_separator_in_a_variable_name_is_refused(self):
        a = StepEncoder("age×band", [30.0])
        b = CategoricalEncoder("r", ["A", "B"])
        with pytest.raises(ValueError, match="separator"):
            InteractionEncoder(a, b, [(0, 0)], np.zeros((a.n_rows, b.n_rows)).tolist())

    def test_zero_kept_cells(self, book):
        s = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [30, 45]},
            min_level_share=0.02,
            weight_col="Exposure",
            interactions=[("DrivAge", "Region")],
            min_cell_exposure=0.99,  # above every cell's share
        )
        enc = s["DrivAge×Region"]
        assert enc.cells == [] and enc.n_features == 0
        fit = fit_glm(
            book,
            s,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.001,
        )
        rm = to_rate_model(fit)
        score = _scoring(book)
        np.testing.assert_allclose(
            rm.predict(score, exposure_col=None), fit.predict(score), rtol=1e-10
        )
        assert (rate_tables(fit)["DrivAge×Region"]["relativity"] == 1.0).all()

    def test_monotone_on_interaction_raises(self, spec):
        with pytest.raises(ValueError, match="step"):
            monotone_bounds(spec, {"DrivAge×Region": "increasing"})


# --------------------------------------------------------------------------
# penalties
# --------------------------------------------------------------------------
class TestPenalty:
    def test_p1_aligns_with_features(self, book, spec):
        design = spec.build(book)
        p1 = penalty_weights(
            spec, design, book["Exposure"].to_numpy(), scale_predictors=True
        )
        assert p1 is not None and p1.shape == (spec.n_features,)
        kinds = [f.kind for f in spec.features]
        assert all(p1[i] == 1.0 for i, k in enumerate(kinds) if k != "cell")
        cells = np.array([p1[i] for i, k in enumerate(kinds) if k == "cell"])
        assert (cells > 0).all()
        # thin cells are penalised harder than fat ones
        sl = spec.slices()["DrivAge×Region"]
        share = design[:, sl].mean(axis=0)
        assert p1[sl][np.argmin(share)] > p1[sl][np.argmax(share)]
        # no interactions and no linear terms -> glum default
        plain = DesignSpec({"DrivAge": spec["DrivAge"]})
        assert (
            penalty_weights(plain, plain.build(book), None, scale_predictors=True)
            is None
        )

    def test_user_p1_unpenalises_a_main(self, book):
        s = DesignSpec.from_data(
            book, ["DrivAge", "Region"], knots={"DrivAge": [25, 40, 60]}
        )
        kw = {
            "family": "poisson",
            "weight_col": "Exposure",
            "divide_target_by_weight": True,
        }
        strong = fit_glm(book, s, "ClaimNb", alpha=0.05, **kw)
        sl = s.slices()["Region"]
        assert np.all(strong.coef[sl] == 0)  # heavy lasso removes Region
        p1 = np.ones(s.n_features)
        p1[sl] = 0.0
        free = fit_glm(book, s, "ClaimNb", alpha=0.05, P1=p1, **kw)
        assert np.any(free.coef[sl] != 0)  # unpenalised main keeps a coefficient


# --------------------------------------------------------------------------
# engine: interaction tables
# --------------------------------------------------------------------------
class TestRateModelInteractions:
    def test_tables_and_exactness(self, book, fitted):
        fit, rm = fitted
        score = _scoring(book)
        np.testing.assert_allclose(
            rm.predict(score, exposure_col=None), fit.predict(score), rtol=1e-10, atol=0
        )
        cfg = rm.variables["DrivAge×Region"]
        assert cfg.type == "interaction" and cfg.parents == ("DrivAge", "Region")
        assert len(cfg.table) == len(rm.variables["DrivAge"].table) * len(
            rm.variables["Region"].table
        )
        tab = rate_tables(fit)["DrivAge×Region"]
        assert set(tab.columns) >= {
            "from_a",
            "to_a",
            "from_b",
            "to_b",
            "exposure",
            "kept",
            "coef",
            "relativity",
        }
        assert (tab.filter(~pl.col("kept"))["relativity"] == 1.0).all()
        assert tab["exposure"].sum() == pytest.approx(book["Exposure"].sum())
        # base rate excludes interactions: prediction of the base risk before cells
        assert rm.base_rate == pytest.approx(base_rate(fit))

    def test_all_cells_one_equals_zeroed_slice(self, book, fitted):
        fit, rm = fitted
        rm2 = rm.clone()
        for row in rm2.variables["DrivAge×Region"].table:
            if row.relativity != 1.0:
                rm2.update_relativity(
                    "DrivAge×Region",
                    row.from_a,
                    row.to_a,
                    1.0,
                    from_b=row.from_b,
                    to_b=row.to_b,
                )
        score = _scoring(book)
        design = fit.spec.build(score)
        coef = fit.coef.copy()
        coef[fit.spec.slices()["DrivAge×Region"]] = 0.0
        manual = np.exp(fit.intercept + design @ coef)
        np.testing.assert_allclose(
            rm2.predict(score, exposure_col=None), manual, rtol=1e-10
        )

    def test_cell_update_snapshot_json_switch_diff(self, fitted, tmp_path):
        _, rm = fitted
        rm = rm.clone()
        name = "DrivAge×Region"
        cell = next(r for r in rm.variables[name].table if r.exposure > 0)
        before = rm.variables[name].cell_matrix.copy()
        rm.update_relativity(
            name, cell.from_a, cell.to_a, 2.5, from_b=cell.from_b, to_b=cell.to_b
        )
        assert rm.variables[name].cell_matrix.sum() != before.sum()
        v = rm.create_snapshot("cell edit")
        changes = rm.diff(1, v)
        assert len(changes) == 1 and changes[0].is_cell
        assert (
            changes[0].from_,
            changes[0].to_,
            changes[0].from_b,
            changes[0].to_b,
        ) == cell.key
        assert changes[0].new_relativity == 2.5
        rm.to_json(tmp_path / "m.easyglm")
        back = RateModel.from_json(tmp_path / "m.easyglm")
        np.testing.assert_array_equal(
            back.variables[name].cell_matrix, rm.variables[name].cell_matrix
        )
        back.switch_to(1)
        np.testing.assert_array_equal(back.variables[name].cell_matrix, before)
        back.switch_to(v)
        np.testing.assert_array_equal(
            back.variables[name].cell_matrix, rm.variables[name].cell_matrix
        )
        with pytest.raises(ValueError, match="from_b"):
            rm.update_relativity(name, cell.from_a, cell.to_a, 1.0)
        with pytest.raises(ValueError, match="No cell"):
            rm.update_relativity(name, "nope", "nope", 1.0, from_b=None, to_b=None)

    def test_excel_long_and_matrix_sheets(self, fitted, tmp_path):
        _, rm = fitted
        rm = rm.clone()
        name = "DrivAge×Region"
        cell = next(r for r in rm.variables[name].table if r.exposure > 0)
        rm.update_relativity(
            name, cell.from_a, cell.to_a, 1.7, from_b=cell.from_b, to_b=cell.to_b
        )
        path = rm.to_excel(tmp_path / "m.xlsx")
        sheets = pl.read_excel(path, sheet_id=0)
        assert name in sheets and f"{name} (matrix)" in sheets
        long = sheets[name]
        expected = rate_model_tables(rm)[name]
        np.testing.assert_allclose(
            long["relativity"].to_numpy(), expected["relativity"].to_numpy(), rtol=1e-9
        )
        np.testing.assert_allclose(
            long["exposure"].to_numpy(), expected["exposure"].to_numpy(), rtol=1e-9
        )
        assert (
            "fitted" in long.columns
            and long["fitted"].to_list() != long["relativity"].to_list()
        )
        rows_a, rows_b, rel, exp = interaction_matrices(rm, name)
        matrix = pl.read_excel(
            path, sheet_name=f"{name} (matrix)", has_header=False, infer_schema_length=0
        )
        # row 1 (0-based) holds the column labels, rows 2.. the row labels + values
        header = matrix.row(1)
        assert header[1 : 1 + len(rows_b)] == tuple(rows_b)
        first = matrix.row(2)
        assert first[0] == rows_a[0]
        np.testing.assert_allclose(
            [float(v) for v in first[1 : 1 + len(rows_b)]], rel[0], rtol=1e-9
        )
        # lumped cells: 1.0 with exposure below threshold
        lumped = expected.filter(pl.col("relativity") == 1.0)
        assert lumped.height > 0

    def test_labels_match_between_diagnostic_and_tables(self, book, fitted):
        """ae_by_pair with the model's knots and levels is joinable onto the
        interaction table by (label_a, label_b)."""
        fit, rm = fitted
        name = "DrivAge×Region"
        frame = book.tail(2000)
        actual = frame["ClaimNb"].to_numpy()
        w = frame["Exposure"].to_numpy()
        pair = ae_by_pair(
            frame,
            "DrivAge",
            "Region",
            actual,
            fit.predict(frame) * w,
            w,
            knots_a=fit.spec["DrivAge"].knots,
            levels_b=fit.spec["Region"].levels,
        )
        table = rate_tables(fit)[name]
        rows_a, rows_b, _, _ = interaction_matrices(rm, name)
        assert set(pair["label_a"]) <= set(rows_a)
        assert set(pair["label_b"]) <= set(rows_b)
        assert band_labels(fit.spec["DrivAge"].knots) == rows_a[:-1]
        joined = pair.join(table, on=["label_a", "label_b"], how="left")
        assert joined["relativity"].null_count() == 0
        assert "Other / Unknown" in pair["label_a"].to_list()

    def test_long_parent_names_keep_the_matrix_suffix(self, book, tmp_path):
        names = {
            "DrivAge": "DriverAgeAtInceptionInWholeYears",
            "Region": "GeographicalRegionCodeOfRisk",
            "Density": "PopulationDensityOfAreaOfRisk",
        }
        long = book.rename(names)
        s = DesignSpec.from_data(
            long,
            list(names.values()),
            knots={names["DrivAge"]: [30, 45], names["Density"]: [800, 1600]},
            min_level_share=0.02,
            weight_col="Exposure",
            interactions=[
                (names["DrivAge"], names["Region"]),
                (names["DrivAge"], names["Density"]),
            ],
            min_cell_exposure=0.01,
        )
        fit = fit_glm(
            long,
            s,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.001,
        )
        path = to_rate_model(fit).to_excel(tmp_path / "long.xlsx")
        sheet_names = list(pl.read_excel(path, sheet_id=0))
        matrix = [n for n in sheet_names if n.endswith(" (matrix)")]
        assert len(matrix) == 2 and len(set(sheet_names)) == len(sheet_names)
        assert all(len(n) <= 31 for n in sheet_names)

    def test_excel_long_sheets_rebuild_an_adjusted_model(self, book, tmp_path):
        s = DesignSpec.from_data(
            book,
            ["DrivAge", "Region", "VehPower"],
            knots={"DrivAge": [30, 45]},
            categorical=["VehPower"],
            min_level_share=0.02,
            weight_col="Exposure",
            interactions=[("DrivAge", "Region"), ("VehPower", "Region")],
            min_cell_exposure=0.01,
        )
        fit = fit_glm(
            book,
            s,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.001,
        )
        rm = to_rate_model(fit, exposure_col="Exposure")
        cell = next(r for r in rm.variables["VehPower×Region"].table if r.exposure > 0)
        rm.update_relativity(
            "VehPower×Region",
            cell.from_a,
            cell.to_a,
            1.77,
            from_b=cell.from_b,
            to_b=cell.to_b,
        )
        rm.update_relativity("Region", "R2", "R2", 0.5)
        path = rm.to_excel(tmp_path / "two.xlsx")
        sheets = pl.read_excel(path, sheet_id=0)
        tables = {v: sheets[v] for v in rm.variables}
        back = RateModel.from_rate_tables(tables, rm.base_rate, exposure_col="Exposure")
        score = _scoring(book)
        np.testing.assert_allclose(back.predict(score), rm.predict(score), rtol=1e-9)

    def test_from_rate_tables_roundtrip_and_errors(self, book, fitted):
        fit, rm = fitted
        score = _scoring(book)
        tables = rate_model_tables(rm)
        back = RateModel.from_rate_tables(tables, rm.base_rate)
        np.testing.assert_allclose(
            back.predict(score, exposure_col=None),
            rm.predict(score, exposure_col=None),
            rtol=1e-12,
        )
        also = RateModel.from_rate_tables(rate_tables(fit), base_rate(fit))
        np.testing.assert_allclose(
            also.predict(score, exposure_col=None), fit.predict(score), rtol=1e-10
        )
        # parent missing
        with pytest.raises(ValueError, match="A×B"):
            RateModel.from_rate_tables(
                {
                    "DrivAge×Region": tables["DrivAge×Region"],
                    "DrivAge": tables["DrivAge"],
                },
                1.0,
            )
        # cell that is not a row of the parents
        broken = tables["DrivAge×Region"].with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit("R99"))
            .otherwise(pl.col("from_b"))
            .alias("from_b"),
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit("R99"))
            .otherwise(pl.col("to_b"))
            .alias("to_b"),
        )
        with pytest.raises(ValueError, match="does not match"):
            RateModel.from_rate_tables({**tables, "DrivAge×Region": broken}, 1.0)
        dup = pl.concat([tables["DrivAge×Region"], tables["DrivAge×Region"].head(1)])
        with pytest.raises(ValueError, match="twice"):
            RateModel.from_rate_tables({**tables, "DrivAge×Region": dup}, 1.0)

    def test_interaction_name_split_is_robust(self):
        from easy_glm.engine.rate_model import _split_interaction_name

        assert _split_interaction_name("a×b×c", {"a×b": 1, "c": 1}) == ("a×b", "c")
        assert _split_interaction_name("a×b×c", {"a": 1, "b×c": 1}) == ("a", "b×c")
        with pytest.raises(ValueError, match="ambiguous"):
            _split_interaction_name("a×b×c", {"a×b": 1, "c": 1, "a": 1, "b×c": 1})
        with pytest.raises(ValueError, match="must be named"):
            _split_interaction_name("a×zzz", {"a": 1, "b": 1})

    def test_unknown_type_still_rejected(self, fitted):
        _, rm = fitted
        raw = rm._to_dict()
        raw["variables"]["DrivAge×Region"]["type"] = "spline"
        with pytest.raises(ValueError, match="spline"):
            RateModel._from_dict(raw)

    def test_actual_expected_per_cell(self, book, fitted):
        _, rm = fitted
        rm.metadata.target = "ClaimNb"
        data = book.tail(2000)
        res = compute_actual_expected(
            rm, data, "DrivAge×Region", formula="sum_over_weight"
        )
        rows = res["subsets"]["all"]
        assert len(rows) == len(rm.variables["DrivAge×Region"].table)
        assert sum(r["exposure"] for r in rows) == pytest.approx(data["Exposure"].sum())
        assert all("|" in r["level"] for r in rows)


def test_cellrow_key_and_label():
    r = CellRow(None, 25.0, "R2", "R2", 1.3, 12.0)
    assert r.key == (None, 25.0, "R2", "R2")
    from easy_glm.engine.models import level_label

    assert level_label(r) == "< 25.0 | R2"


# --------------------------------------------------------------------------
# A2: two stages — the mains are fitted first and frozen, the cells go on top
# --------------------------------------------------------------------------
FIT_KW = {
    "family": "poisson",
    "weight_col": "Exposure",
    "divide_target_by_weight": True,
}


@pytest.fixture(scope="module")
def two_stage(book, spec):
    return fit_two_stage(book, spec, "ClaimNb", alpha=0.001, **FIT_KW)


class TestTwoStageFit:
    def test_it_is_a_glmfit_with_both_stages_composed(self, book, spec, two_stage):
        """The composed object *is* a ``GLMFit``: same spec (mains then cells),
        stage 1's coefficients followed by stage 2's, stage 1's intercept — so
        every consumer (rate tables, base rate, coef_table, diagnostics) works
        with no special case."""
        fit = two_stage
        assert isinstance(fit, TwoStageFit)
        assert fit.spec.variables == spec.variables
        assert fit.spec.main_effects == spec.main_effects
        assert fit.intercept == fit.stage1.intercept
        np.testing.assert_array_equal(
            fit.coef, np.concatenate([fit.stage1.coef, fit.stage2.coef])
        )
        assert len(fit.coef) == spec.n_features
        # the coefficient table lines up with the composed spec
        table = fit.coef_table()
        assert table.height == spec.n_features + 1
        assert table["feature"].to_list()[1:] == spec.feature_names
        # linear_predictor = eta1 + eta2, and predict is its inverse link
        score = _scoring(book)
        eta1 = fit.stage1.linear_predictor(score)
        eta2 = fit.stage2.design_matrix(score) @ fit.stage2.coef
        np.testing.assert_allclose(fit.linear_predictor(score), eta1 + eta2, rtol=1e-12)
        np.testing.assert_allclose(fit.predict(score), np.exp(eta1 + eta2), rtol=1e-12)
        assert "TwoStageFit" in repr(fit) and "alpha_stage2" in repr(fit)

    def test_mains_and_base_rate_are_untouched_by_the_interaction(self, book, spec):
        """Q5: the main tables and the base rate are the ones a model without
        the interaction produces, to floating-point noise."""
        mains_only = DesignSpec({v: spec[v] for v in spec.main_effects})
        alone = fit_glm(book, mains_only, "ClaimNb", alpha=0.001, **FIT_KW)
        with_cells = fit_two_stage(book, spec, "ClaimNb", alpha=0.001, **FIT_KW)
        t0, t1 = rate_tables(alone), rate_tables(with_cells)
        for var in spec.main_effects:
            np.testing.assert_allclose(
                t1[var]["relativity"].to_numpy(),
                t0[var]["relativity"].to_numpy(),
                rtol=1e-13,  # glum's run-to-run noise, not a modelling difference
            )
        assert base_rate(with_cells) == pytest.approx(base_rate(alone), rel=1e-13)
        assert with_cells.modal_bins == alone.modal_bins

    def test_rate_model_is_exact_and_cells_are_pure_adjustments(self, book, two_stage):
        fit = two_stage
        rm = to_rate_model(fit, exposure_col="Exposure")
        score = _scoring(book)
        np.testing.assert_allclose(
            rm.predict(score, exposure_col=None), fit.predict(score), rtol=1e-10, atol=0
        )
        assert rm.base_rate == pytest.approx(base_rate(fit))
        # setting every cell to 1.00 leaves the stage-1 model exactly
        flat = rm.clone()
        for row in flat.variables["DrivAge×Region"].table:
            if row.relativity != 1.0:
                flat.update_relativity(
                    "DrivAge×Region",
                    row.from_a,
                    row.to_a,
                    1.0,
                    from_b=row.from_b,
                    to_b=row.to_b,
                )
        np.testing.assert_allclose(
            flat.predict(score, exposure_col=None),
            fit.stage1.predict(score),
            rtol=1e-10,
        )

    def test_offset_column_is_carried_by_stage_one_only(self, book, spec):
        """The user's offset belongs to the model once: stage 2 receives it
        inside its own offset (eta1 + log premium) and does not add it again."""
        with_offset = book.with_columns(
            pl.Series("logprem", np.log(np.linspace(150.0, 900.0, book.height)))
        )
        fit = fit_two_stage(
            with_offset, spec, "ClaimNb", alpha=0.001, offset_col="logprem", **FIT_KW
        )
        assert fit.offset_col == "logprem" and fit.stage2.offset_col is None
        score = _scoring(with_offset)
        eta = (
            fit.stage1.linear_predictor(score)
            + fit.stage2.design_matrix(score) @ fit.stage2.coef
            + score["logprem"].to_numpy()
        )
        np.testing.assert_allclose(fit.predict(score), np.exp(eta), rtol=1e-12)
        rm = to_rate_model(fit, exposure_col="Exposure")
        np.testing.assert_allclose(
            rm.predict(score, exposure_col=None), fit.predict(score), rtol=1e-10
        )

    def test_stage_two_alpha_defaults_to_stage_one_and_can_be_set(self, book, spec):
        same = fit_two_stage(book, spec, "ClaimNb", alpha=0.001, **FIT_KW)
        assert same.alpha == pytest.approx(0.001)
        assert same.alpha_stage2 == pytest.approx(0.001)
        harder = fit_two_stage(
            book, spec, "ClaimNb", alpha=0.001, stage2_alpha=0.5, **FIT_KW
        )
        assert harder.alpha_stage2 == pytest.approx(0.5)
        # the mains are the same fit either way; only the cells shrink
        np.testing.assert_allclose(harder.stage1.coef, same.stage1.coef, rtol=1e-12)
        assert np.abs(harder.stage2.coef).max() < np.abs(same.stage2.coef).max()

    def test_no_kept_cell_means_no_second_stage(self, book):
        s = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [30, 45]},
            min_level_share=0.02,
            weight_col="Exposure",
            interactions=[("DrivAge", "Region")],
            min_cell_exposure=0.99,  # above every cell's share
        )
        fit = fit_two_stage(book, s, "ClaimNb", alpha=0.001, **FIT_KW)
        assert not isinstance(fit, TwoStageFit)  # nothing for stage 2 to fit
        assert (rate_tables(fit)["DrivAge×Region"]["relativity"] == 1.0).all()
        score = _scoring(book)
        np.testing.assert_allclose(
            to_rate_model(fit).predict(score, exposure_col=None),
            fit.predict(score),
            rtol=1e-10,
        )

    def test_what_the_two_stages_refuse(self, book, spec, two_stage):
        mains_only = DesignSpec({v: spec[v] for v in spec.main_effects})
        with pytest.raises(ValueError, match="at least one interaction"):
            fit_two_stage(book, mains_only, "ClaimNb", alpha=0.001, **FIT_KW)
        with pytest.raises(ValueError, match="only hold interactions"):
            TwoStageFit(two_stage.stage1, two_stage.stage1)
        with pytest.raises(ValueError, match="cannot be a stage of another"):
            TwoStageFit(two_stage, two_stage.stage2)
        joint = fit_glm(book, spec, "ClaimNb", alpha=0.001, **FIT_KW)
        with pytest.raises(ValueError, match="main effects only"):
            TwoStageFit(joint, two_stage.stage2)
        # a second stage fitted *with* an intercept would move the base rate
        with_intercept = fit_glm(
            book,
            spec.interactions_spec(),
            "ClaimNb",
            alpha=0.001,
            offset=two_stage.stage1.linear_predictor(book),
            **FIT_KW,
        )
        with pytest.raises(ValueError, match="fit_intercept=False"):
            TwoStageFit(two_stage.stage1, with_intercept)
        # ... and a cell whose parent is not the mains' own encoder
        other = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [25, 30, 40, 50, 60]},
            min_level_share=0.02,
            weight_col="Exposure",
            interactions=[("DrivAge", "Region")],
        )
        with pytest.raises(ValueError, match="not \\(the same encoder"):
            TwoStageFit(
                fit_glm(
                    book, other.main_effects_spec(), "ClaimNb", alpha=0.001, **FIT_KW
                ),
                two_stage.stage2,
            )

    def test_fit_glm_offset_and_intercept_arguments(self, book, spec):
        mains = spec.main_effects_spec()
        eta = np.zeros(book.height)
        with pytest.raises(ValueError, match="not both"):
            fit_glm(
                book,
                mains,
                "ClaimNb",
                alpha=0.001,
                offset=eta,
                offset_col="Exposure",
                **FIT_KW,
            )
        with pytest.raises(ValueError, match="one value per training row"):
            fit_glm(book, mains, "ClaimNb", alpha=0.001, offset=eta[:10], **FIT_KW)
        bad = eta.copy()
        bad[0] = np.nan
        with pytest.raises(ValueError, match="offset contains NaN"):
            fit_glm(book, mains, "ClaimNb", alpha=0.001, offset=bad, **FIT_KW)
        with pytest.raises(ValueError, match="scale_predictors=False"):
            fit_glm(
                book,
                spec.interactions_spec(),
                "ClaimNb",
                alpha=0.001,
                offset=eta,
                fit_intercept=False,
                **FIT_KW,
            )


class TestStageTwoPenalty:
    def test_the_cell_rule_is_the_same_penalty_in_stage_two(self, book, spec):
        """R3's cell rule is ``P1 = penalty_weight * 0.5 / sd`` under glum's
        standardisation. glum refuses to standardise without an intercept, and
        stage 2 has no intercept — so the rule is written unstandardised as
        ``penalty_weight * 0.5``. The two are the same penalty because glum
        multiplies a standardised column's ``P1`` by that column's ``sd``."""
        cells = spec.interactions_spec()
        design = cells.build(book)
        w = book["Exposure"].to_numpy()
        std = penalty_weights(cells, design, w, scale_predictors=True)
        raw = penalty_weights(cells, design, w, scale_predictors=False)
        ww = w / w.sum()
        sd = np.sqrt(ww @ (design**2) - (ww @ design) ** 2)
        np.testing.assert_allclose(std * sd, raw, rtol=1e-12)
        np.testing.assert_allclose(raw, 0.5, rtol=1e-12)  # penalty_weight = 1
        # penalty_weight still multiplies it, and thin cells are still shrunk
        # harder per unit of *standardised* coefficient
        assert std[np.argmin(design.mean(axis=0))] > std[np.argmax(design.mean(axis=0))]

    def test_penalty_weight_scales_the_cells_of_that_interaction(self, book):
        def _fit(weight: float):
            s = DesignSpec.from_data(
                book,
                ["DrivAge", "Region"],
                knots={"DrivAge": [25, 40, 60]},
                min_level_share=0.02,
                weight_col="Exposure",
                interactions=[("DrivAge", "Region")],
                min_cell_exposure=0.005,
                interaction_penalty_weight=weight,
            )
            return fit_two_stage(book, s, "ClaimNb", alpha=0.002, **FIT_KW)

        light, heavy = _fit(1.0), _fit(20.0)
        assert np.abs(heavy.stage2.coef).max() < np.abs(light.stage2.coef).max()
        np.testing.assert_allclose(heavy.stage1.coef, light.stage1.coef, rtol=1e-12)
