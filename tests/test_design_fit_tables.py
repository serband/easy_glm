"""Tests for the 0.3 core: DesignSpec -> fit_glm -> exact rate tables."""

import numpy as np
import polars as pl
import pytest

from easy_glm import (
    CategoricalEncoder,
    DesignSpec,
    EasyGLM,
    StepEncoder,
    base_rate,
    fit_glm,
    rate_tables,
    to_rate_model,
)
from easy_glm.core.design import frequent_levels, quantile_knots
from easy_glm.core.fit import monotone_bounds
from easy_glm.engine import RateModel
from easy_glm.engine._scoring import score_numeric
from easy_glm.engine.models import FromToRow, VariableConfig
from easy_glm.ui.metrics import compute_actual_expected


@pytest.fixture
def messy_data() -> pl.DataFrame:
    """Poisson frequency data with nulls, a rare level and a holdout split."""
    rng = np.random.default_rng(7)
    n = 4000
    age = rng.integers(18, 80, n).astype(float)
    bm = rng.integers(50, 200, n).astype(float)
    region = rng.choice(
        ["R1", "R2", "R3", "R4", "Rare"], n, p=[0.45, 0.3, 0.15, 0.09, 0.01]
    ).astype(object)
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + 0.004 * (bm - 100)
        + np.where(region == "R1", 0.0, 0.25)
    )
    exposure = rng.uniform(0.2, 1.0, n)
    claims = rng.poisson(mu * exposure).astype(float)
    age[rng.random(n) < 0.05] = np.nan
    region[rng.random(n) < 0.03] = None
    return pl.DataFrame(
        {
            "ClaimNb": claims,
            "Exposure": exposure,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    ).with_columns(pl.col("DrivAge").fill_nan(None))


PREDICTORS = ["DrivAge", "BonusMalus", "Region"]


def _fit(data, **kwargs):
    train = data.filter(pl.col("traintest") == 1)
    spec = DesignSpec.from_data(train, PREDICTORS, min_level_share=0.02)
    defaults = {
        "family": "poisson",
        "weight_col": "Exposure",
        "divide_target_by_weight": True,
        "alpha": 0.002,
    }
    defaults.update(kwargs)
    return fit_glm(train, spec, "ClaimNb", **defaults)


# --------------------------------------------------------------------------
# DesignSpec
# --------------------------------------------------------------------------
class TestDesignSpec:
    def test_quantile_knots_are_observed_values_and_exclude_min(self):
        s = pl.Series([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None])
        knots = quantile_knots(s, n_bins=5)
        assert knots == sorted(set(knots))
        assert all(k in {2, 3, 4, 5, 6, 7, 8, 9, 10} for k in knots)
        assert 1 not in knots
        assert quantile_knots(pl.Series([None, None], dtype=pl.Float64)) == []

    def test_step_encoder_columns_bins_and_nulls(self):
        enc = StepEncoder("x", [7.0, 3.0], null_indicator=True)  # unsorted on purpose
        assert enc.knots == [3.0, 7.0]
        assert [f.name for f in enc.features()] == ["x>=3", "x>=7", "x is null"]
        assert enc.bins() == [(None, 3.0), (3.0, 7.0), (7.0, None)]
        mat = enc.transform(pl.Series([1.0, None, 5.0, 10.0, 3.0]))
        expected = np.array(
            [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=float
        )
        np.testing.assert_array_equal(mat, expected)

    def test_step_encoder_validation(self):
        with pytest.raises(ValueError):
            StepEncoder("x", [])
        with pytest.raises(ValueError):
            StepEncoder("x", [1.0, float("inf")])

    def test_categorical_encoder_reference_and_other(self):
        enc = CategoricalEncoder("r", ["A", "B"])
        assert enc.reference == "A"
        assert [f.name for f in enc.features()] == ["r=B", "r=Other"]
        mat = enc.transform(pl.Series(["A", "B", "C", None]))
        np.testing.assert_array_equal(mat, [[0, 0], [1, 0], [0, 1], [0, 1]])

    def test_categorical_encoder_validation(self):
        with pytest.raises(ValueError):
            CategoricalEncoder("r", [])
        with pytest.raises(ValueError):
            CategoricalEncoder("r", ["A", "A"])
        with pytest.raises(ValueError):
            CategoricalEncoder("r", ["A", "Other"])

    def test_frequent_levels_orders_by_frequency_and_drops_rare(self):
        s = pl.Series(["b", "a", "a", "a", "b", "c", None])
        assert frequent_levels(s, min_share=0.0) == ["a", "b", "c"]
        assert frequent_levels(s, min_share=0.3) == ["a", "b"]
        assert frequent_levels(s, min_share=0.0, max_levels=1) == ["a"]
        w = pl.Series([10.0, 1, 1, 1, 10, 1, 1])
        assert frequent_levels(s, min_share=0.0, weights=w)[0] == "b"

    def test_from_data_infers_kinds(self, messy_data):
        spec = DesignSpec.from_data(messy_data, PREDICTORS, min_level_share=0.02)
        assert isinstance(spec["DrivAge"], StepEncoder)
        assert isinstance(spec["Region"], CategoricalEncoder)
        assert spec["Region"].reference == "R1"
        assert "Rare" not in spec["Region"].levels
        assert spec.variables == PREDICTORS
        assert spec.n_features == len(spec.feature_names) == len(spec.features)
        assert spec.slices()["Region"].stop == spec.n_features

    def test_from_data_options(self, messy_data):
        spec = DesignSpec.from_data(
            messy_data,
            ["DrivAge", "BonusMalus"],
            knots={"DrivAge": [25, 40, 60]},
            categorical=["BonusMalus"],
            null_indicator=False,
        )
        assert spec["DrivAge"].knots == [25.0, 40.0, 60.0]
        assert spec["DrivAge"].null_indicator is False
        assert isinstance(spec["BonusMalus"], CategoricalEncoder)

    def test_from_data_errors(self, messy_data):
        with pytest.raises(KeyError):
            DesignSpec.from_data(messy_data, ["Nope"])
        constant = messy_data.with_columns(pl.lit(1.0).alias("Const"))
        with pytest.raises(ValueError, match="knots"):
            DesignSpec.from_data(constant, ["Const"])

    def test_json_roundtrip(self, messy_data, tmp_path):
        spec = DesignSpec.from_data(messy_data, PREDICTORS)
        spec.to_json(tmp_path / "spec.json")
        back = DesignSpec.from_json(tmp_path / "spec.json")
        assert back.to_dict() == spec.to_dict()
        np.testing.assert_array_equal(
            back.build(messy_data.head(50)), spec.build(messy_data.head(50))
        )

    def test_build_missing_column_raises(self, messy_data):
        spec = DesignSpec.from_data(messy_data, PREDICTORS)
        with pytest.raises(KeyError, match="Region"):
            spec.build(messy_data.drop("Region"))


# --------------------------------------------------------------------------
# fit_glm
# --------------------------------------------------------------------------
class TestFitGLM:
    def test_requires_alpha_or_cv(self, messy_data):
        spec = DesignSpec.from_data(messy_data, PREDICTORS)
        with pytest.raises(ValueError, match="alpha"):
            fit_glm(messy_data, spec, "ClaimNb")

    def test_fit_predict_and_coef_table(self, messy_data):
        fit = _fit(messy_data)
        assert fit.family == "poisson" and fit.link == "log"
        assert fit.alpha == pytest.approx(0.002)
        preds = fit.predict(messy_data.head(20))
        assert preds.shape == (20,) and np.all(np.isfinite(preds)) and np.all(preds > 0)
        np.testing.assert_allclose(
            preds, np.exp(fit.linear_predictor(messy_data.head(20)))
        )
        table = fit.coef_table()
        assert table.height == fit.spec.n_features + 1
        assert table["kind"][0] == "intercept"
        assert set(table["kind"].unique()) <= {
            "intercept",
            "step",
            "null",
            "level",
            "other",
        }
        kept = fit.coef_table(drop_zero=True)
        assert kept.height <= table.height
        assert "GLMFit" in repr(fit)

    def test_monotone_bounds_and_effect(self, messy_data):
        fit = _fit(messy_data, monotone={"DrivAge": "increasing"})
        lower, upper = monotone_bounds(fit.spec, {"DrivAge": "increasing"})
        sl = fit.spec.slices()["DrivAge"]
        n_knots = len(fit.spec["DrivAge"].knots)
        assert np.all(lower[sl][:n_knots] == 0) and np.isinf(lower[sl][n_knots])
        assert np.all(np.isinf(upper))
        age_coefs = fit.coef[sl][:n_knots]
        assert np.all(age_coefs >= 0)
        # unconstrained truth is decreasing in age, so the constraint bites
        assert fit.monotone == {"DrivAge": "increasing"}

    def test_monotone_rejects_bad_input(self, messy_data):
        spec = DesignSpec.from_data(messy_data, PREDICTORS)
        with pytest.raises(ValueError, match="categorical"):
            monotone_bounds(spec, {"Region": "increasing"})
        with pytest.raises(ValueError, match="increasing"):
            monotone_bounds(spec, {"DrivAge": "up"})
        with pytest.raises(KeyError):
            monotone_bounds(spec, {"Nope": "increasing"})

    def test_cv_selects_alpha(self, messy_data):
        fit = _fit(messy_data, alpha=None, cv=2, n_alphas=4)
        assert hasattr(fit.model, "alpha_")
        assert np.isfinite(fit.alpha) and fit.alpha > 0

    def test_cv_is_seeded_shuffled_and_stable_to_row_permutation(self, messy_data):
        kwargs = {"alpha": None, "cv": 3, "n_alphas": 8, "cv_seed": 91}
        ordered = _fit(messy_data, **kwargs)
        permuted = _fit(
            messy_data.sample(fraction=1.0, shuffle=True, seed=123), **kwargs
        )
        assert ordered.model.cv.shuffle is True
        assert ordered.model.cv.random_state == 91
        assert ordered.alpha == pytest.approx(permuted.alpha, rel=1e-12)

    def test_step_modal_base_never_uses_the_null_row(self):
        frame = pl.DataFrame(
            {
                "x": [None] * 80 + [1.0] * 12 + [2.0] * 8,
                "y": [0.0] * 80 + [1.0] * 20,
            }
        )
        spec = DesignSpec({"x": StepEncoder("x", [1.5], null_indicator=True)})
        fit = fit_glm(frame, spec, "y", family="poisson", alpha=0.01)
        assert fit.modal_bins["x"] != spec["x"].n_rows - 1

    def test_input_validation(self, messy_data):
        spec = DesignSpec.from_data(messy_data, PREDICTORS)
        with pytest.raises(ValueError, match="weight_col"):
            fit_glm(
                messy_data, spec, "ClaimNb", alpha=0.1, divide_target_by_weight=True
            )
        with pytest.raises(KeyError):
            fit_glm(messy_data, spec, "Nope", alpha=0.1)
        with pytest.raises(ValueError, match="Unknown family"):
            fit_glm(messy_data, spec, "ClaimNb", alpha=0.1, family="weibull")
        bad = messy_data.with_columns(pl.lit(-1.0).alias("Exposure"))
        with pytest.raises(ValueError, match="Weights"):
            fit_glm(bad, spec, "ClaimNb", alpha=0.1, weight_col="Exposure")


# --------------------------------------------------------------------------
# rate tables / RateModel exactness
# --------------------------------------------------------------------------
class TestRateTables:
    def test_rate_model_reproduces_glm_exactly(self, messy_data):
        fit = _fit(messy_data)
        rm = to_rate_model(fit, exposure_col="Exposure")
        holdout = messy_data.filter(pl.col("traintest") == 0)
        # inject an unseen level on top of the nulls already present
        holdout = holdout.with_columns(
            pl.when(pl.arange(0, holdout.height) % 97 == 0)
            .then(pl.lit("Unseen"))
            .otherwise(pl.col("Region"))
            .alias("Region")
        )
        assert holdout["DrivAge"].null_count() > 0
        assert holdout["Region"].null_count() > 0
        glm_pred = fit.predict(holdout)
        np.testing.assert_allclose(
            rm.predict(holdout, exposure_col=None), glm_pred, rtol=1e-10, atol=0
        )
        # exposure multiplication still works
        np.testing.assert_allclose(
            rm.predict(holdout), glm_pred * holdout["Exposure"].to_numpy(), rtol=1e-10
        )

    def test_rate_model_json_roundtrip_keeps_null_row(self, messy_data, tmp_path):
        fit = _fit(messy_data)
        rm = to_rate_model(fit)
        rm.to_json(tmp_path / "m.easyglm")
        back = RateModel.from_json(tmp_path / "m.easyglm")
        sample = messy_data.head(300)
        np.testing.assert_allclose(back.predict(sample), rm.predict(sample), rtol=1e-12)
        assert back.variables["DrivAge"].null_relativity is not None

    def test_tables_structure_and_base(self, messy_data):
        fit = _fit(messy_data)
        tables = rate_tables(fit)
        assert set(tables) == set(PREDICTORS)
        age = tables["DrivAge"]
        assert age.columns == [
            "from",
            "to",
            "label",
            "coef",
            "relativity",
            "exposure",
            "is_base",
        ]
        # D5: every row carries the training exposure (the weight) that fell in it
        train_exposure = messy_data.filter(pl.col("traintest") == 1)["Exposure"].sum()
        assert age["exposure"].sum() == pytest.approx(train_exposure)
        assert tables["Region"]["exposure"].sum() == pytest.approx(train_exposure)
        assert age.height == len(fit.spec["DrivAge"].knots) + 2  # bins + null row
        assert age["from"].dtype == pl.Float64
        assert age["label"][0].startswith("<") and age["label"][-1] == "Other / Unknown"
        assert age.filter(pl.col("is_base"))["relativity"][0] == pytest.approx(1.0)
        region = tables["Region"]
        assert region["from"].dtype == pl.Utf8
        assert region["label"].to_list()[:-1] == fit.spec["Region"].levels
        np.testing.assert_allclose(region["relativity"], np.exp(region["coef"]))

    def test_reference_base_and_base_rate(self, messy_data):
        fit = _fit(messy_data)
        ref = rate_tables(fit, base="reference")
        for tbl in ref.values():
            assert tbl["relativity"][0] == pytest.approx(1.0)
        assert base_rate(fit, base="reference") == pytest.approx(np.exp(fit.intercept))
        rm_ref = to_rate_model(fit, base="reference")
        rm_mod = to_rate_model(fit, base="modal")
        sample = messy_data.head(100)
        np.testing.assert_allclose(rm_ref.predict(sample), rm_mod.predict(sample))
        rm_override = to_rate_model(fit, base_rate_override=0.05)
        assert rm_override.base_rate == 0.05

    def test_a_logit_link_gives_odds_relativities(self, messy_data):
        """Piece E3 (Q7): a binomial fit *does* have rate tables — the numbers
        multiply the odds instead of the rate. Before E3 this raised."""
        train = messy_data.filter(pl.col("traintest") == 1).with_columns(
            (pl.col("ClaimNb") > 0).cast(pl.Float64).alias("AnyClaim")
        )
        spec = DesignSpec.from_data(train, PREDICTORS)
        fit = fit_glm(train, spec, "AnyClaim", family="binomial", alpha=0.01)
        assert fit.link == "logit"
        assert np.all(
            (fit.predict(train.head(5)) > 0) & (fit.predict(train.head(5)) < 1)
        )
        tables = rate_tables(fit)
        assert set(tables) == set(spec.variables)
        rm = to_rate_model(fit)
        assert rm.relativity_label == "odds relativity"
        assert np.allclose(rm.predict(train, exposure_col=None), fit.predict(train))

    def test_a_link_that_is_not_multiplicative_raises(self, messy_data):
        train = messy_data.filter(pl.col("traintest") == 1)
        spec = DesignSpec.from_data(train, PREDICTORS)
        fit = fit_glm(
            train, spec, "ClaimNb", family="gaussian", link="identity", alpha=0.01
        )
        with pytest.raises(NotImplementedError, match="logit link"):
            rate_tables(fit)


# --------------------------------------------------------------------------
# engine: numeric null row
# --------------------------------------------------------------------------
class TestEngineNullRow:
    @staticmethod
    def _config(with_null_row: bool) -> VariableConfig:
        rows = [
            FromToRow(None, 30.0, 1.5),
            FromToRow(30.0, 60.0, 1.0),
            FromToRow(60.0, None, 0.8),
        ]
        if with_null_row:
            rows.append(FromToRow(None, None, 1.2))
        cfg = VariableConfig(type="numeric", table=rows)
        RateModel._precompute_variables({"x": cfg})
        return cfg

    def test_score_numeric_uses_null_row(self):
        cfg = self._config(True)
        assert cfg.null_relativity == 1.2
        np.testing.assert_array_equal(
            score_numeric(np.array([10.0, np.nan, 45.0, 60.0]), cfg),
            [1.5, 1.2, 1.0, 0.8],
        )
        cfg.breakpoints = None  # exercise the slow path too
        np.testing.assert_array_equal(
            score_numeric(np.array([10.0, np.nan, 45.0, 60.0]), cfg),
            [1.5, 1.2, 1.0, 0.8],
        )

    def test_score_numeric_without_null_row_still_raises(self):
        cfg = self._config(False)
        assert cfg.null_relativity is None
        with pytest.raises(ValueError, match="NaN"):
            score_numeric(np.array([1.0, np.nan]), cfg)

    def test_update_relativity_keeps_null_row(self):
        cfg = self._config(True)
        rm = RateModel(base_rate=1.0, variables={"x": cfg})
        rm.update_relativity("x", None, None, 2.0)
        assert rm.variables["x"].null_relativity == 2.0
        data = pl.DataFrame({"x": [None, 10.0]})
        np.testing.assert_array_equal(rm.predict(data), [2.0, 1.5])


# --------------------------------------------------------------------------
# EasyGLM front door
# --------------------------------------------------------------------------
class TestEasyGLM:
    def test_fit_predict_save_load(self, messy_data, tmp_path):
        eglm = EasyGLM.fit(
            messy_data,
            target="ClaimNb",
            model_type="Poisson",
            predictors=PREDICTORS,
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.002,
            min_level_share=0.02,
            monotone={"BonusMalus": "increasing"},
        )
        assert set(eglm.relativities) == set(PREDICTORS)
        assert eglm.summary()["alpha"] == pytest.approx(0.002)
        assert eglm.summary()["model_type"] == "Poisson"
        assert eglm.rate_model.metadata.exposure_col == "Exposure"
        holdout = messy_data.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            eglm.rate_model.predict(holdout, exposure_col=None),
            eglm.predict(holdout).to_numpy(),
            rtol=1e-10,
        )
        eglm.save(tmp_path / "m")
        assert (tmp_path / "m" / "spec.json").exists()
        loaded = EasyGLM.load(tmp_path / "m")
        assert (
            loaded.predict(holdout.head(20)).to_list()
            == eglm.predict(holdout.head(20)).to_list()
        )
        assert loaded.glm.monotone == {"BonusMalus": "increasing"}
        assert loaded.relativities["Region"].equals(eglm.relativities["Region"])
        assert "EasyGLM" in repr(loaded)

    def test_use_cv_false_requires_alpha(self, messy_data):
        with pytest.raises(ValueError, match="alpha"):
            EasyGLM.fit(
                messy_data,
                target="ClaimNb",
                model_type="Poisson",
                predictors=PREDICTORS,
                use_cv=False,
            )

    def test_legacy_cv_params_are_mapped(self, messy_data):
        eglm = EasyGLM.fit(
            messy_data,
            target="ClaimNb",
            model_type="Poisson",
            predictors=["BonusMalus", "Region"],
            weight_col="Exposure",
            divide_target_by_weight=True,
            use_cv=True,
            cv_params={"n_alphas": 3, "l1_ratio": [1.0], "cv": 2},
        )
        assert hasattr(eglm.model, "alpha_")

    def test_load_rejects_pre_0_3_directories(self, tmp_path):
        (tmp_path / "blueprint.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="spec.json"):
            EasyGLM.load(tmp_path)


# --------------------------------------------------------------------------
# A/E metrics with Other / null rows
# --------------------------------------------------------------------------
def test_actual_expected_masks_other_and_null_rows(messy_data):
    fit = _fit(messy_data)
    rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
    data = messy_data.with_columns(
        (pl.col("ClaimNb") / pl.col("Exposure")).alias("ClaimNb")
    )
    region = compute_actual_expected(rm, data, "Region")["subsets"]["all"]
    other = region[-1]
    assert other["level"] == "Other / Unknown"
    n_other = data.filter(
        ~pl.col("Region").is_in(fit.spec["Region"].levels) | pl.col("Region").is_null()
    )
    assert other["exposure"] == pytest.approx(n_other["Exposure"].sum())
    total = sum(r["exposure"] for r in region)
    assert total == pytest.approx(data["Exposure"].sum())
    age = compute_actual_expected(rm, data, "DrivAge")["subsets"]["all"]
    assert age[-1]["exposure"] == pytest.approx(
        data.filter(pl.col("DrivAge").is_null())["Exposure"].sum()
    )
    assert sum(r["exposure"] for r in age) == pytest.approx(data["Exposure"].sum())


# --------------------------------------------------------------------------
# Excel export
# --------------------------------------------------------------------------
def _sheet_names(path) -> list[str]:
    import re
    import zipfile

    xml = zipfile.ZipFile(path).read("xl/workbook.xml").decode()
    return re.findall(r'<sheet [^>]*?name="([^"]+)"', xml)


def test_sheet_name_is_excel_safe_and_unique():
    from easy_glm.core.excel import sheet_name

    used: set[str] = set()
    first = sheet_name("a/very[long]:sheet*name?with\\bad" + "x" * 30, used)
    second = sheet_name("a/very[long]:sheet*name?with\\bad" + "x" * 30, used)
    for name in (first, second):
        assert len(name) <= 31
        assert not any(c in name for c in "[]:*?/\\")
    assert first != second and second.endswith("(2)")
    assert sheet_name("Region", used) == "Region"
    assert (
        sheet_name("region", used) == "region (2)"
    )  # Excel names are case-insensitive


def test_to_excel_writes_summary_coefficients_and_one_sheet_per_variable(
    messy_data, tmp_path
):
    from easy_glm.core.excel import rate_model_tables

    fit = _fit(messy_data)
    rm = to_rate_model(fit)
    eglm = EasyGLM(fit, rm)

    path = eglm.to_excel(tmp_path / "tables.xlsx")
    names = _sheet_names(path)
    assert names[:3] == ["Summary", "Index", "Coefficients"]
    assert names[3:] == PREDICTORS

    rm_path = rm.to_excel(tmp_path / "rm.xlsx")
    assert _sheet_names(rm_path) == ["Summary", "Index", *PREDICTORS]

    tables = rate_model_tables(rm)
    age = tables["DrivAge"]
    assert age.columns == [
        "from",
        "to",
        "label",
        "fitted",
        "relativity",
        "exposure",
    ]
    np.testing.assert_allclose(age["fitted"].to_numpy(), age["relativity"].to_numpy())
    assert age.height == len(fit.spec["DrivAge"].knots) + 2
    assert age["from"].dtype == pl.Float64
    np.testing.assert_allclose(
        age["relativity"].to_numpy(),
        rate_tables(fit)["DrivAge"]["relativity"].to_numpy(),
    )
    region = tables["Region"]
    assert region["from"].dtype == pl.Utf8
    assert region["label"][-1] == "Other / Unknown"
