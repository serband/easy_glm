"""Pieces E and F — modelling extras and the command-line interface.

E1  rate-change setup: the ``current_premium`` role, the derived
    ``log(premium)`` offset, the "multiplier on current premium" labels and the
    algebraic identity between an offset model and a premium-weighted one
    (Poisson only, ``scale_predictors=False``, alpha rescaled — plan §R6/S1).
E2  per-variable penalty weights (``VariableDesign.penalty_weight``).
E3  Tweedie power and binomial (logit) models in the workbench.
E4  the target-loss-ratio base-rate solver.
F   the ``easy-glm`` console script, driven through ``subprocess``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, fit_glm, to_rate_model
from easy_glm.core.design import CategoricalEncoder, StepEncoder
from easy_glm.core.fit import penalty_weights
from easy_glm.engine import RateModel
from easy_glm.workflow import (
    Project,
    VariableDesign,
    premium_offset_column,
    prepare,
    run_model,
    solve_base_rate,
    to_script,
)

SRC = str(Path(__file__).resolve().parents[1] / "src")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    """A small book with a current premium that is *not* the right price."""
    rng = np.random.default_rng(20260903)
    n = 6000
    age = rng.integers(18, 80, n).astype(float)
    region = rng.choice(["North", "South", "Urban"], n, p=[0.4, 0.35, 0.25])
    exposure = rng.uniform(0.25, 1.0, n)
    # today's premium: knows about age, ignores region
    premium = np.exp(4.0 + 0.015 * (60 - age)) * exposure
    truth = np.exp(
        np.log(premium) - 4.4 + np.where(region == "Urban", 0.35, 0.0)
    )  # the "right" expected cost
    claims = rng.poisson(truth).astype(float)
    lapsed = rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.02 * (age - 45))))
    return pl.DataFrame(
        {
            "DrivAge": age,
            "Region": region,
            "Exposure": exposure,
            "Premium": premium,
            "log_Premium_expected": np.log(premium),
            "ClaimNb": claims,
            "Lapsed": lapsed.astype(float),
            "traintest": (rng.random(n) < 0.7).astype(np.int64),
        }
    )


@pytest.fixture(scope="module")
def data_path(book, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("efdata") / "book.parquet"
    book.write_parquet(path)
    return path


def rate_change_project(data_path: Path) -> Project:
    """The standard rate-review setup: Premium is the current premium, so
    ``log_Premium`` is derived and becomes the offset of every new model."""
    p = Project(name="ratechange")
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Premium": "current_premium",
        "DrivAge": "predictor",
        "Region": "predictor",
        "Exposure": "ignore",
        "log_Premium_expected": "ignore",
        "Lapsed": "ignore",
    }
    p.data.filters = ["pl.col('Premium') > 0"]
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.data.split.train_value = 1
    p.design.variables["DrivAge"] = VariableDesign(knots=[25.0, 40.0, 60.0])
    p.new_model("change", family="poisson", predictors=["DrivAge", "Region"])
    p.models["change"].penalty.alpha = 0.001
    p.models["change"].penalty.cv = None
    return p


# --------------------------------------------------------------------------
# E1 — the rate-change setup
# --------------------------------------------------------------------------
class TestRateChangeSetup:
    def test_prep_derives_the_log_premium_column(self, book, data_path):
        p = rate_change_project(data_path)
        df = prepare(p)
        col = premium_offset_column("Premium")
        assert col == "log_Premium" and col in df.columns
        assert np.allclose(df[col].to_numpy(), np.log(df["Premium"].to_numpy()))

    def test_new_models_offset_on_it(self, data_path):
        p = rate_change_project(data_path)
        assert p.offset_column == "log_Premium"
        assert p.models["change"].offset == "log_Premium"

    def test_a_premium_that_has_no_logarithm_is_refused_by_name(self, book, tmp_path):
        bad = book.with_columns(
            pl.when(pl.arange(0, pl.len()) < 3)
            .then(0.0)
            .otherwise(pl.col("Premium"))
            .alias("Premium")
        )
        path = tmp_path / "bad.parquet"
        bad.write_parquet(path)
        p = rate_change_project(path)
        p.data.filters = []  # the user has not filtered them out
        with pytest.raises(ValueError, match=r"Premium.*3 row"):
            prepare(p)

    def test_the_filter_runs_before_the_logarithm(self, book, tmp_path):
        """A row filter is the fix the message asks for, so it must apply first."""
        bad = book.with_columns(
            pl.when(pl.arange(0, pl.len()) < 3)
            .then(0.0)
            .otherwise(pl.col("Premium"))
            .alias("Premium")
        )
        path = tmp_path / "bad2.parquet"
        bad.write_parquet(path)
        p = rate_change_project(path)  # keeps the pl.col('Premium') > 0 filter
        df = prepare(p)
        assert df.height == book.height - 3
        assert np.isfinite(df["log_Premium"].to_numpy()).all()

    def test_tables_are_labelled_multipliers_on_the_current_premium(self, data_path):
        p = rate_change_project(data_path)
        run = run_model(p, prepare(p), "change")
        rm = run.rate_model
        assert rm.metadata.offset_col == "log_Premium"
        assert rm.metadata.offset_is_premium is True
        assert rm.relativity_label == "multiplier on current premium"
        assert "overall" in rm.relativity_note and "differential" in rm.relativity_note

    def test_excel_summary_says_how_to_read_the_tables(self, data_path, tmp_path):
        p = rate_change_project(data_path)
        run = run_model(p, prepare(p), "change")
        path = run.rate_model.to_excel(tmp_path / "tables.xlsx")
        summary = pl.read_excel(path, sheet_name="Summary", has_header=False)
        text = " ".join(str(v) for v in summary.to_series(1).to_list())
        assert "multiplier on current premium" in text

    def test_the_label_survives_a_json_round_trip(self, data_path, tmp_path):
        p = rate_change_project(data_path)
        run = run_model(p, prepare(p), "change")
        path = tmp_path / "m.easyglm"
        run.rate_model.to_json(path)
        back = RateModel.from_json(path)
        assert back.metadata.offset_is_premium is True
        assert back.relativity_label == "multiplier on current premium"

    def test_a_file_written_before_this_field_reads_as_no_premium(self, tmp_path):
        rm = to_rate_model(_tiny_fit())
        path = tmp_path / "old.easyglm"
        raw = json.loads(json.dumps(rm._to_dict()))
        raw["metadata"].pop("offset_is_premium")
        path.write_text(json.dumps(raw))
        assert RateModel.from_json(path).metadata.offset_is_premium is False

    def test_the_exported_script_shows_the_derivation(self, data_path, tmp_path):
        p = rate_change_project(data_path)
        run = run_model(p, prepare(p), "change")
        src = to_script(p, "change", run=run, output_prefix="change")
        assert "pl.col('Premium').cast(pl.Float64).log().alias('log_Premium')" in src
        assert "offset_col='log_Premium'" in src
        assert "offset_is_premium=True" in src
        script = tmp_path / "rebuild.py"
        script.write_text(src)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=_env(),
        )
        assert proc.returncode == 0, proc.stderr
        rebuilt = RateModel.from_json(tmp_path / "change.easyglm")
        df = prepare(p)
        assert np.allclose(
            rebuilt.predict(df, exposure_col=None),
            run.rate_model.predict(df, exposure_col=None),
            rtol=1e-12,
        )

    def test_renaming_the_premium_follows_the_offset(self, data_path):
        p = rate_change_project(data_path)
        p.data.renames["Premium"] = "Prem"
        touched = p.rename_column("Premium", "Prem")
        assert touched == ["change"]
        assert p.models["change"].offset == "log_Prem"

    def test_dropping_the_role_clears_the_offset(self, data_path):
        p = rate_change_project(data_path)
        notices = p.apply_role_change("Premium", "ignore")
        assert p.models["change"].offset is None
        assert any("log_Premium" in n for n in notices)


class TestOffsetIdentity:
    """Plan §R6/S1: ``offset = log(P)`` and ``target/P`` weighted by ``P`` are
    the same model — for the **Poisson** deviance, with ``scale_predictors=
    False`` and the offset model's alpha multiplied by ``sum(P) / n``."""

    ALPHA = 1e-4
    #: the identity is exact only at the optimum, so both fits are solved to a
    #: far tighter gradient than a working fit needs
    KW = {"scale_predictors": False, "gradient_tol": 1e-12, "max_iter": 100_000}

    def _spec(self, book: pl.DataFrame) -> DesignSpec:
        return DesignSpec(
            {
                "DrivAge": StepEncoder("DrivAge", [25.0, 40.0, 60.0]),
                "Region": CategoricalEncoder("Region", ["North", "South", "Urban"]),
            }
        )

    def test_poisson_offset_equals_the_premium_weighted_fit(self, book):
        spec = self._spec(book)
        scale = book["Premium"].sum() / book.height
        offset_fit = fit_glm(
            book,
            spec,
            "ClaimNb",
            family="poisson",
            offset_col="log_Premium_expected",
            alpha=self.ALPHA * scale,
            **self.KW,
        )
        weighted_fit = fit_glm(
            book,
            spec,
            "ClaimNb",
            family="poisson",
            weight_col="Premium",
            divide_target_by_weight=True,
            alpha=self.ALPHA,
            **self.KW,
        )
        assert np.max(np.abs(offset_fit.coef - weighted_fit.coef)) < 1e-8
        assert abs(offset_fit.intercept - weighted_fit.intercept) < 1e-8
        premium = book["Premium"].to_numpy()
        assert np.allclose(
            offset_fit.predict(book),
            weighted_fit.predict(book) * premium,
            rtol=1e-8,
        )

    def test_gamma_is_not_expected_to_match(self, book):
        """Only the Poisson deviance is invariant under the swap, so the same
        two fits on a Gamma target are genuinely different models. Recorded so
        nobody reads the Poisson test as a general law."""
        spec = self._spec(book)
        cost = book.with_columns(
            (pl.col("ClaimNb").clip(lower_bound=0.1) * 400.0).alias("Cost")
        )
        scale = cost["Premium"].sum() / cost.height
        # ordinary tolerances here: the two fits are far apart, and a Gamma fit
        # squeezed to a gradient of 1e-12 takes minutes without changing that
        kw = {"scale_predictors": False}
        a = fit_glm(
            cost,
            spec,
            "Cost",
            family="gamma",
            offset_col="log_Premium_expected",
            alpha=self.ALPHA * scale,
            **kw,
        )
        b = fit_glm(
            cost,
            spec,
            "Cost",
            family="gamma",
            weight_col="Premium",
            divide_target_by_weight=True,
            alpha=self.ALPHA,
            **kw,
        )
        assert np.max(np.abs(a.coef - b.coef)) > 1e-3


# --------------------------------------------------------------------------
# E2 — per-variable penalty weights
# --------------------------------------------------------------------------
class TestPenaltyWeights:
    def test_p1_aligns_with_the_features(self, book):
        spec = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [25.0, 40.0, 60.0]},
            penalty_weight={"Region": 0.0, "DrivAge": 3.0},
        )
        design = spec.build(book)
        p1 = penalty_weights(spec, design, None, scale_predictors=True)
        assert p1 is not None and p1.shape == (spec.n_features,)
        by_var = {
            v: p1[sl] for v, sl in spec.slices().items()
        }  # one block per variable
        assert np.all(by_var["Region"] == 0.0)
        assert np.all(by_var["DrivAge"] == 3.0)
        for feature, weight in zip(spec.features, p1, strict=True):
            assert weight == (0.0 if feature.variable == "Region" else 3.0)

    def test_no_weights_anywhere_leaves_glum_to_its_default(self, book):
        spec = DesignSpec.from_data(
            book, ["DrivAge", "Region"], knots={"DrivAge": [25.0, 40.0, 60.0]}
        )
        assert (
            penalty_weights(spec, spec.build(book), None, scale_predictors=True) is None
        )

    def test_an_unpenalised_categorical_keeps_every_level(self, book):
        """At an alpha that lassoes the region levels away, penalty_weight = 0
        keeps all of them — the point of the setting."""
        knots = {"DrivAge": [25.0, 40.0, 60.0]}
        alpha = 0.05
        plain = DesignSpec.from_data(book, ["DrivAge", "Region"], knots=knots)
        kept = DesignSpec.from_data(
            book, ["DrivAge", "Region"], knots=knots, penalty_weight={"Region": 0.0}
        )
        squeezed = fit_glm(book, plain, "ClaimNb", family="poisson", alpha=alpha)
        free = fit_glm(book, kept, "ClaimNb", family="poisson", alpha=alpha)

        def level_coefs(fit, spec):
            # the "Other" column is empty here (every level is kept), so it is
            # zero whatever the penalty; the levels are what the setting is about
            return np.array(
                [
                    c
                    for f, c in zip(spec.features, fit.coef, strict=True)
                    if f.variable == "Region" and f.kind == "level"
                ]
            )

        assert (level_coefs(squeezed, plain) == 0).any()  # the lasso killed one
        assert (level_coefs(free, kept) != 0).all()

    def test_a_heavier_weight_shrinks_a_factor_harder(self, book):
        knots = {"DrivAge": [25.0, 40.0, 60.0]}
        light = DesignSpec.from_data(book, ["DrivAge", "Region"], knots=knots)
        heavy = DesignSpec.from_data(
            book, ["DrivAge", "Region"], knots=knots, penalty_weight={"Region": 25.0}
        )
        a = fit_glm(book, light, "ClaimNb", family="poisson", alpha=0.002)
        b = fit_glm(book, heavy, "ClaimNb", family="poisson", alpha=0.002)
        assert (
            np.abs(b.coef[heavy.slices()["Region"]]).sum()
            < np.abs(a.coef[light.slices()["Region"]]).sum()
        )

    def test_the_workbench_design_carries_it(self, data_path):
        p = rate_change_project(data_path)
        p.design.variables["Region"] = VariableDesign(penalty_weight=0.0)
        run = run_model(p, prepare(p), "change")
        assert run.spec["Region"].penalty_weight == 0.0
        src = to_script(p, "change", run=run)
        assert "penalty_weight=0.0" in src

    def test_a_negative_weight_is_a_validation_problem(self, data_path):
        p = rate_change_project(data_path)
        p.design.variables["Region"] = VariableDesign(penalty_weight=-1.0)
        assert any("penalty_weight" in m for m in p.validate())

    def test_it_survives_a_spec_round_trip(self, book):
        spec = DesignSpec.from_data(
            book,
            ["DrivAge", "Region"],
            knots={"DrivAge": [25.0, 40.0, 60.0]},
            penalty_weight={"Region": 0.0},
        )
        back = DesignSpec.from_dict(spec.to_dict())
        assert back["Region"].penalty_weight == 0.0


def _tiny_fit():
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame(
        {
            "x": rng.integers(0, 10, n).astype(float),
            "y": rng.poisson(1.0, n).astype(float),
        }
    )
    spec = DesignSpec({"x": StepEncoder("x", [3.0, 6.0])})
    return fit_glm(df, spec, "y", family="poisson", alpha=0.01)


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


# --------------------------------------------------------------------------
# E3 — Tweedie power and binomial (logit) models
# --------------------------------------------------------------------------
def binomial_project(data_path: Path) -> Project:
    p = Project(name="lapse")
    p.data.source.path = str(data_path)
    p.data.roles = {
        "Lapsed": "target",
        "DrivAge": "predictor",
        "Region": "predictor",
        "Exposure": "ignore",
        "Premium": "ignore",
        "log_Premium_expected": "ignore",
        "ClaimNb": "ignore",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.data.split.train_value = 1
    p.design.variables["DrivAge"] = VariableDesign(knots=[25.0, 40.0, 60.0])
    p.new_model("lapse", family="binomial", predictors=["DrivAge", "Region"])
    p.models["lapse"].penalty.alpha = 0.001
    p.models["lapse"].penalty.cv = None
    return p


class TestTweediePower:
    def test_the_power_reaches_the_distribution(self, book):
        spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [40.0])})
        fit = fit_glm(
            book, spec, "ClaimNb", family="tweedie", tweedie_power=1.7, alpha=0.01
        )
        assert fit.family == "tweedie"
        assert fit.model.family_instance.power == pytest.approx(1.7)

    def test_the_default_is_one_and_a_half(self, book):
        spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [40.0])})
        fit = fit_glm(book, spec, "ClaimNb", family="tweedie", alpha=0.01)
        assert fit.model.family_instance.power == pytest.approx(1.5)

    def test_a_power_outside_one_to_two_is_refused(self, book):
        spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [40.0])})
        with pytest.raises(ValueError, match="strictly between 1 and 2"):
            fit_glm(
                book, spec, "ClaimNb", family="tweedie", tweedie_power=2.5, alpha=0.01
            )

    def test_it_is_only_for_the_tweedie_family(self, book):
        spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [40.0])})
        with pytest.raises(ValueError, match="only meaningful for the tweedie"):
            fit_glm(
                book, spec, "ClaimNb", family="poisson", tweedie_power=1.7, alpha=0.01
            )

    def test_the_workbench_carries_it_into_the_fit_and_the_script(self, data_path):
        p = rate_change_project(data_path)
        cfg = p.models["change"]
        cfg.family = "tweedie"
        cfg.tweedie_power = 1.8
        run = run_model(p, prepare(p), "change")
        assert run.fit.model.family_instance.power == pytest.approx(1.8)
        assert "tweedie_power=1.8" in to_script(p, "change", run=run)

    def test_the_project_refuses_a_power_outside_the_range(self, data_path):
        p = rate_change_project(data_path)
        p.models["change"].family = "tweedie"
        p.models["change"].tweedie_power = 2.0
        assert any("tweedie_power" in m for m in p.validate("change"))


class TestBinomial:
    def test_the_scorer_returns_probabilities_and_matches_the_glm(self, data_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        df = prepare(p)
        pred = run.rate_model.predict(df, exposure_col=None)
        assert ((pred > 0) & (pred < 1)).all()
        assert np.allclose(pred, run.fit.predict(df), rtol=1e-10)

    def test_the_tables_are_odds_relativities(self, data_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        assert run.rate_model.metadata.link == "logit"
        assert run.rate_model.relativity_label == "odds relativity"
        assert "odds" in run.rate_model.relativity_note

    def test_the_base_rate_is_the_base_risk_odds(self, data_path):
        """base rate x relativities = odds, so the base rate is odds, and the
        probability it implies is what the scorer returns for the base risk."""
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        odds = run.rate_model.base_rate
        assert 0 < odds / (1 + odds) < 1

    def test_excel_says_odds_relativity(self, data_path, tmp_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        path = run.rate_model.to_excel(tmp_path / "lapse.xlsx")
        summary = pl.read_excel(path, sheet_name="Summary", has_header=False)
        text = " ".join(str(v) for v in summary.to_series(1).to_list())
        assert "odds relativity" in text

    def test_exposure_multiplication_is_refused(self, data_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        df = prepare(p)
        with pytest.raises(ValueError, match="probability"):
            run.rate_model.predict(df, exposure_col="Exposure")

    def test_a_model_never_gets_an_exposure_column_to_begin_with(self, data_path):
        p = binomial_project(data_path)
        p.data.roles["Exposure"] = "exposure"
        run = run_model(p, prepare(p), "lapse")
        assert run.rate_model.metadata.exposure_col is None

    def test_to_rate_model_refuses_an_exposure_column(self, data_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        with pytest.raises(ValueError, match="cannot be multiplied by an exposure"):
            to_rate_model(run.fit, exposure_col="Exposure")

    def test_it_round_trips_through_json(self, data_path, tmp_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        df = prepare(p)
        path = tmp_path / "lapse.easyglm"
        run.rate_model.to_json(path)
        back = RateModel.from_json(path)
        assert back.relativity_label == "odds relativity"
        assert np.allclose(
            back.predict(df, exposure_col=None),
            run.rate_model.predict(df, exposure_col=None),
        )

    def test_the_exported_script_reproduces_it(self, data_path, tmp_path):
        p = binomial_project(data_path)
        run = run_model(p, prepare(p), "lapse")
        src = to_script(p, "lapse", run=run, output_prefix="lapse")
        assert "exposure_col=None" in src
        script = tmp_path / "lapse_rebuild.py"
        script.write_text(src)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=_env(),
        )
        assert proc.returncode == 0, proc.stderr
        rebuilt = RateModel.from_json(tmp_path / "lapse.easyglm")
        assert np.allclose(
            rebuilt.predict(prepare(p), exposure_col=None),
            run.rate_model.predict(prepare(p), exposure_col=None),
        )

    def test_a_link_that_is_not_multiplicative_is_still_refused(self, book):
        spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [40.0])})
        fit = fit_glm(
            book, spec, "ClaimNb", family="gaussian", link="identity", alpha=0.01
        )
        with pytest.raises(NotImplementedError, match="logit link"):
            to_rate_model(fit)


# --------------------------------------------------------------------------
# E4 — the target-loss-ratio base rate
# --------------------------------------------------------------------------
class TestSolveBaseRate:
    def _run(self, data_path):
        p = rate_change_project(data_path)
        df = prepare(p)
        return p, df, run_model(p, df, "change")

    def _expected(self, run, df) -> float:
        from easy_glm.workflow import totals

        _, expected, _ = totals(df, run.config, run.predict(df))
        return float(expected.sum())

    def test_it_hits_the_target_against_the_current_premium(self, data_path):
        p, df, run = self._run(data_path)
        target = 0.62
        p.models["change"].base_rate_override = solve_base_rate(run, df, target)
        run = run_model(p, df, "change")
        assert self._expected(run, df) / float(df["Premium"].sum()) == pytest.approx(
            target, rel=1e-10
        )

    def test_without_a_premium_it_balances_against_the_actual(self, data_path):
        p, df, run = self._run(data_path)
        p.data.roles["Premium"] = "ignore"  # no current premium any more
        p.models["change"].offset = None
        df = prepare(p)
        run = run_model(p, df, "change")
        p.models["change"].base_rate_override = solve_base_rate(run, df, 1.0)
        run = run_model(p, df, "change")
        actual = float(df["ClaimNb"].sum())
        assert self._expected(run, df) == pytest.approx(actual, rel=1e-10)

    def test_an_existing_override_does_not_change_the_answer(self, data_path):
        p, df, run = self._run(data_path)
        first = solve_base_rate(run, df, 0.62)
        p.models["change"].base_rate_override = 12345.0  # somebody's typo
        overridden = run_model(p, df, "change")
        assert overridden.rate_model.base_rate == 12345.0
        assert solve_base_rate(overridden, df, 0.62) == pytest.approx(first, rel=1e-12)

    def test_solving_twice_is_idempotent(self, data_path):
        p, df, run = self._run(data_path)
        p.models["change"].base_rate_override = solve_base_rate(run, df, 0.62)
        again = run_model(p, df, "change")
        assert solve_base_rate(again, df, 0.62) == pytest.approx(
            again.rate_model.base_rate, rel=1e-12
        )

    def test_weights_are_respected(self, data_path):
        """A frequency model: the expected total is per-unit prediction times
        exposure, so the solved base rate has to use the same weights."""
        p = rate_change_project(data_path)
        p.data.roles["Premium"] = "ignore"
        p.data.roles["Exposure"] = "weight"
        cfg = p.models["change"]
        cfg.offset = None
        cfg.weight = "Exposure"
        cfg.divide_target_by_weight = True
        df = prepare(p)
        run = run_model(p, df, "change")
        cfg.base_rate_override = solve_base_rate(run, df, 1.0)
        run = run_model(p, df, "change")
        pred = run.predict(df) * df["Exposure"].to_numpy()
        assert float(pred.sum()) == pytest.approx(float(df["ClaimNb"].sum()), rel=1e-10)
        assert run.rate_model.metadata.exposure_col == "Exposure"

    def test_an_explicit_weight_column_overrides_the_convention(self, data_path):
        p, df, run = self._run(data_path)
        value = solve_base_rate(run, df, 1.0, weight="Exposure", against="ClaimNb")
        scaled = value / run.rate_model.base_rate
        expected = float((run.predict(df) * df["Exposure"].to_numpy()).sum())
        assert scaled == pytest.approx(float(df["ClaimNb"].sum()) / expected, rel=1e-12)

    def test_a_binomial_model_is_refused(self, data_path):
        p = binomial_project(data_path)
        df = prepare(p)
        run = run_model(p, df, "lapse")
        with pytest.raises(ValueError, match="log-link"):
            solve_base_rate(run, df, 1.0)

    def test_a_target_that_is_not_positive_is_refused(self, data_path):
        p, df, run = self._run(data_path)
        with pytest.raises(ValueError, match="positive number"):
            solve_base_rate(run, df, 0.0)


# --------------------------------------------------------------------------
# F — the command line
# --------------------------------------------------------------------------
def cli(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ``easy-glm`` the way a user would, in its own process."""
    return subprocess.run(
        [sys.executable, "-m", "easy_glm.cli", *argv],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=_env(),
    )


@pytest.fixture
def cli_project(data_path, tmp_path) -> Path:
    """The fixture project on disk: a frequency model with a fixed alpha, so a
    fresh fit is reproducible."""
    p = Project(name="cli fixture")
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "DrivAge": "predictor",
        "Region": "predictor",
        "Premium": "ignore",
        "log_Premium_expected": "ignore",
        "Lapsed": "ignore",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.data.split.train_value = 1
    p.design.variables["DrivAge"] = VariableDesign(knots=[25.0, 40.0, 60.0])
    p.new_model(
        "freq",
        family="poisson",
        divide_target_by_weight=True,
        predictors=["DrivAge", "Region"],
    )
    p.models["freq"].penalty.alpha = 0.001
    p.models["freq"].penalty.cv = None
    path = tmp_path / "project.json"
    p.to_json(path)
    return path


class TestCliValidate:
    def test_a_good_project_is_reported_valid(self, cli_project):
        proc = cli("validate", str(cli_project))
        assert proc.returncode == 0, proc.stderr
        assert "valid" in proc.stdout and "freq" in proc.stdout

    def test_an_invalid_project_exits_non_zero_with_the_problems(
        self, cli_project, tmp_path
    ):
        p = Project.from_json(cli_project)
        p.models["freq"].predictors = []  # no predictors
        p.models["freq"].target = None  # and no target
        bad = tmp_path / "bad.json"
        p.to_json(bad)
        proc = cli("validate", str(bad))
        assert proc.returncode == 1
        assert "no predictors" in proc.stderr and "no target column" in proc.stderr
        assert "Traceback" not in proc.stderr

    def test_a_missing_project_file_is_a_message(self, tmp_path):
        proc = cli("validate", str(tmp_path / "nope.json"))
        assert proc.returncode == 1
        assert "no project file" in proc.stderr and "Traceback" not in proc.stderr

    def test_a_file_that_is_not_a_project_is_a_message(self, tmp_path):
        junk = tmp_path / "junk.json"
        junk.write_text("this is not JSON")
        proc = cli("validate", str(junk))
        assert proc.returncode == 1
        assert "not a readable easy_glm project" in proc.stderr

    def test_a_missing_data_file_is_one_of_the_problems(self, cli_project, tmp_path):
        p = Project.from_json(cli_project)
        p.data.source.path = str(tmp_path / "gone.parquet")
        bad = tmp_path / "nodata.json"
        p.to_json(bad)
        proc = cli("validate", str(bad))
        assert proc.returncode == 1
        assert "the data cannot be prepared" in proc.stderr

    def test_a_predictor_that_is_not_in_the_data_is_caught_without_fitting(
        self, cli_project, tmp_path
    ):
        p = Project.from_json(cli_project)
        p.data.roles["Nonexistent"] = "predictor"
        p.models["freq"].predictors.append("Nonexistent")
        bad = tmp_path / "ghost.json"
        p.to_json(bad)
        proc = cli("validate", str(bad))
        assert proc.returncode == 1
        assert "Nonexistent" in proc.stderr


class TestCliRun:
    def test_it_writes_every_artefact_and_prints_a_summary(self, cli_project, tmp_path):
        out = tmp_path / "artefacts"
        proc = cli("run", str(cli_project), "--out", str(out))
        assert proc.returncode == 0, proc.stderr
        names = sorted(f.name for f in out.iterdir())
        assert names == [
            "cli fixture_freq.easyglm",
            "cli fixture_freq.py",
            "cli fixture_freq_rate_tables.xlsx",
            "cli fixture_freq_report.html",
        ]
        assert "holdout" in proc.stdout and "A/E" in proc.stdout
        assert "base rate" in proc.stdout

    def test_the_report_is_self_contained(self, cli_project, tmp_path):
        out = tmp_path / "artefacts"
        assert cli("run", str(cli_project), "--out", str(out)).returncode == 0
        html = (out / "cli fixture_freq_report.html").read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert 'src="http' not in html and 'href="http' not in html

    def test_the_written_script_rebuilds_the_same_model(self, cli_project, tmp_path):
        """The script is run in a folder of its own: it writes artefacts under
        the same names, so running it next to the CLI's would overwrite them."""
        out = tmp_path / "artefacts"
        assert cli("run", str(cli_project), "--out", str(out)).returncode == 0
        rebuild = tmp_path / "rebuild"
        rebuild.mkdir()
        script = rebuild / "rebuild.py"
        script.write_text((out / "cli fixture_freq.py").read_text())
        proc = subprocess.run(
            [sys.executable, "rebuild.py"],
            cwd=str(rebuild),
            capture_output=True,
            text=True,
            env=_env(),
        )
        assert proc.returncode == 0, proc.stderr
        df = prepare(Project.from_json(cli_project))
        from_cli = RateModel.from_json(out / "cli fixture_freq.easyglm")
        from_script = RateModel.from_json(rebuild / "cli fixture_freq.easyglm")
        assert np.allclose(
            from_cli.predict(df, exposure_col=None),
            from_script.predict(df, exposure_col=None),
            rtol=1e-12,
            atol=0.0,
        )

    def test_the_easyglm_file_is_the_workbench_model(self, cli_project, tmp_path):
        """The scorer the CLI writes is `to_rate_model` from the same fit.

        Not compared byte for byte, for two reasons. Every snapshot carries the
        wall-clock time it was written, so two runs a second apart differ in
        those strings. And glum's solver is not bit-reproducible: two fits of
        the same model in the *same* process already disagree in the last
        floating-point digit (measured: 9e-16 on a relativity), because the
        linear algebra underneath sums in whatever order its threads finish in.
        So the comparison is: the same structure, the same keys, every number
        within 1e-12 relative — and predictions within 1e-12 too.
        """
        out = tmp_path / "artefacts"
        assert cli("run", str(cli_project), "--out", str(out)).returncode == 0
        p = Project.from_json(cli_project)
        df = prepare(p)
        here = run_model(p, df, "freq")
        mine = tmp_path / "mine.easyglm"
        here.rate_model.to_json(mine)
        theirs = out / "cli fixture_freq.easyglm"
        _assert_same_model_json(
            json.loads(theirs.read_text()), json.loads(mine.read_text())
        )
        assert np.allclose(
            RateModel.from_json(theirs).predict(df, exposure_col=None),
            here.rate_model.predict(df, exposure_col=None),
            rtol=1e-12,
            atol=0.0,
        )

    def test_the_output_folder_is_created(self, cli_project, tmp_path):
        out = tmp_path / "deep" / "nested"
        assert cli("run", str(cli_project), "--out", str(out)).returncode == 0
        assert out.is_dir() and any(out.iterdir())

    def test_a_named_model_is_used(self, cli_project, tmp_path):
        p = Project.from_json(cli_project)
        p.models["other"] = p.models["freq"]
        p.to_json(cli_project)
        out = tmp_path / "named"
        proc = cli("run", str(cli_project), "--model", "other", "--out", str(out))
        assert proc.returncode == 0, proc.stderr
        assert (out / "cli fixture_other.easyglm").exists()

    def test_an_unknown_model_name_is_a_message(self, cli_project, tmp_path):
        proc = cli("run", str(cli_project), "--model", "nope", "--out", str(tmp_path))
        assert proc.returncode == 1
        assert "no model named 'nope'" in proc.stderr

    def test_an_invalid_project_exits_non_zero(self, cli_project, tmp_path):
        p = Project.from_json(cli_project)
        p.models["freq"].predictors = []
        bad = tmp_path / "bad.json"
        p.to_json(bad)
        proc = cli("run", str(bad), "--out", str(tmp_path / "never"))
        assert proc.returncode == 1
        assert "cannot be fitted" in proc.stderr
        assert not (tmp_path / "never").exists()


class TestCliExport:
    def test_each_flag_writes_its_own_artefact(self, cli_project, tmp_path):
        out = tmp_path / "one"
        assert (
            cli("export", str(cli_project), "--script", "--out", str(out)).returncode
            == 0
        )
        assert [f.name for f in out.iterdir()] == ["cli fixture_freq.py"]

    def test_two_flags_write_two_artefacts(self, cli_project, tmp_path):
        out = tmp_path / "two"
        proc = cli("export", str(cli_project), "--report", "--excel", "--out", str(out))
        assert proc.returncode == 0, proc.stderr
        assert sorted(f.name for f in out.iterdir()) == [
            "cli fixture_freq_rate_tables.xlsx",
            "cli fixture_freq_report.html",
        ]

    def test_no_flag_at_all_is_refused(self, cli_project, tmp_path):
        proc = cli("export", str(cli_project), "--out", str(tmp_path / "none"))
        assert proc.returncode == 1
        assert "at least one of --script, --report, --excel" in proc.stderr

    def test_the_exported_script_has_the_resolved_design(self, cli_project, tmp_path):
        out = tmp_path / "script"
        assert (
            cli("export", str(cli_project), "--script", "--out", str(out)).returncode
            == 0
        )
        src = (out / "cli fixture_freq.py").read_text()
        assert "StepEncoder('DrivAge', [25, 40, 60]" in src
        assert "DesignSpec.from_data" not in src  # the design is written out
        assert "alpha=0.001" in src


class TestCliWorkbench:
    def test_it_checks_the_project_before_launching(self, tmp_path):
        proc = cli("workbench", str(tmp_path / "missing.json"))
        assert proc.returncode == 1
        assert "no project file" in proc.stderr

    def test_it_delegates_to_the_app_launcher(self, cli_project, monkeypatch):
        """Driven in-process: launching a real Streamlit server from a unit test
        would leave a port open."""
        import easy_glm.app as app
        import easy_glm.cli as cli_mod

        seen: dict[str, object] = {}

        class FakeProc:
            returncode = 0

        def fake_launch(path, *, port, block, headless):
            seen.update(path=path, port=port, block=block, headless=headless)
            return FakeProc()

        monkeypatch.setattr(app, "launch", fake_launch)
        code = cli_mod.main(
            ["workbench", str(cli_project), "--port", "8599", "--headless"]
        )
        assert code == 0
        assert seen == {
            "path": str(cli_project),
            "port": 8599,
            "block": True,
            "headless": True,
        }


def _assert_same_model_json(a, b, path: str = "") -> None:
    """Two ``.easyglm`` documents describe the same model: identical structure,
    numbers equal to 1e-12 relative, snapshot timestamps ignored."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"different keys at {path or '<root>'}"
        for key in a:
            if key == "timestamp":
                continue
            _assert_same_model_json(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"different lengths at {path}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same_model_json(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-12, abs=0.0), path
    else:
        assert a == b, path
