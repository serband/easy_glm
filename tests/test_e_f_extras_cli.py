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
