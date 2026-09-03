"""Tests for the workflow engine: Project spec, prep, explore, run, diagnostics, export."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from easy_glm.core.design import CategoricalEncoder, StepEncoder
from easy_glm.engine import RateModel
from easy_glm.workflow import (
    Derived,
    Interaction,
    Project,
    Recode,
    VariableDesign,
    ae_by_variable,
    alpha_path,
    apply_variables,
    build_design,
    column_summary,
    double_lift,
    eval_expr,
    gini,
    leakage_report,
    lift_table,
    null_model_predict,
    predictions_effectively_equal,
    prepare,
    residual_factor_search,
    run_model,
    to_script,
    totals,
    train_holdout,
    univariate,
)


@pytest.fixture(scope="module")
def raw_frame() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 6000
    age = rng.integers(18, 80, n).astype(float)
    bm = rng.integers(50, 200, n).astype(float)
    region = rng.choice(
        ["R1", "R2", "R3", "R4", "Rare"], n, p=[0.45, 0.3, 0.14, 0.1, 0.01]
    ).astype(object)
    gas = rng.choice(["Regular", "Diesel"], n)
    lic = rng.choice(["Q", "M"], n)
    exp_years = rng.integers(0, 30, n).astype(float)
    exposure = rng.uniform(0.1, 1.0, n)
    mu = np.exp(
        -2.0
        - 0.03 * np.maximum(45 - age, 0)
        + 0.004 * (bm - 100)
        + np.where(region == "R1", 0.0, 0.25)
        + np.where(gas == "Diesel", 0.15, 0.0)
    )
    claims = rng.poisson(mu * exposure).astype(float)
    age[rng.random(n) < 0.04] = np.nan
    region[rng.random(n) < 0.02] = None
    leak = claims / exposure * (1 + 0.01 * rng.normal(size=n))
    claim_cost = claims * rng.gamma(2.0, 500.0, n)
    return pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": claims,
            "Exposure": exposure,
            "DrivAge": age,
            "BM": bm,
            "Region": region,
            "VehGas": gas,
            "Lic": lic,
            "Exp": exp_years,
            "Leak": leak,
            "claim_cost": claim_cost,
        }
    ).with_columns(pl.col("DrivAge").fill_nan(None))


@pytest.fixture(scope="module")
def data_path(raw_frame, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "policies.parquet"
    raw_frame.write_parquet(path)
    return path


@pytest.fixture
def project(data_path) -> Project:
    p = Project(name="unit")
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.renames = {"BM": "BonusMalus"}
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Region": "predictor",
        "VehGas": "predictor",
        "Exp_Q": "predictor",
        "Leak": "predictor",
        "claim_cost": "predictor",
        "Lic": "ignore",
        "Exp": "ignore",
    }
    p.data.recodes = {"Region": Recode(mapping={"R4": "R3"}, default=None)}
    p.data.derived = [
        Derived(
            "Exp_Q", "pl.when(pl.col('Lic') == 'Q').then(pl.col('Exp')).otherwise(0.0)"
        )
    ]
    p.data.filters = ["pl.col('Exposure') > 0.15"]
    p.data.split.mode = "random"
    p.data.split.column = "traintest"
    p.data.split.fraction = 0.7
    p.data.split.seed = 3
    p.design.variables["DrivAge"] = VariableDesign(
        monotone="decreasing", knots=[30, 45, 60]
    )
    p.design.variables["Exp_Q"] = VariableDesign(knots="integer")
    p.new_model(
        "freq",
        family="poisson",
        divide_target_by_weight=True,
        predictors=["DrivAge", "BonusMalus", "Region", "VehGas", "Exp_Q"],
    )
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    p.models["freq"].interactions = [
        Interaction("DrivAge", "Region", min_cell_exposure=0.02)
    ]
    return p


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------
class TestProject:
    def test_roles_and_new_model(self, project):
        assert project.target == "ClaimNb" and project.weight == "Exposure"
        assert project.exposure is None
        assert "Leak" in project.predictors
        cfg = project.models["freq"]
        assert cfg.target == "ClaimNb" and cfg.weight == "Exposure"
        assert project.champion == "freq"

    def test_set_role_keeps_single_roles_unique(self, project):
        project.set_role("Exp", "target")
        assert project.columns_with_role("target") == ["Exp"]
        assert project.data.roles["ClaimNb"] == "ignore"
        with pytest.raises(ValueError):
            project.set_role("Exp", "hero")
        project.set_role("IDpol", "split")
        assert (
            project.data.split.column == "IDpol" and project.data.split.mode == "column"
        )

    def test_validate(self, project):
        assert project.validate() == []
        project.models["freq"].predictors.append("Lic")  # role = ignore
        project.models["freq"].penalty.alpha = None
        project.models["freq"].penalty.cv = None
        problems = project.validate("freq")
        assert any("not predictor-role" in p for p in problems)
        assert any("alpha or cv" in p for p in problems)
        assert any("No model named" in p for p in project.validate("nope"))

    def test_adjustment_cell_flag_must_match_the_table_type(self, project):
        from easy_glm.workflow import Adjustment
        from easy_glm.workflow.run import apply_adjustments

        df = prepare(project)
        run = run_model(project, df, "freq")
        cfg = project.models["freq"]
        cfg.adjustments = [Adjustment("Region", "R2", "R2", 1.1, cell=True)]
        with pytest.raises(ValueError, match="main effect"):
            apply_adjustments(run.rate_model.clone(), cfg)
        cfg.adjustments = [Adjustment("DrivAge×Region", None, 30.0, 1.1)]
        with pytest.raises(ValueError, match="cell=True"):
            apply_adjustments(run.rate_model.clone(), cfg)
        # a variable the model does not have is an AdjustmentError like every
        # other refusal (it used to be a bare KeyError, which tracebacked the
        # page instead of being dropped and reported — D5 review S1)
        from easy_glm.workflow import AdjustmentError

        cfg.adjustments = [Adjustment("Nope", 1, 2, 1.1)]
        with pytest.raises(AdjustmentError, match="not a variable"):
            apply_adjustments(run.rate_model.clone(), cfg)
        cfg.adjustments = []

    def test_model_hash_reacts_to_interaction_settings_not_cell_edits(self, project):
        from easy_glm.app.state import model_hash
        from easy_glm.workflow import Adjustment

        base = model_hash(project, "freq")
        it = project.models["freq"].interactions[0]
        it.min_cell_exposure = 0.03
        assert model_hash(project, "freq") != base
        it.min_cell_exposure = 0.02
        it.penalty_weight = 2.0
        assert model_hash(project, "freq") != base
        it.penalty_weight = 1.0
        assert model_hash(project, "freq") == base
        project.models["freq"].adjustments.append(
            Adjustment(
                "DrivAge×Region", None, 30.0, 1.1, from_b="R2", to_b="R2", cell=True
            )
        )
        assert model_hash(project, "freq") == base  # adjustments never force a refit
        project.models["freq"].adjustments.clear()
        removed = project.models["freq"].interactions.pop()
        assert model_hash(project, "freq") != base
        project.models["freq"].interactions.append(removed)

    def test_validate_interactions(self, project):
        cfg = project.models["freq"]
        cfg.interactions = [
            Interaction("DrivAge", "DrivAge"),
            Interaction("Region", "Lic"),
            Interaction("DrivAge", "Region"),
            Interaction("Region", "DrivAge"),
            Interaction("BonusMalus", "Region", min_cell_exposure=1.5),
        ]
        problems = project.validate("freq")
        assert any("× itself" in p for p in problems)
        assert any("'Lic' is not one of" in p for p in problems)
        assert any("listed twice" in p for p in problems)
        assert any("min_cell_exposure" in p for p in problems)

    def test_a_non_numeric_field_is_reported_not_raised(self, project):
        """Breaker #3: a hand-edited project.json can put anything in a
        numeric field (``"abc"``, a boolean, ``NaN``, the wrong shape). Every
        comparison below used to assume a real number and raised
        ``TypeError``/``ValueError`` straight out of ``validate()`` — which the
        CLI and the Model/Design pages call unguarded, so this was a crash,
        not a message. Each case must come back as one more problem string."""
        cfg = project.models["freq"]

        cfg.penalty.alpha = "abc"
        assert any("alpha must be > 0" in p for p in project.validate("freq"))
        cfg.penalty.alpha = float("nan")
        assert any("alpha must be > 0" in p for p in project.validate("freq"))
        cfg.penalty.alpha = 0.002  # restore

        cfg.family = "tweedie"
        cfg.tweedie_power = "abc"
        assert any(
            "tweedie_power must be strictly between" in p
            for p in project.validate("freq")
        )
        cfg.family = "poisson"
        cfg.tweedie_power = 1.5  # restore

        it = cfg.interactions[0]
        it.penalty_weight = "abc"
        assert any("penalty_weight must be >= 0" in p for p in project.validate("freq"))
        it.penalty_weight = 1.0
        it.penalty_weight = 0.0
        assert project.validate("freq") == []
        it.penalty_weight = 1.0
        it.min_cell_exposure = "abc"
        assert any("min_cell_exposure" in p for p in project.validate("freq"))
        it.min_cell_exposure = 0.02
        it.alpha = "abc"
        assert any(
            "alpha must be > 0 (leave it unset" in p for p in project.validate("freq")
        )
        it.alpha = None

        project.design.variables["DrivAge"].clamp = ["abc", 10]
        assert any("clamp must be" in p for p in project.validate("freq"))
        project.design.variables["DrivAge"].clamp = [1, "abc"]
        assert any("clamp must be" in p for p in project.validate("freq"))
        project.design.variables["DrivAge"].clamp = 5  # not even a pair
        assert any("clamp must be" in p for p in project.validate("freq"))
        project.design.variables["DrivAge"].clamp = None

        project.data.split.fraction = "abc"
        assert any("split.fraction" in p for p in project.validate("freq"))
        project.data.split.fraction = 0.7

        assert project.validate("freq") == []  # everything restored cleanly

    def test_json_roundtrip(self, project, tmp_path):
        project.models["freq"].adjustments.append(
            __import__("easy_glm.workflow", fromlist=["Adjustment"]).Adjustment(
                "Region", "R2", "R2", 1.1
            )
        )
        project.to_json(tmp_path / "p.json")
        back = Project.from_json(tmp_path / "p.json")
        assert back.to_dict() == project.to_dict()
        assert back.models["freq"].adjustments[0].from_ == "R2"
        assert back.models["freq"].interactions[0].name == "DrivAge×Region"
        cell = __import__("easy_glm.workflow", fromlist=["Adjustment"]).Adjustment(
            "DrivAge×Region", None, 30.0, 0.9, from_b="R2", to_b="R2", cell=True
        )
        with_cell = project.copy()
        with_cell.models["freq"].adjustments.append(cell)
        back2 = Project.from_dict(with_cell.to_dict())
        got = back2.models["freq"].adjustments[-1]
        assert got.cell and got.from_b == "R2" and got.to_ == 30.0
        assert back2.to_dict() == with_cell.to_dict()
        assert back.data.recodes["Region"].mapping == {"R4": "R3"}
        assert back.design.variables["DrivAge"].monotone == "decreasing"
        assert back.copy().to_dict() == project.to_dict()
        with pytest.raises(ValueError, match="newer"):
            Project.from_dict({"version": 99})


# --------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------
class TestPrep:
    def test_prepare_applies_every_step(self, project, raw_frame):
        df = prepare(project)
        assert "BonusMalus" in df.columns and "BM" not in df.columns
        assert "R4" not in df["Region"].drop_nulls().unique().to_list()
        assert df["Region"].dtype == pl.Utf8
        assert df["Region"].null_count() > 0  # default=None keeps nulls
        assert "Exp_Q" in df.columns
        q = df.filter(pl.col("Lic") == "Q")
        assert (q["Exp_Q"] == q["Exp"]).all()
        assert (df.filter(pl.col("Lic") == "M")["Exp_Q"] == 0).all()
        assert df["Exposure"].min() > 0.15
        assert set(df["traintest"].unique().to_list()) == {0, 1}
        assert abs(df["traintest"].mean() - 0.7) < 0.03
        again = prepare(project)
        assert again["traintest"].equals(
            df["traintest"]
        )  # seeded split is deterministic

    def test_apply_variables_with_default_and_types(self, raw_frame):
        cfg = Project().data
        cfg.recodes = {"Region": Recode(mapping={"R1": "A"}, default="Other")}
        cfg.types = {"BM": "categorical"}
        out = apply_variables(raw_frame, cfg)
        assert set(out["Region"].drop_nulls().unique().to_list()) == {"A", "Other"}
        assert out["BM"].dtype == pl.Utf8

    def test_eval_expr_guards(self):
        assert isinstance(eval_expr("pl.col('x') * 2"), pl.Expr)
        with pytest.raises(ValueError, match="not a polars expression"):
            eval_expr("1 + 1")
        with pytest.raises(ValueError):
            eval_expr("__import__('os').getcwd()")
        with pytest.raises(ValueError):
            eval_expr("pl.col('x'")

    def test_column_summary(self, raw_frame):
        s = column_summary(raw_frame)
        assert set(s["column"]) == set(raw_frame.columns)
        assert s.filter(pl.col("column") == "IDpol")["n_unique"][0] == raw_frame.height

    def test_split_column_mode(self, project, raw_frame):
        project.data.split.mode = "column"
        project.data.split.column = "Lic"
        project.data.split.train_value = "Q"
        df = prepare(project)
        assert (df["Lic"] == 1).sum() == (
            raw_frame.filter(pl.col("Exposure") > 0.15)["Lic"] == "Q"
        ).sum()


# --------------------------------------------------------------------------
# explore
# --------------------------------------------------------------------------
class TestExplore:
    def test_univariate_numeric_and_categorical(self, project):
        df = prepare(project)
        u = univariate(
            df,
            "DrivAge",
            target="ClaimNb",
            weight="Exposure",
            divide_target_by_weight=True,
        )
        t = u["table"]
        assert u["kind"] == "numeric" and u["null_share"] > 0
        assert t["share"].sum() == pytest.approx(1.0)
        assert t["label"][-1] == "Other / Unknown"  # same label as the rate tables
        assert t["rate"].drop_nulls().min() >= 0
        c = univariate(
            df,
            "Region",
            target="ClaimNb",
            weight="Exposure",
            divide_target_by_weight=True,
        )
        assert c["kind"] == "categorical"
        assert c["table"]["label"][0] == "R1"  # most exposed first
        assert "Other / Unknown" in c["table"]["label"].to_list()
        many = univariate(
            df.with_columns(pl.col("IDpol").cast(pl.Utf8)), "IDpol", max_levels=5
        )
        assert many["table"].height == 6 and many["table"]["label"][-1].startswith(
            "(other"
        )

    def test_leakage_report_flags_planted_leaks(self, project):
        project.exploration["leakage"]["acknowledged"].append("VehGas")
        df = prepare(project)
        rep = leakage_report(df, project, sample_rows=4000)
        rec = dict(zip(rep["variable"], rep["recommendation"], strict=True))
        flags = dict(zip(rep["variable"], rep["flags"], strict=True))
        assert rec["IDpol"] == "ignore" and "identifier-like" in flags["IDpol"]
        assert rec["Leak"] == "ignore"
        assert "deviance" in flags["Leak"] or "proxy" in flags["Leak"]
        assert rec["claim_cost"] in ("ignore", "check")
        assert "post-outcome name" in flags["claim_cost"]
        assert rec["DrivAge"] == "ok" and rec["BonusMalus"] == "ok"
        assert rec["VehGas"] == "acknowledged"
        assert rep["score"].is_sorted(descending=True)
        assert "Lic" not in rec  # role = ignore is not a candidate


# --------------------------------------------------------------------------
# design + run
# --------------------------------------------------------------------------
class TestRun:
    def test_build_design_honours_overrides(self, project):
        df = prepare(project)
        train = df.filter(pl.col("traintest") == 1)
        project.design.variables["BonusMalus"] = VariableDesign(knots=[80, 100, 120])
        project.design.variables["VehGas"] = VariableDesign(
            levels=["Diesel", "Regular"]
        )
        project.design.variables["DrivAge"] = VariableDesign(
            kind="categorical", max_levels=5
        )
        spec = build_design(
            project,
            train,
            ["BonusMalus", "VehGas", "DrivAge", "Exp_Q"],
            weight_col="Exposure",
        )
        assert spec["BonusMalus"].knots == [80.0, 100.0, 120.0]
        assert spec["VehGas"].reference == "Diesel"
        assert (
            isinstance(spec["DrivAge"], CategoricalEncoder)
            and len(spec["DrivAge"].levels) == 5
        )
        exp_q = spec["Exp_Q"]
        assert isinstance(exp_q, StepEncoder)
        assert exp_q.knots == [
            float(k) for k in range(1, 30)
        ]  # integer knots above the minimum

    def test_run_model_metrics_and_exactness(self, project):
        df = prepare(project)
        run = run_model(project, df, "freq")
        assert set(run.metrics) == {"train", "holdout"}
        for m in run.metrics.values():
            assert 0.8 < m["ae"] < 1.2
            assert 0 <= m["deviance_explained"] < 1
            assert -1 <= m["gini"] <= 1
        assert run.train_rows + run.holdout_rows == df.height
        # rate model reproduces the GLM
        holdout = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            run.predict(holdout), run.fit.predict(holdout), rtol=1e-10
        )
        # monotone from the design level was applied
        assert run.fit.monotone == {"DrivAge": "decreasing"}
        assert "DrivAge×Region" in run.spec.variables
        assert run.rate_model.variables["DrivAge×Region"].type == "interaction"
        s = run.summary()
        assert s["name"] == "freq" and s["non_zero"] <= s["features"]
        assert set(run.tables) == set(project.models["freq"].predictors) | {
            "DrivAge×Region"
        }

    def test_run_model_fits_interactions_in_two_stages(self, project):
        """A2 / Q5: with an interaction the run holds a two-stage fit; the main
        tables and the base rate are the ones the same model without the
        interaction produces, and the cells are adjustments on top."""
        import copy

        from easy_glm.core.fit import TwoStageFit
        from easy_glm.workflow.project import ModelConfig

        df = prepare(project)
        run = run_model(project, df, "freq")
        assert isinstance(run.fit, TwoStageFit)
        assert run.alpha_stage2 == pytest.approx(run.alpha)  # no override set
        assert run.cells_kept == len(run.spec["DrivAge×Region"].cells) > 0

        cfg = project.models["freq"]
        without = ModelConfig(**{**copy.deepcopy(cfg.__dict__), "interactions": []})
        project.models["no_inter"] = without
        plain = run_model(project, df, "no_inter")
        assert plain.alpha_stage2 is None
        # 1e-13 is glum's own run-to-run noise (two identical fits differ by
        # about 1e-15 on a relativity), not a modelling difference: the joint
        # fit used to move these tables by several per cent
        for var in plain.spec.main_effects:
            np.testing.assert_allclose(
                run.tables[var]["relativity"].to_numpy(),
                plain.tables[var]["relativity"].to_numpy(),
                rtol=1e-13,
            )
        assert run.rate_model.base_rate == pytest.approx(
            plain.rate_model.base_rate, rel=1e-13
        )
        # both alphas are on the record
        model_metrics_entry = run.rate_model.snapshots[-1].metrics["model"]
        assert model_metrics_entry["stages"] == 2
        assert model_metrics_entry["alpha_stage2"] == run.alpha_stage2
        assert plain.rate_model.snapshots[-1].metrics["model"]["stages"] == 1
        assert run.summary()["cells_kept"] == run.cells_kept

    def test_interaction_alpha_overrides_the_second_stage(self, project):
        df = prepare(project)
        cfg = project.models["freq"]
        base = run_model(project, df, "freq")
        cfg.interactions[0].alpha = 0.5
        assert project.validate("freq") == []
        harder = run_model(project, df, "freq")
        assert harder.alpha_stage2 == pytest.approx(0.5)
        assert harder.alpha == pytest.approx(base.alpha)  # the mains do not move
        np.testing.assert_allclose(
            harder.fit.stage1.coef, base.fit.stage1.coef, rtol=1e-12
        )
        assert np.abs(harder.fit.stage2.coef).max() < np.abs(base.fit.stage2.coef).max()
        cfg.interactions[0].alpha = -1.0
        assert any("alpha must be > 0" in m for m in project.validate("freq"))

    def test_stage2_alpha_is_the_largest_any_interaction_asks_for(self, project):
        """The second stage is one fit with one alpha, so several interactions
        asking for different ones resolve to the most cautious."""
        from easy_glm.workflow.project import Interaction
        from easy_glm.workflow.run import stage2_alpha

        cfg = project.models["freq"]
        assert stage2_alpha(cfg) is None  # nothing asked: follow the mains
        cfg.interactions = [
            Interaction("DrivAge", "Region", alpha=0.1),
            Interaction("BonusMalus", "Region", alpha=0.7),
            Interaction("VehGas", "Region"),
        ]
        assert stage2_alpha(cfg) == 0.7
        cfg.interactions[1].alpha = None
        assert stage2_alpha(cfg) == 0.1

    def test_rebuild_rate_model_keeps_the_two_stages(self, project):
        from easy_glm.workflow import Adjustment, rebuild_rate_model

        df = prepare(project)
        run = run_model(project, df, "freq")
        holdout = df.filter(pl.col("traintest") == 0)
        before = run.predict(holdout)
        project.models["freq"].adjustments.append(
            Adjustment("VehGas", "Diesel", "Diesel", 1.5)
        )
        again = rebuild_rate_model(project, run, df)
        assert again.fit is run.fit  # no refit
        diesel = holdout["VehGas"].to_numpy() == "Diesel"
        np.testing.assert_allclose(again.predict(holdout)[~diesel], before[~diesel])
        np.testing.assert_allclose(
            again.predict(holdout)[diesel], before[diesel] * 1.5, rtol=1e-10
        )
        assert again.rate_model.snapshots[-1].metrics["model"]["stages"] == 2

    def test_run_model_applies_adjustments(self, project):
        from easy_glm.workflow import Adjustment

        df = prepare(project)
        base = run_model(project, df, "freq")
        project.models["freq"].adjustments.append(
            Adjustment("VehGas", "Diesel", "Diesel", 2.0)
        )
        adj = run_model(project, df, "freq")
        holdout = df.filter(pl.col("traintest") == 0)
        diesel = holdout.filter(pl.col("VehGas") == "Diesel")
        petrol = holdout.filter(pl.col("VehGas") == "Regular")
        ratio = adj.predict(diesel) / base.predict(diesel)
        np.testing.assert_allclose(ratio, ratio[0])
        np.testing.assert_allclose(adj.predict(petrol), base.predict(petrol))
        assert len(adj.rate_model.snapshots) == 2

    def test_run_model_with_cv(self, project):
        project.models["freq"].penalty.alpha = None
        project.models["freq"].penalty.cv = 2
        project.models["freq"].penalty.n_alphas = 4
        df = prepare(project)
        run = run_model(project, df, "freq")
        path = alpha_path(run.fit)
        # the model has an interaction, so there are two stages and two paths
        assert path["stage"].unique().sort().to_list() == [1, 2]
        for stage in (1, 2):
            sub = path.filter(pl.col("stage") == stage)
            assert sub.height == 4 and sub["selected"].sum() == 1
            assert sub["cv_deviance"].null_count() == 0
        # Stage 2 cross-validates on its own path over its own columns, with
        # exactly the user's CV configuration rather than stage 1's alpha.
        assert run.fit.stage1.model.cv == run.fit.stage2.model.cv == 2
        assert run.fit.stage1.model.n_alphas == run.fit.stage2.model.n_alphas == 4
        assert run.alpha_stage2 != pytest.approx(run.alpha)
        assert run.alpha_stage2 is not None
        assert run.summary()["alpha_stage2"] == run.alpha_stage2

    def test_run_model_rejects_invalid(self, project):
        project.models["freq"].predictors = []
        with pytest.raises(ValueError, match="no predictors"):
            run_model(project, prepare(project), "freq")


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
class TestDiagnostics:
    def test_prediction_equivalence_detects_exact_near_and_different_vectors(self):
        prediction = np.array([0.0, 0.02, 1.5, 100.0])
        assert predictions_effectively_equal(prediction, prediction.copy())
        assert predictions_effectively_equal(
            prediction, prediction + np.array([1e-13, 1e-12, 1e-10, 1e-8])
        )
        assert not predictions_effectively_equal(
            prediction, prediction + np.array([0.0, 0.0, 0.0, 1e-4])
        )

    def test_null_benchmark_is_calibrated_on_train_and_does_not_read_holdout_target(
        self, project
    ):
        df = prepare(project)
        train, holdout = train_holdout(df, project.data.split)
        cfg = project.models["freq"]
        pred = null_model_predict(project, cfg, train, holdout)
        changed_outcomes = holdout.with_columns(pl.lit(999.0).alias(cfg.target))
        assert null_model_predict(
            project, cfg, train, changed_outcomes
        ) == pytest.approx(pred)
        train_pred = null_model_predict(project, cfg, train, train)
        actual, expected, _weight = totals(train, cfg, train_pred)
        assert expected.sum() == pytest.approx(actual.sum(), rel=1e-8)

    def test_lift_gini_double_lift(self):
        rng = np.random.default_rng(0)
        n = 5000
        w = rng.uniform(0.5, 1.5, n)
        rate = np.exp(rng.normal(-2, 0.6, n))
        actual = rng.poisson(rate * w).astype(float)
        expected = rate * w
        lt = lift_table(actual, expected, w, n_bins=10)
        assert lt.height == 10
        assert lt["actual"].sum() == pytest.approx(actual.sum())
        assert lt["expected"].sum() == pytest.approx(expected.sum())
        assert lt["expected_rate"].is_sorted()
        assert lt["cum_exposure_share"][-1] == pytest.approx(1.0)
        g_model = gini(actual, expected, w)
        g_perfect = gini(actual, actual, w)
        g_random = gini(actual, rng.random(n) * w, w)
        assert g_perfect == pytest.approx(1.0)
        assert 0.2 < g_model < 1.0
        assert abs(g_random) < 0.1
        dl = double_lift(
            actual, expected, expected * rng.uniform(0.5, 1.5, n), w, n_bins=5
        )
        assert dl.height == 5 and dl["mean_ratio"].is_sorted()
        assert dl["actual"].sum() == pytest.approx(actual.sum())

    def test_ae_by_variable_and_residual_search(self, project):
        df = prepare(project)
        run = run_model(project, df, "freq")
        holdout = df.filter(pl.col("traintest") == 0)
        actual, expected, w = totals(holdout, run.config, run.predict(holdout))
        tbl = ae_by_variable(holdout, "DrivAge", actual, expected, w)
        assert tbl["actual"].sum() == pytest.approx(actual.sum())
        assert tbl["label"][-1] == "Other / Unknown"
        cat = ae_by_variable(holdout, "Region", actual, expected, w)
        assert cat["exposure"].sum() == pytest.approx(w.sum())
        # plant a missing factor: inflate actuals for Lic == 'Q'
        planted = np.where(holdout["Lic"].to_numpy() == "Q", actual * 3.0, actual)
        noise = holdout.with_columns(
            pl.Series("Noise", np.random.default_rng(1).random(holdout.height))
        )
        search = residual_factor_search(
            noise, ["Lic", "Noise", "Exp"], planted, expected, w
        )
        assert search["variable"][0] == "Lic"

    def test_alpha_path_fixed_alpha(self, project):
        run = run_model(project, prepare(project), "freq")
        path = alpha_path(run.fit)
        # one row per stage: the mains' fixed alpha, and the cells' (the same,
        # since no interaction of this model asks for its own)
        assert path.height == 2 and path["stage"].to_list() == [1, 2]
        assert path["alpha"].to_list() == pytest.approx([0.002, 0.002])


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
class TestExport:
    def test_script_without_run_mentions_from_data(self, project):
        src = to_script(project, "freq")
        assert "DesignSpec.from_data" in src
        # this model has an interaction, so the fit is the two-stage one (which
        # stage-by-stage form the script takes is asserted below)
        assert "fit_two_stage(" in src

    def test_unfitted_cv_interaction_script_preserves_stage_two_cv(self, project):
        cfg = project.models["freq"]
        cfg.penalty.alpha = None
        cfg.penalty.cv = 2
        cfg.penalty.n_alphas = 4
        src = to_script(project, "freq")
        assert "fit = fit_two_stage(" in src
        assert "cv=2, n_alphas=4" in src
        assert "second stage independently evaluates the same folds" in src
        assert (
            "replace_strict" in src
            and "Exp_Q" in src
            and "pl.col('Exposure') > 0.15" in src
        )

    def test_exported_script_reproduces_the_model(self, project, tmp_path):
        from easy_glm.workflow import Adjustment

        project.models["freq"].adjustments.append(Adjustment("Region", "R2", "R2", 0.9))
        project.exploration["leakage"]["ignored"] = ["Leak", "IDpol"]
        df = prepare(project)
        probe = run_model(project, df, "freq")
        kept = next(
            r
            for r in probe.rate_model.variables["DrivAge×Region"].table
            if r.exposure > 0
        )
        project.models["freq"].adjustments.append(
            Adjustment(
                "DrivAge×Region",
                kept.from_a,
                kept.to_a,
                1.25,
                from_b=kept.from_b,
                to_b=kept.to_b,
                cell=True,
            )
        )
        run = run_model(project, df, "freq")
        applied = next(
            r
            for r in run.rate_model.variables["DrivAge×Region"].table
            if r.key == kept.key
        )
        assert applied.relativity == 1.25
        src = to_script(project, "freq", run=run, output_prefix="freq_v1")
        assert "StepEncoder('DrivAge'" in src and "CategoricalEncoder('Region'" in src
        assert "alpha=0.002" in src and "monotone={'DrivAge': 'decreasing'}" in src
        assert "rm.update_relativity('Region', 'R2', 'R2', 0.9)" in src
        assert "spec.add_interaction(InteractionEncoder(" in src
        assert "from_b=" in src and "1.25" in src
        # A2: both stages are written out, and the RateModel is built from the pair
        assert "stage1 = fit_glm(\n    train,\n    spec.main_effects_spec()," in src
        assert "eta1 = stage1.linear_predictor(train)" in src
        assert "stage2 = fit_glm(\n    train,\n    spec.interactions_spec()," in src
        assert "offset=eta1" in src and "fit_intercept=False" in src
        assert "fit = TwoStageFit(stage1, stage2)" in src
        assert "to_rate_model(\n    fit," in src
        assert "excluded after the leakage review: Leak, IDpol" in src
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
        assert "holdout A/E" in proc.stdout
        rebuilt = RateModel.from_json(tmp_path / "freq_v1.easyglm")
        holdout = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            rebuilt.predict(holdout, exposure_col=None),
            run.predict(holdout),
            rtol=1e-10,
        )
        assert (tmp_path / "freq_v1_rate_tables.xlsx").exists()

    def test_exported_script_runs_when_no_cell_was_rated(self, project, tmp_path):
        """An interaction whose every cell is below the exposure floor has an
        encoder but no columns, so there is no second stage. The script must not
        emit one: a `fit_glm` on a zero-column design cannot run."""
        from easy_glm.core.fit import TwoStageFit

        project.models["freq"].interactions[0].min_cell_exposure = 0.99
        df = prepare(project)
        run = run_model(project, df, "freq")
        assert not isinstance(run.fit, TwoStageFit) and run.cells_kept == 0
        assert (run.tables["DrivAge×Region"]["relativity"] == 1.0).all()

        src = to_script(project, "freq", run=run, output_prefix="thin_v1")
        assert "TwoStageFit" not in src and "spec.interactions_spec()" not in src
        script = tmp_path / "thin.py"
        script.write_text(src)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        rebuilt = RateModel.from_json(tmp_path / "thin_v1.easyglm")
        holdout = df.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            rebuilt.predict(holdout, exposure_col=None),
            run.predict(holdout),
            rtol=1e-10,
        )

    def test_script_without_a_run_lets_the_data_decide_the_stages(
        self, project, tmp_path
    ):
        """Without a fit, whether any cell clears its floor is only known when
        the script runs, so the script calls `fit_two_stage`, which decides then
        — and each interaction keeps its own floor and penalty weight."""
        src = to_script(project, "freq", output_prefix="norun_v1")
        assert "fit = fit_two_stage(" in src
        assert "InteractionEncoder.from_data(" in src
        assert "min_cell_exposure=0.02" in src and "penalty_weight=1.0" in src
        script = tmp_path / "norun.py"
        script.write_text(src)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert (tmp_path / "norun_v1.easyglm").exists()


class TestGiniTies:
    @staticmethod
    def _tied_example():
        # two tie groups; scores identical within a group
        a = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0])
        e = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0])
        w = np.ones(7)
        return a, e, w

    @staticmethod
    def _unpooled(a, e, w):
        def curve(score):
            idx = np.argsort(-score, kind="stable")
            cw = np.concatenate([[0.0], np.cumsum(w[idx]) / w.sum()])
            ca = np.concatenate([[0.0], np.cumsum(a[idx]) / a.sum()])
            return 2 * np.trapezoid(ca, cw) - 1

        return curve(e / w) / curve(a / w)

    def test_pooled_value_is_the_order_free_expectation(self):
        import itertools

        a, e, w = self._tied_example()
        pooled = gini(a, e, w)
        values = [
            self._unpooled(a[list(p)], e[list(p)], w[list(p)])
            for p in itertools.permutations(range(7))
        ]
        assert pooled == pytest.approx(np.mean(values), abs=1e-12)
        assert min(values) < pooled < max(values)

    def test_row_order_does_not_matter(self):
        a, e, w = self._tied_example()
        rng = np.random.default_rng(0)
        seen = {gini(a[p], e[p], w[p]) for p in (rng.permutation(7) for _ in range(50))}
        assert len(seen) == 1

    def test_perfect_scaled_and_constant(self):
        rng = np.random.default_rng(1)
        w = rng.uniform(0.5, 1.5, 500)
        a = rng.poisson(0.2 * w).astype(float)
        assert gini(a, a, w) == pytest.approx(1.0)
        assert gini(a, 3 * a, w) == pytest.approx(1.0)
        constant = np.full(500, 0.2) * w
        assert gini(a, constant, w) == pytest.approx(0.0, abs=1e-12)
        assert gini(a, constant, w, normalize=False) == pytest.approx(0.0, abs=1e-12)
