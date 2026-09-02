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
    prepare,
    residual_factor_search,
    run_model,
    to_script,
    totals,
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
    p.design.variables["DrivAge"] = VariableDesign(monotone="decreasing")
    p.design.variables["Exp_Q"] = VariableDesign(knots="integer")
    p.new_model(
        "freq",
        family="poisson",
        divide_target_by_weight=True,
        predictors=["DrivAge", "BonusMalus", "Region", "VehGas", "Exp_Q"],
    )
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
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
        assert t["label"][-1] == "null"
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
        assert "null" in c["table"]["label"].to_list()
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
        s = run.summary()
        assert s["name"] == "freq" and s["non_zero"] <= s["features"]
        assert set(run.tables) == set(project.models["freq"].predictors)

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
        assert path.height == 4 and path["selected"].sum() == 1
        assert path["cv_deviance"].null_count() == 0

    def test_run_model_rejects_invalid(self, project):
        project.models["freq"].predictors = []
        with pytest.raises(ValueError, match="no predictors"):
            run_model(project, prepare(project), "freq")


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
class TestDiagnostics:
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
        assert tbl["label"][-1] == "null"
        cat = ae_by_variable(holdout, "Region", actual, expected, w)
        assert cat["exposure"].sum() == pytest.approx(w.sum())
        # plant a missing factor: inflate actuals for Lic == 'Q'
        planted = np.where(holdout["Lic"].to_numpy() == "Q", actual * 2.0, actual)
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
        assert path.height == 1 and path["alpha"][0] == pytest.approx(0.002)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
class TestExport:
    def test_script_without_run_mentions_from_data(self, project):
        src = to_script(project, "freq")
        assert "DesignSpec.from_data" in src and "fit_glm(" in src
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
        run = run_model(project, df, "freq")
        src = to_script(project, "freq", run=run, output_prefix="freq_v1")
        assert "StepEncoder('DrivAge'" in src and "CategoricalEncoder('Region'" in src
        assert "alpha=0.002" in src and "monotone={'DrivAge': 'decreasing'}" in src
        assert "rm.update_relativity('Region', 'R2', 'R2', 0.9)" in src
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
