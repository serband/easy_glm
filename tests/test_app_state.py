"""Workbench state: exploration sample vs full-data fits (D2) and persisted
fitted runs (D1). Pure hash helpers are tested directly; everything that
touches ``st.session_state`` runs inside Streamlit's AppTest."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.app import state as S  # noqa: E402
from easy_glm.workflow import Project  # noqa: E402

N_ROWS = 3000


def _frame(seed: int = 5, n: int = N_ROWS) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    bm = rng.integers(50, 200, n).astype(float)
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.5, 0.3, 0.15, 0.05]).astype(
        object
    )
    expo = rng.uniform(0.2, 1.0, n)
    mu = np.exp(
        -2.2
        - 0.02 * np.maximum(45 - age, 0)
        + 0.004 * (bm - 100)
        + np.where(region == "R1", 0, 0.2)
    )
    return pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


def _project(data_path: Path) -> Project:
    p = Project(name="wb")
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Region": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.new_model("freq", divide_target_by_weight=True)
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    return p


@pytest.fixture
def workspace(tmp_path) -> dict[str, Path]:
    data = tmp_path / "policies.parquet"
    _frame().write_parquet(data)
    project = tmp_path / "wb.easyglm-project.json"
    _project(data).to_json(project)
    return {"data": data, "project": project, "runs": tmp_path / "wb.easyglm-runs"}


def _run(script: str, timeout: int = 180) -> AppTest:
    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _script(project_path: Path, body: str, *, saved: bool = True) -> str:
    path_arg = repr(str(project_path)) if saved else "None"
    return f"""
import numpy as np
import polars as pl
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(project_path)!r}), {path_arg})
    st.session_state._loaded = True
p = S.project()
out = {{}}
{body}
st.session_state["out"] = out
"""


# --------------------------------------------------------------------------
# hashes (pure)
# --------------------------------------------------------------------------
class TestHashes:
    def test_model_and_data_hash_ignore_the_exploration_sample(self, workspace):
        p = Project.from_json(workspace["project"])
        before = (S.model_hash(p, "freq"), S.data_hash(p), S.source_hash(p))
        p.data.sample_rows = 100
        p.data.sample_seed = 7
        assert (S.model_hash(p, "freq"), S.data_hash(p), S.source_hash(p)) == before
        assert S.sample_hash(p) != S.sample_hash(
            Project.from_json(workspace["project"])
        )

    def test_model_hash_reacts_to_spec_but_not_to_adjustments(self, workspace):
        from easy_glm.workflow import Adjustment

        p = Project.from_json(workspace["project"])
        h0 = S.model_hash(p, "freq")
        p.models["freq"].adjustments.append(Adjustment("Region", "R2", "R2", 1.1))
        p.models["freq"].base_rate_override = 0.05
        assert S.model_hash(p, "freq") == h0
        p.models["freq"].penalty.alpha = 0.01
        assert S.model_hash(p, "freq") != h0

    def test_run_key_includes_data_identity_and_versions(self, workspace, monkeypatch):
        p = Project.from_json(workspace["project"])
        k0 = S.run_key(p, "freq")
        os.utime(workspace["data"], (1_000_000_000, 1_000_000_000))
        assert S.run_key(p, "freq") != k0
        k1 = S.run_key(p, "freq")
        monkeypatch.setattr(S, "_versions", lambda: {"glum": "0.0.0"})
        assert S.run_key(p, "freq") != k1


# --------------------------------------------------------------------------
# D2 — exploration sample vs full-data fits
# --------------------------------------------------------------------------
class TestSampleVsFull:
    def test_fit_uses_full_data_while_exploration_uses_the_sample(self, workspace):
        body = """
p.data.sample_rows = 50
S.touch()
full = S.prepared_frame()
sample = S.sample_frame()
train = S.train_frame()
run = S.fit_model("freq")
out["full"] = full.height
out["sample"] = sample.height
out["train"] = train.height
out["run_train_rows"] = run.train_rows
out["is_sampled"] = S.is_sampled()
out["raw_sample"] = S.raw_sample().height
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        n_train = int(_frame()["traintest"].sum())
        assert o["full"] == N_ROWS and o["sample"] == 50 and o["raw_sample"] == 50
        assert o["train"] == n_train and o["run_train_rows"] == n_train
        assert o["is_sampled"] is True

    def test_knots_come_from_the_full_training_rows(self, workspace):
        body = """
from easy_glm.core.design import DesignSpec
from easy_glm.workflow import VariableDesign, encoder_for
p.data.sample_rows = 40
S.touch()
train = S.train_frame()
enc = encoder_for("DrivAge", train["DrivAge"], VariableDesign(), p)
ref = DesignSpec.from_data(train, ["DrivAge"])["DrivAge"]
out["knots_equal"] = enc.knots == ref.knots
out["n_knots"] = len(enc.knots)
out["sample_rows"] = S.train_sample().height
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        assert o["knots_equal"] and o["n_knots"] > 5
        assert o["sample_rows"] <= 40

    def test_changing_the_sample_keeps_the_fit_valid(self, workspace):
        body = """
run = S.fit_model("freq")
p.data.sample_rows = 100
S.touch()
out["still_fitted"] = S.get_run("freq") is run
out["leakage_rows"] = S.leakage().height
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        assert o["still_fitted"] is True and o["leakage_rows"] >= 3

    def test_pages_render_with_a_sample(self, workspace):
        for page in (
            "pages_explore",
            "pages_design",
            "pages_variables",
            "pages_project",
        ):
            body = f"""
import importlib
p.data.sample_rows = 60
S.touch()
importlib.import_module("easy_glm.app.{page}").render()
"""
            at = _run(_script(workspace["project"], body))
            captions = " ".join(c.value for c in at.caption)
            if page in ("pages_explore", "pages_design"):
                assert "exploration sample" in captions


# --------------------------------------------------------------------------
# D1 — persisted runs
# --------------------------------------------------------------------------
class TestPersistedRuns:
    def _fit_and_persist(self, workspace, extra: str = "") -> AppTest:
        body = f"""
run = S.fit_model("freq")
hold = S.prepared_frame().filter(pl.col("traintest") == 0)
pl.DataFrame({{"pred": run.predict(hold)}}).write_parquet({str(workspace["project"].parent / "pred.parquet")!r})
out["persisted"] = [f.name for f in sorted(S.runs_dir().glob("*.pkl"))]
out["sidecar"] = [f.name for f in sorted(S.runs_dir().glob("*.json"))]
{extra}
"""
        return _run(_script(workspace["project"], body))

    def _reload(self, workspace, extra: str = "") -> dict:
        body = f"""
run = S.get_run("freq")
out["restored"] = run is not None
if run is not None:
    hold = S.prepared_frame().filter(pl.col("traintest") == 0)
    ref = pl.read_parquet({str(workspace["project"].parent / "pred.parquet")!r})["pred"].to_numpy()
    out["max_abs_diff"] = float(np.abs(run.predict(hold) - ref).max())
    out["in_session"] = "freq" in st.session_state.runs
out["files"] = [f.name for f in sorted(S.runs_dir().glob("*.pkl"))] if S.runs_dir() and S.runs_dir().exists() else []
{extra}
"""
        return _run(_script(workspace["project"], body)).session_state["out"]

    def test_run_is_restored_after_a_reload_with_identical_predictions(self, workspace):
        at = self._fit_and_persist(workspace)
        o = at.session_state["out"]
        assert len(o["persisted"]) == 1
        assert o["persisted"][0].startswith(S._model_tag("freq") + "-")
        assert (
            o["sidecar"]
            and json.loads((workspace["runs"] / o["sidecar"][0]).read_text())["model"]
            == "freq"
        )
        o2 = self._reload(workspace)
        assert o2["restored"] and o2["max_abs_diff"] == 0.0 and o2["in_session"]

    def test_changing_the_data_file_invalidates(self, workspace):
        self._fit_and_persist(workspace)
        _frame(seed=9).write_parquet(workspace["data"])  # new content, new size/mtime
        o = self._reload(workspace)
        # not restored; the old file stays until a newer run of the model is saved
        assert o["restored"] is False and len(o["files"]) == 1

    def test_touching_only_the_mtime_invalidates(self, workspace):
        self._fit_and_persist(workspace)
        os.utime(workspace["data"], (1_000_000_000, 1_000_000_000))
        o = self._reload(workspace)
        assert o["restored"] is False

    def test_corrupt_pickle_is_a_cache_miss_not_a_crash(self, workspace):
        at = self._fit_and_persist(workspace)
        pkl = workspace["runs"] / at.session_state["out"]["persisted"][0]
        pkl.write_bytes(b"not a pickle")
        o = self._reload(
            workspace,
            extra="""
run = S.fit_model("freq")
out["refit_ok"] = run is not None
out["files_after_refit"] = [f.name for f in sorted(S.runs_dir().glob("*.pkl"))]
""",
        )
        assert (
            o["restored"] is False
            and o["refit_ok"]
            and len(o["files_after_refit"]) == 1
        )

    def test_spec_change_invalidates_and_a_new_fit_replaces_the_file(self, workspace):
        self._fit_and_persist(workspace)
        p = Project.from_json(workspace["project"])
        p.models["freq"].penalty.alpha = 0.01
        p.to_json(workspace["project"])
        o = self._reload(
            workspace,
            extra="""
S.fit_model("freq")
out["files_after_refit"] = [f.name for f in sorted(S.runs_dir().glob("*.pkl"))]
""",
        )
        assert o["restored"] is False and len(o["files"]) == 1  # kept until replaced
        assert len(o["files_after_refit"]) == 1 and o["files_after_refit"] != o["files"]

    def test_version_change_invalidates(self, workspace):
        self._fit_and_persist(workspace)
        o = self._reload(
            workspace,
            extra="",
        )
        assert o["restored"]
        body = """
S._versions = lambda: {"python": "9.9", "easy_glm": "x", "glum": "x", "polars": "x", "numpy": "x"}
out["restored"] = S.get_run("freq") is not None
"""
        at = _run(_script(workspace["project"], body))
        assert at.session_state["out"]["restored"] is False

    def test_adjustment_made_after_persisting_is_applied_on_reload(self, workspace):
        self._fit_and_persist(workspace)
        p = Project.from_json(workspace["project"])
        from easy_glm.workflow import Adjustment

        p.models["freq"].adjustments.append(Adjustment("Region", "R2", "R2", 2.0))
        p.to_json(workspace["project"])
        o = self._reload(
            workspace,
            extra="""
rm = S.get_run("freq").rate_model
rows = {str(r.from_): r.relativity for r in rm.variables["Region"].table}
out["r2"] = rows["R2"]
out["adjustments_on_run"] = len(S.get_run("freq").config.adjustments)
""",
        )
        assert o["restored"] and o["max_abs_diff"] > 0
        assert o["r2"] == pytest.approx(2.0) and o["adjustments_on_run"] == 1

    def test_two_models_persist_independently_latest_only(self, workspace):
        body = """
p.new_model("freq-2", divide_target_by_weight=True, predictors=["DrivAge", "Region"])
p.models["freq-2"].penalty.alpha = 0.003
p.models["freq-2"].penalty.cv = None
S.touch()
S.fit_model("freq")
S.fit_model("freq-2")
first = sorted(f.name for f in S.runs_dir().glob("*.pkl"))
p.models["freq"].penalty.alpha = 0.004
S.touch()
S.fit_model("freq")
second = sorted(f.name for f in S.runs_dir().glob("*.pkl"))
st.session_state.runs = {}
out["first"] = first
out["second"] = second
out["both"] = S.get_run("freq") is not None and S.get_run("freq-2") is not None
out["tag_freq"] = S._model_tag("freq")
out["tag_freq2"] = S._model_tag("freq-2")
out["tmp_left"] = [f.name for f in S.runs_dir().glob("*.tmp")]
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        t1, t2 = o["tag_freq"], o["tag_freq2"]
        assert len(o["first"]) == 2 and len(o["second"]) == 2
        # the prefix-sharing model keeps its file across the other model's refit
        assert [f for f in o["first"] if f.startswith(t2)] == [
            f for f in o["second"] if f.startswith(t2)
        ]
        assert [f for f in o["first"] if f.startswith(t1)] != [
            f for f in o["second"] if f.startswith(t1)
        ]
        assert o["both"] and o["tmp_left"] == []

    def test_unsaved_project_persists_nothing(self, workspace):
        body = """
run = S.fit_model("freq")
out["runs_dir"] = S.runs_dir()
out["note"] = S.persistence_note()
out["folder_exists"] = (p.data.source.path and (__import__("pathlib").Path(p.data.source.path).parent / "wb.easyglm-runs").exists())
"""
        at = _run(_script(workspace["project"], body, saved=False))
        o = at.session_state["out"]
        assert o["runs_dir"] is None and "not persisted" in o["note"]
        assert not o["folder_exists"]

    def test_refresh_adjustments_persists_the_adjusted_run(self, workspace):
        self._fit_and_persist(workspace)
        body = """
from easy_glm.workflow import Adjustment
S.get_run("freq")
p.models["freq"].adjustments.append(Adjustment("Region", "R3", "R3", 0.5))
S.touch()
S.refresh_adjustments("freq")
"""
        _run(_script(workspace["project"], body))
        o = self._reload(
            workspace,
            extra="""
rm = S.get_run("freq").rate_model
out["r3"] = {str(r.from_): r.relativity for r in rm.variables["Region"].table}["R3"]
""",
        )
        assert o["restored"] and o["r3"] == pytest.approx(0.5)


class TestReviewFollowUps:
    def test_derived_preview_works_with_and_without_a_sample(self, workspace):
        for sample in (0, 60):
            body = f"""
from easy_glm.app import pages_variables
p.data.sample_rows = {sample} or None
S.touch()
pages_variables.render()
"""
            script = _script(workspace["project"], body)
            at = AppTest.from_string(script, default_timeout=180)
            at.run()
            assert not at.exception
            at.text_input(key="derived_name").set_value("AgeSq")
            at.text_input(key="derived_expr").set_value("pl.col('DrivAge') ** 2")
            at.button(key="derived_preview").click().run()
            assert not at.exception, [e.value for e in at.exception]
            assert not at.error, [e.value for e in at.error]
            assert at.dataframe or at.markdown  # a preview appeared

    def test_sample_widget_does_not_leak_into_another_project(
        self, workspace, tmp_path
    ):
        other = tmp_path / "other.easyglm-project.json"
        q = Project.from_json(workspace["project"])
        q.name = "other"
        q.to_json(other)
        body = f"""
from easy_glm.app import pages_project
if st.session_state.get("phase", 0) == 0:
    p.data.sample_rows = 60
    S.touch()
    pages_project.render()
else:
    if st.session_state.phase == 1:
        S.set_project(Project.from_json({str(other)!r}), {str(other)!r})
    pages_project.render()
    out["other_sample"] = S.project().data.sample_rows
    out["on_disk"] = Project.from_json({str(other)!r}).data.sample_rows
    out["widget"] = st.session_state.get("sample_rows")
"""
        script = _script(workspace["project"], body)
        at = AppTest.from_string(script, default_timeout=180)
        at.run()
        assert not at.exception
        assert at.session_state["sample_rows"] == 60
        for phase in (1, 2):  # open the other project, then one more rerun
            at.session_state["phase"] = phase
            at.run()
            assert not at.exception, [e.value for e in at.exception]
            o = at.session_state["out"]
            assert o["other_sample"] is None and o["on_disk"] is None
            assert o["widget"] in (None, 0)

    def test_transient_spec_edit_does_not_erase_the_persisted_fit(self, workspace):
        body = """
run = S.fit_model("freq")
files = [f.name for f in S.runs_dir().glob("*.pkl")]
p.models["freq"].penalty.alpha = 0.01   # look at a different alpha ...
S.touch()
st.session_state.runs = {}
out["while_edited"] = S.get_run("freq") is not None
out["files_kept"] = [f.name for f in S.runs_dir().glob("*.pkl")] == files
p.models["freq"].penalty.alpha = 0.002  # ... and change your mind
S.touch()
out["after_revert"] = S.get_run("freq") is not None
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        assert o["while_edited"] is False and o["files_kept"] and o["after_revert"]

    def test_refused_adjustment_is_dropped_on_load_not_refitted(self, workspace):
        body = """
run = S.fit_model("freq")
"""
        _run(_script(workspace["project"], body))
        p = Project.from_json(workspace["project"])
        from easy_glm.workflow import Adjustment

        p.models["freq"].adjustments.append(Adjustment("Region", "R2", "R2", 1.3))
        p.models["freq"].adjustments.append(Adjustment("Region", "NOPE", "NOPE", 1.1))
        p.to_json(workspace["project"])
        body = """
run = S.get_run("freq")
out["restored"] = run is not None
out["adjustments"] = [a.from_ for a in p.models["freq"].adjustments]
out["r2"] = {str(r.from_): r.relativity for r in run.rate_model.variables["Region"].table}["R2"]
out["files"] = len(list(S.runs_dir().glob("*.pkl")))
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        assert (
            o["restored"]
            and o["adjustments"] == ["R2"]
            and o["r2"] == pytest.approx(1.3)
        )
        assert o["files"] == 1 and at.error  # the refused entry was reported

    def test_orphaned_model_files_are_removed_on_save(self, workspace):
        body = """
p.new_model("temp", divide_target_by_weight=True, predictors=["DrivAge"])
p.models["temp"].penalty.alpha = 0.003
p.models["temp"].penalty.cv = None
S.touch()
S.fit_model("temp")
S.fit_model("freq")
before = len(list(S.runs_dir().glob("*.pkl")))
p.models.pop("temp")
S.touch()
S.refresh_adjustments("freq")  # any save of a live model tidies the folder
out["before"] = before
out["after"] = len(list(S.runs_dir().glob("*.pkl")))
"""
        at = _run(_script(workspace["project"], body))
        o = at.session_state["out"]
        assert o["before"] == 2 and o["after"] == 1

    def test_persist_format_is_part_of_the_key(self, workspace, monkeypatch):
        p = Project.from_json(workspace["project"])
        k0 = S.run_key(p, "freq")
        monkeypatch.setattr(S, "PERSIST_FORMAT", 99)
        assert S.run_key(p, "freq") != k0


def test_easyglm_summary_exposes_offset(workspace):
    from easy_glm import EasyGLM

    df = _frame()
    df = df.with_columns((pl.col("Exposure") * 100).log().alias("logprem"))
    p = _project(workspace["data"])
    eglm = EasyGLM.fit(
        df,
        target="ClaimNb",
        model_type="Poisson",
        predictors=["DrivAge", "Region"],
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.01,
    )
    assert eglm.summary()["offset_col"] is None
    assert p.models["freq"].target == "ClaimNb"
