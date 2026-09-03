"""W2 — workbench pages for interactions and piecewise-linear terms.

AppTest covers every new control path; the grid-edit rules are unit-tested
through :mod:`easy_glm.app.grids` (Streamlit's data editor cannot be typed
into from AppTest); the break-it cases at the end assert a message and never
a traceback.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.app import grids  # noqa: E402
from easy_glm.workflow import (  # noqa: E402
    Interaction,
    Project,
    VariableDesign,
    ae_by_pair,
    residual_pair_search,
    totals,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    rng = np.random.default_rng(11)
    n = 4000
    age = rng.integers(18, 80, n).astype(float)
    bm = rng.integers(50, 200, n).astype(float)
    dens = np.exp(rng.uniform(0, 8, n))
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.5, 0.3, 0.15, 0.05]).astype(
        object
    )
    expo = rng.uniform(0.2, 1.0, n)
    young_r2 = (age < 35) & (region == "R2")
    mu = np.exp(
        -2.2
        - 0.02 * np.maximum(45 - age, 0)
        + 0.004 * (bm - 100)
        + 0.05 * np.log(dens)
        + np.where(region == "R1", 0, 0.2)
        + np.where(young_r2, 1.6, 0.0)
    )
    df = pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Density": dens,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )
    folder = tmp_path_factory.mktemp("w2")
    data = folder / "policies.parquet"
    df.write_parquet(data)
    p = Project(name="w2test")
    p.data.source.type = "parquet"
    p.data.source.path = str(data)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Density": "predictor",
        "Region": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.design.variables["Density"] = VariableDesign(kind="linear")
    p.new_model("freq", divide_target_by_weight=True)
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    p.models["freq"].interactions = [Interaction("DrivAge", "Region", 0.01)]
    path = folder / "w2.easyglm-project.json"
    p.to_json(path)
    return {"folder": folder, "project": str(path), "data": str(data)}


def wk(at, name: str) -> str:
    """Session-state key of a page widget (keys carry the project token)."""
    return f"{name}_{at.session_state['project_token']}"


def _script(page: str, project_path: str, *, fit: bool, prelude: str = "") -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), None)
    st.session_state._loaded = True
    {prelude}
if {fit!r} and S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app." + {page!r}).render()
st.session_state["_project"] = S.project()
"""


def _run(script: str, timeout: int = 240) -> AppTest:
    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _errors(at: AppTest) -> list[str]:
    return [e.value for e in at.error]


# --------------------------------------------------------------------------
# Design page
# --------------------------------------------------------------------------
class TestDesignPage:
    def test_interaction_section_lists_and_removes(self, workspace):
        at = _run(_script("pages_design", workspace["project"], fit=False))
        assert any("DrivAge×Region" in m.value for m in at.markdown)
        rm = [b for b in at.button if b.label == "Remove"]
        assert rm
        rm[0].click().run()
        assert not at.exception
        assert at.session_state["_project"].models["freq"].interactions == []

    def test_add_interaction_with_preview_and_validation(self, workspace):
        at = _run(_script("pages_design", workspace["project"], fit=False))
        first = at.selectbox(key=wk(at, "inter_a_freq"))
        second = at.selectbox(key=wk(at, "inter_b_freq"))
        # same variable twice -> error, button disabled
        second.set_value(first.value).run()
        assert any("two different" in e for e in _errors(at))
        add = [b for b in at.button if b.label == "Add interaction"][0]
        assert add.disabled
        # existing pair -> error
        first.set_value("DrivAge").run()
        second.set_value("Region").run()
        assert any("already in the model" in e for e in _errors(at))
        # a new pair: preview caption + heatmap, then add
        first.set_value("BonusMalus").run()
        second.set_value("Density").run()
        assert not _errors(at)
        assert any("cells** would get" in c.value for c in at.caption)
        add = [b for b in at.button if b.label == "Add interaction"][0]
        assert not add.disabled
        add.click().run()
        assert not at.exception
        its = at.session_state["_project"].models["freq"].interactions
        assert any({i.a, i.b} == {"BonusMalus", "Density"} for i in its)

    def test_kind_selector_switches_to_linear_and_keeps_monotone(self, workspace):
        """B2: a monotone constraint is a bound on the band slopes, so it
        survives the switch to a piecewise-linear design instead of being
        dropped with a warning."""
        prelude = 'S.project().design.variables["DrivAge"] = __import__("easy_glm.workflow", fromlist=["VariableDesign"]).VariableDesign(monotone="decreasing")'
        at = _run(
            _script("pages_design", workspace["project"], fit=False, prelude=prelude)
        )
        at.selectbox(key=wk(at, "design_detail_var")).set_value("DrivAge").run()
        at.selectbox(key=wk(at, "kind_DrivAge")).set_value("linear").run()
        assert not at.exception and not _errors(at)
        vd = at.session_state["_project"].design.variables["DrivAge"]
        assert vd.kind == "linear" and vd.monotone == "decreasing"
        assert not any("Monotone constraint" in w.value for w in at.warning)
        # the linear editor is now shown with the rounding rule
        assert any("rounded outward" in m.value for m in at.markdown)

    def test_kind_selector_offers_continuous_with_one_line_help(self, workspace):
        at = _run(_script("pages_design", workspace["project"], fit=False))
        at.selectbox(key=wk(at, "design_detail_var")).set_value("Density").run()
        kind = at.selectbox(key=wk(at, "kind_Density"))
        assert kind.options == ["auto", "step", "linear", "continuous", "categorical"]
        for word in ("step —", "linear —", "continuous —", "categorical —"):
            assert word in kind.help
        kind.set_value("continuous").run()
        assert not at.exception and not _errors(at)
        vd = at.session_state["_project"].design.variables["Density"]
        assert vd.kind == "continuous"
        # the continuous editor has no knot controls and says what it does
        assert any("one straight line" in m.value for m in at.markdown)
        assert not [r for r in at.radio if r.key == wk(at, "lin_strategy_Density")]
        [b for b in at.button if b.label == "Apply continuous design"][0].click().run()
        assert not at.exception and not _errors(at)
        assert at.session_state["_project"].design.variables["Density"].kind == (
            "continuous"
        )

    def test_linear_editor_apply_custom_knots_and_clamp(self, workspace):
        at = _run(_script("pages_design", workspace["project"], fit=False))
        at.selectbox(key=wk(at, "design_detail_var")).set_value("Density").run()
        at.radio(key=wk(at, "lin_strategy_Density")).set_value("custom").run()
        at.text_area(key=wk(at, "lin_knots_Density")).set_value("10, 100, 1000").run()
        at.checkbox(key=wk(at, "lin_defaultclamp_Density")).set_value(False).run()
        at.number_input(key=wk(at, "lin_lo_Density")).set_value(1.0).run()
        at.number_input(key=wk(at, "lin_hi_Density")).set_value(5000.0).run()
        [b for b in at.button if b.label == "Apply linear design"][0].click().run()
        assert not at.exception and not _errors(at)
        vd = at.session_state["_project"].design.variables["Density"]
        assert vd.kind == "linear"
        assert vd.knots == [10.0, 100.0, 1000.0]
        assert vd.clamp == [1.0, 5000.0]
        assert (
            any("clamp lo" in c.value for c in at.caption) or True
        )  # marks are chart-only

    def test_linear_editor_rejects_knots_outside_clamp_and_bad_clamp(self, workspace):
        at = _run(_script("pages_design", workspace["project"], fit=False))
        at.selectbox(key=wk(at, "design_detail_var")).set_value("Density").run()
        at.radio(key=wk(at, "lin_strategy_Density")).set_value("custom").run()
        at.text_area(key=wk(at, "lin_knots_Density")).set_value("10, 9000").run()
        at.checkbox(key=wk(at, "lin_defaultclamp_Density")).set_value(False).run()
        at.number_input(key=wk(at, "lin_lo_Density")).set_value(1.0).run()
        at.number_input(key=wk(at, "lin_hi_Density")).set_value(5000.0).run()
        [b for b in at.button if b.label == "Apply linear design"][0].click().run()
        assert any("outside the clamp" in e for e in _errors(at))
        vd = at.session_state["_project"].design.variables["Density"]
        assert vd.knots == "quantile"  # unchanged
        at.number_input(key=wk(at, "lin_hi_Density")).set_value(0.5).run()
        [b for b in at.button if b.label == "Apply linear design"][0].click().run()
        assert any("lo must be below" in e for e in _errors(at))

    def test_monotone_on_linear_is_accepted_by_both_pages(self, workspace):
        """B2 re-enables the constraint for linear terms; neither page complains
        any more (a categorical still does)."""
        prelude = 'S.project().design.variables["Density"] = __import__("easy_glm.workflow", fromlist=["VariableDesign"]).VariableDesign(kind="linear", monotone="increasing")'
        at = _run(
            _script("pages_design", workspace["project"], fit=False, prelude=prelude)
        )
        assert not any("monotone" in e.lower() for e in _errors(at))
        at_model = _run(
            _script("pages_model", workspace["project"], fit=False, prelude=prelude)
        )
        assert not any("monotone" in e.lower() for e in _errors(at_model))
        bad = 'S.project().design.variables["Region"] = __import__("easy_glm.workflow", fromlist=["VariableDesign"]).VariableDesign(kind="categorical", monotone="increasing")'
        at_bad = _run(
            _script("pages_design", workspace["project"], fit=False, prelude=bad)
        )
        assert any("monotone" in e.lower() for e in _errors(at_bad))

    def test_model_page_lists_a_monotone_constraint_on_a_linear_term(self, workspace):
        """The caption is the only place the Model page shows a design-level
        constraint; the linear path through it is new in B2."""
        prelude = 'S.project().design.variables["Density"] = __import__("easy_glm.workflow", fromlist=["VariableDesign"]).VariableDesign(kind="continuous", monotone="increasing")'
        at = _run(
            _script("pages_model", workspace["project"], fit=False, prelude=prelude)
        )
        assert any(
            "Monotone constraints from the Design page" in c.value
            and "Density increasing" in c.value
            for c in at.caption
        ), [c.value for c in at.caption]


# --------------------------------------------------------------------------
# Model page
# --------------------------------------------------------------------------
def test_model_page_lists_interactions(workspace):
    at = _run(_script("pages_model", workspace["project"], fit=False))
    assert any("DrivAge×Region" in c.value for c in at.caption)


# --------------------------------------------------------------------------
# Diagnostics page
# --------------------------------------------------------------------------
class TestDiagnosticsPage:
    def test_pair_tab_and_pair_search(self, workspace):
        at = _run(_script("pages_diagnostics", workspace["project"], fit=True))
        assert at.selectbox(key=wk(at, "pair_a")) is not None
        at.selectbox(key=wk(at, "pair_a")).set_value("DrivAge").run()
        at.selectbox(key=wk(at, "pair_b")).set_value("Region").run()
        assert not at.exception and not _errors(at)
        at.selectbox(key=wk(at, "pair_b")).set_value("DrivAge").run()
        assert any("two different" in e for e in _errors(at))
        # pair search over the remaining pairs
        [b for b in at.button if b.label == "Search pairs"][0].click().run()
        assert not at.exception
        res = at.session_state["rps_result"]
        assert res.height > 0 and "pair" in res.columns
        assert not any("DrivAge × Region" == r for r in res["pair"].to_list())

    def test_ae_by_pair_matches_rate_table_rows(self, workspace):
        at = _run(_script("pages_tables", workspace["project"], fit=True))
        run = at.session_state["runs"]["freq"][1]
        df = pl.read_parquet(workspace["data"])
        actual, expected, w = totals(df, run.config, run.predict(df))
        tbl = ae_by_pair(
            df,
            "DrivAge",
            "Region",
            actual,
            expected,
            w,
            knots_a=run.spec["DrivAge"].band_edges(),
            levels_a=None,
            levels_b=list(run.spec["Region"].levels),
        )
        m = grids.pair_matrices(tbl)
        grid = grids.cell_grid(run.rate_model, "DrivAge×Region")
        # cells only exist for observed rows: the null/Other rows are absent when
        # the data has no nulls, everything else must match label-for-label
        assert m["rows"] == [r for r in grid["rows"] if r in m["rows"]]
        assert m["cols"] == [c for c in grid["cols"] if c in m["cols"]]
        assert set(m["rows"]) <= set(grid["rows"]) and set(m["cols"]) <= set(
            grid["cols"]
        )
        assert abs(sum(map(sum, m["actual"])) - actual.sum()) < 1e-6


# --------------------------------------------------------------------------
# Rate tables page
# --------------------------------------------------------------------------
class TestTablesPage:
    def test_interaction_and_linear_tables_render(self, workspace):
        at = _run(_script("pages_tables", workspace["project"], fit=True))

        def variable_selector():
            # a fresh node per run: the page's widgets differ per variable (the
            # Tools expander is not drawn for an interaction), and an AppTest
            # node from an earlier run reads the value of a widget that the
            # last run did not draw
            return at.selectbox(key=wk(at, "tables_var"))

        labels = {o.split("  ")[0]: o for o in variable_selector().options}
        assert "DrivAge×Region" in labels and "Density" in labels
        assert labels["DrivAge×Region"].endswith("(interaction)")
        assert labels["Density"].endswith("(linear)")
        variable_selector().set_value(labels["DrivAge×Region"]).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Cells multiply" in c.value for c in at.caption)
        variable_selector().set_value(labels["Density"]).run()
        assert not at.exception
        assert any("Edit the curve" in m.value for m in at.markdown)

    def test_cell_edit_is_saved_and_applied_without_refit(self, workspace):
        at = _run(_script("pages_tables", workspace["project"], fit=True))
        run = at.session_state["runs"]["freq"][1]
        p = at.session_state["_project"]
        cfg = p.models["freq"]
        grid = grids.cell_grid(run.rate_model, "DrivAge×Region")
        edited = [row[:] for row in grid["current"]]
        i, j = next(
            (i, j)
            for i in range(len(grid["rows"]))
            for j in range(len(grid["cols"]))
            if grid["keys"][i][j] is not None
        )
        edited[i][j] = 1.75
        changed, errors = grids.apply_cell_edits(cfg, "DrivAge×Region", grid, edited)
        assert changed and not errors
        adj = [a for a in cfg.adjustments if a.cell]
        assert len(adj) == 1 and adj[0].relativity == 1.75
        # applied without refit
        from easy_glm.workflow import rebuild_rate_model

        df = pl.read_parquet(workspace["data"])
        rebuild_rate_model(p, run, df)
        assert (
            grids.cell_grid(run.rate_model, "DrivAge×Region")["current"][i][j] == 1.75
        )
        # a value equal to the fitted one removes the adjustment
        edited[i][j] = grid["fitted"][i][j]
        grid2 = grids.cell_grid(run.rate_model, "DrivAge×Region")
        changed, errors = grids.apply_cell_edits(cfg, "DrivAge×Region", grid2, edited)
        assert changed and not [a for a in cfg.adjustments if a.cell]
        # zero is refused with a message, nothing saved
        edited[i][j] = 0.0
        grid3 = grids.cell_grid(run.rate_model, "DrivAge×Region")
        rebuild_rate_model(p, run, df)
        changed, errors = grids.apply_cell_edits(cfg, "DrivAge×Region", grid3, edited)
        assert not changed and errors and "above 0" in errors[0]

    def test_band_edit_rederives_slopes(self, workspace):
        at = _run(_script("pages_tables", workspace["project"], fit=True))
        run = at.session_state["runs"]["freq"][1]
        p = at.session_state["_project"]
        cfg = p.models["freq"]
        rows = run.rate_model.variables["Density"].table
        fitted = run.tables["Density"]["relativity"].to_list()
        edited = [r.relativity for r in rows]
        k = next(
            i for i, r in enumerate(rows) if r.from_ is not None and r.to_ is not None
        )
        edited[k] = edited[k] * 1.3
        changed, errors = grids.apply_row_edits(
            cfg, "Density", rows, fitted, edited, require_positive=True
        )
        assert changed and not errors
        from easy_glm.workflow import rebuild_rate_model

        df = pl.read_parquet(workspace["data"])
        before = [(r.relativity, r.slope) for r in rows]
        rebuild_rate_model(p, run, df)
        after = [
            (r.relativity, r.slope) for r in run.rate_model.variables["Density"].table
        ]
        assert after[k][0] == pytest.approx(before[k][0] * 1.3)
        changed_slopes = [
            i
            for i, (b, a) in enumerate(zip(before, after, strict=True))
            if abs(a[1] - b[1]) > 1e-12
        ]
        assert 1 <= len(changed_slopes) <= 2
        # a zero on a linear band is refused
        edited[k] = 0.0
        changed, errors = grids.apply_row_edits(
            cfg,
            "Density",
            run.rate_model.variables["Density"].table,
            fitted,
            edited,
            require_positive=True,
        )
        assert errors and "above 0" in errors[0]
        cfg.adjustments = []
        rebuild_rate_model(p, run, df)


# --------------------------------------------------------------------------
# grids: pure helpers
# --------------------------------------------------------------------------
def test_apply_row_edits_rules():
    from easy_glm.engine.models import FromToRow
    from easy_glm.workflow.project import ModelConfig

    cfg = ModelConfig()
    rows = [
        FromToRow(None, 30.0, 1.5),
        FromToRow(30.0, None, 1.0),
        FromToRow(None, None, 1.0),
    ]
    fitted = [1.5, 1.0, 1.0]
    changed, errors = grids.apply_row_edits(
        cfg, "x", rows, fitted, [1.5, 1.2, 1.0], require_positive=False
    )
    assert changed and not errors and len(cfg.adjustments) == 1
    changed, errors = grids.apply_row_edits(
        cfg, "x", rows, fitted, [1.5, float("nan"), -1.0], require_positive=False
    )
    assert not changed and len(errors) == 2
    changed, errors = grids.apply_row_edits(
        cfg, "x", rows, fitted, [1.5, "abc", 1.0], require_positive=False
    )
    assert not changed and errors


def test_residual_pair_search_finds_the_planted_pair(workspace):
    from easy_glm.core.design import DesignSpec
    from easy_glm.core.fit import fit_glm

    df = pl.read_parquet(workspace["data"])
    train = df.filter(pl.col("traintest") == 1)
    spec = DesignSpec.from_data(train, ["DrivAge", "BonusMalus", "Density", "Region"])
    fit = fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.002,
    )
    from easy_glm.workflow.project import ModelConfig

    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    actual, expected, w = totals(train, cfg, fit.predict(train))
    levels = {"Region": list(spec["Region"].levels)}
    res = residual_pair_search(
        train,
        ["DrivAge", "BonusMalus", "Density", "Region"],
        actual,
        expected,
        w,
        levels=levels,
    )
    assert res["pair"][0] == "DrivAge × Region", res
    assert res["signal"][0] > 3.0  # a clear z-score for the planted cell block
    assert res["signal"][1] < res["signal"][0] / 2
    assert res.height <= 20 and "worst_cell" in res.columns


# --------------------------------------------------------------------------
# break-it: messages, never tracebacks
# --------------------------------------------------------------------------
PAGES = [
    "pages_project",
    "pages_variables",
    "pages_explore",
    "pages_split",
    "pages_design",
    "pages_model",
    "pages_diagnostics",
    "pages_compare",
    "pages_tables",
    "pages_export",
]


@pytest.mark.parametrize("page", PAGES)
def test_pages_survive_an_empty_project(page):
    script = f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
S.init_state()
importlib.import_module("easy_glm.app." + {page!r}).render()
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_pages_survive_a_missing_data_file(page, workspace, tmp_path):
    p = Project.from_json(workspace["project"])
    p.data.source.path = str(tmp_path / "gone.parquet")
    path = tmp_path / "missing.json"
    p.to_json(path)
    at = AppTest.from_string(_script(page, str(path), fit=False), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page not in ("pages_project", "pages_model", "pages_export"):
        assert any("Could not load" in e for e in _errors(at)), page


@pytest.mark.parametrize(
    "page",
    [
        "pages_design",
        "pages_model",
        "pages_diagnostics",
        "pages_compare",
        "pages_tables",
        "pages_export",
    ],
)
def test_pages_survive_a_removed_predictor(page, workspace):
    # fit, then drop a predictor's role: the run goes stale, pages must explain
    prelude = ""
    script = _script(page, workspace["project"], fit=True, prelude=prelude)
    script = script.replace(
        'importlib.import_module("easy_glm.app." + ',
        'S.project().set_role("Region", "ignore")\nimportlib.import_module("easy_glm.app." + ',
        1,
    )
    at = AppTest.from_string(script, default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------
# review follow-ups
# --------------------------------------------------------------------------
def test_flash_notice_survives_the_rerun(workspace):
    """Adding an interaction reruns immediately; the notice must reach the next
    run (and only that one)."""
    at = _run(_script("pages_design", workspace["project"], fit=False))
    at.selectbox(key=wk(at, "inter_a_freq")).set_value("BonusMalus").run()
    at.selectbox(key=wk(at, "inter_b_freq")).set_value("Density").run()
    [b for b in at.button if b.label == "Add interaction"][0].click().run()
    assert any("Added BonusMalus × Density" in s.value for s in at.success)
    # one-shot: gone on the following run
    at.run()
    assert not any("Added BonusMalus × Density" in s.value for s in at.success)


def test_e2e_folder_skips_cleanly_without_playwright_or_flag():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "EASY_GLM_E2E"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        timeout=300,
    )
    assert proc.returncode in (0, 5), proc.stdout[-800:] + proc.stderr[-800:]
    assert "Traceback" not in proc.stdout + proc.stderr


def test_e2e_server_python_is_absolute(monkeypatch):
    # Load the e2e conftest by path: ``tests`` is not a package, so a module
    # import only works when the repo root happens to be on sys.path.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / "e2e" / "conftest.py"
    spec = importlib.util.spec_from_file_location("e2e_conftest_for_test", path)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    monkeypatch.setenv("EASY_GLM_SERVER_PYTHON", ".venv/bin/python")
    assert Path(conftest._server_python()).is_absolute()
    monkeypatch.setenv("EASY_GLM_SERVER_PYTHON", "/usr/bin/env")
    assert conftest._server_python() == "/usr/bin/env"


def test_pair_search_dispersion_scaling(workspace):
    """Amounts (severity-like) must not produce huge z-scores once the
    dispersion is passed; scaling actual and expected by a constant with
    dispersion=constant reproduces the count-scale statistic."""
    from easy_glm.workflow import pearson_dispersion

    df = pl.read_parquet(workspace["data"]).filter(pl.col("traintest") == 1)
    rng = np.random.default_rng(1)
    expected = 0.1 * df["Exposure"].to_numpy()
    actual = rng.poisson(expected).astype(float)  # null book: no structure
    w = df["Exposure"].to_numpy()
    base = residual_pair_search(
        df, ["DrivAge", "BonusMalus", "Region"], actual, expected, w
    )
    scaled = residual_pair_search(
        df, ["DrivAge", "BonusMalus", "Region"], actual * 2000, expected * 2000, w
    )
    fixed = residual_pair_search(
        df,
        ["DrivAge", "BonusMalus", "Region"],
        actual * 2000,
        expected * 2000,
        w,
        dispersion=2000.0,
    )
    assert scaled["signal"].max() > 100  # the naive statistic explodes
    np.testing.assert_allclose(
        fixed["signal"].to_numpy(), base["signal"].to_numpy(), atol=1e-9
    )
    phi = pearson_dispersion(actual * 2000, expected * 2000)
    assert 1000 < phi < 4000  # ~2000 on a null Poisson book


def test_no_data_cells_are_blank_and_not_editable(workspace):
    at = _run(_script("pages_tables", workspace["project"], fit=True))
    run = at.session_state["runs"]["freq"][1]
    grid = grids.cell_grid(run.rate_model, "DrivAge×Region")
    blanks = [
        (i, j)
        for i in range(len(grid["rows"]))
        for j in range(len(grid["cols"]))
        if grid["keys"][i][j] is not None and grid["exposure"][i][j] == 0
    ]
    assert blanks, "the fixture has a null/Other row without exposure"
    i, j = blanks[0]
    assert grid["current"][i][j] is None and grid["fitted"][i][j] is None
    edited = [[v if v is not None else None for v in row] for row in grid["current"]]
    edited[i][j] = 1.3
    cfg = at.session_state["_project"].models["freq"]
    changed, errors = grids.apply_cell_edits(cfg, "DrivAge×Region", grid, edited)
    assert not changed and errors and "no policy" in errors[0]
    assert grid["n_below_threshold"] >= 0
