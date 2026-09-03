"""W3 — workbench hardening: one test per breaker finding (docs/reviews/w2-breakage.md).

Each blocking finding (1–13) reproduces the original breakage on the pre-W3
tree (a traceback or silently wrong project state) and passes only with the
fix. Pure rules (grid application, recode mapping, model names, column
checks) are unit-tested; page behaviour runs under Streamlit's AppTest.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.app import pages_variables as pv  # noqa: E402
from easy_glm.app import ui  # noqa: E402
from easy_glm.app.pages_project import open_project_file  # noqa: E402
from easy_glm.engine import RateModel  # noqa: E402
from easy_glm.engine.models import level_label  # noqa: E402
from easy_glm.workflow import (
    Project,
    Split,
    VariableDesign,
    add_split_column,
)  # noqa: E402
from easy_glm.workflow.project import validate_model_name  # noqa: E402
from easy_glm.workflow.run import other_label_for  # noqa: E402

N = 3000


def wk(at, name: str) -> str:
    """Session-state key of a page widget (keys carry the project token)."""
    return f"{name}_{at.session_state['project_token']}"


def _frame(seed: int = 5, n: int = N) -> pl.DataFrame:
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
            "IDpol": [f"P{i:06d}" for i in range(n)],  # text ids (finding 10)
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


def _project(data_path: Path) -> Project:
    p = Project(name="w3")
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
    project = tmp_path / "w3.easyglm-project.json"
    _project(data).to_json(project)
    return {"data": data, "project": project, "dir": tmp_path}


def _script(
    page: str, project_path: str, *, fit: bool = False, prelude: str = ""
) -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), {project_path!r})
    st.session_state._loaded = True
    {prelude}
if {fit!r} and not st.session_state.get("_fitted"):
    S.fit_model("freq")
    st.session_state._fitted = True
importlib.import_module("easy_glm.app." + {page!r}).render()
st.session_state["_project"] = S.project()
st.session_state["_path"] = st.session_state.get("project_path")
"""


def _run(script: str, timeout: int = 180) -> AppTest:
    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _errors(at: AppTest) -> list[str]:
    return [e.value for e in at.error]


def _texts(at: AppTest) -> str:
    parts = [w.value for w in at.warning] + [e.value for e in at.error]
    parts += [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# data loss
# --------------------------------------------------------------------------
def test_breakage_01_new_project_never_overwrites_the_open_file(workspace):
    before = workspace["project"].read_text()
    at = _run(_script("pages_project", str(workspace["project"])))
    btn = at.button(key=wk(at, "new_project_btn"))
    btn.click().run()  # first click: confirmation
    assert any("Click the button again" in w.value for w in at.warning)
    assert at.session_state["_path"] == str(workspace["project"])
    at.button(key=wk(at, "new_project_btn")).click().run()  # second click: new project
    assert not at.exception
    assert at.session_state["_path"] is None
    assert at.session_state["_project"].name == "untitled"
    # an edit on the new project must not autosave anywhere
    at.run()
    assert workspace["project"].read_text() == before


def test_breakage_02_two_tabs_do_not_overwrite_each_other(workspace):
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path))
    tab_b = _run(_script("pages_model", path))
    # tab B edits (notes) -> autosaved
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("note from B").run()
    assert json.loads(workspace["project"].read_text())["models"]["freq"]["notes"] == (
        "note from B"
    )
    # tab A edits -> must NOT overwrite B's version; a notice appears instead
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("note from A").run()
    assert not tab_a.exception
    on_disk = json.loads(workspace["project"].read_text())["models"]["freq"]["notes"]
    assert on_disk == "note from B"
    assert any("changed by another browser tab" in w.value for w in tab_a.warning)
    # reload from disk -> tab A now has B's version
    tab_a.button(key="conflict_reload").click().run()
    assert tab_a.session_state["_project"].models["freq"].notes == "note from B"
    assert not any("changed by another" in w.value for w in tab_a.warning)
    # the other way round: overwrite with this tab's version
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("B again").run()
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    tab_a.button(key="conflict_overwrite").click().run()
    assert json.loads(workspace["project"].read_text())["models"]["freq"]["notes"] == (
        "A wins"
    )


def test_breakage_03_model_referencing_a_missing_column_is_refused(workspace):
    # a fitted model whose target column is no longer in the data
    prelude = (
        'S.fit_model("freq"); S.project().models["freq"].target = "claims"; S.touch()'
    )
    at = _run(_script("pages_model", str(workspace["project"]), prelude=prelude), 240)
    texts = _texts(at)
    assert "claims" in texts and "not" in texts
    assert at.button(key=wk(at, "fit_freq")).disabled
    assert not any("Fitted and up to date" in s.value for s in at.success)
    # the persisted / cached run is not offered anywhere
    assert at.session_state["_project"].models["freq"].target == "claims"


def test_breakage_03_rename_carries_roles_and_model_references(workspace):
    p = Project.from_json(workspace["project"])
    p.models["freq"].monotone["DrivAge"] = "decreasing"
    touched = p.rename_column("ClaimNb", "claims")
    assert touched == ["freq"]
    assert p.data.roles["claims"] == "target" and "ClaimNb" not in p.data.roles
    assert p.models["freq"].target == "claims"
    assert p.rename_column("DrivAge", "age") == ["freq"]
    assert p.models["freq"].predictors == ["age", "BonusMalus", "Region"]
    assert p.models["freq"].monotone == {"age": "decreasing"}
    assert p.validate() == []
    assert p.missing_columns(["claims", "Exposure", "age", "BonusMalus"]) == [
        "split column 'traintest' is not in the data",
        "freq: predictor(s) not in the data: ['Region']",
    ]


def test_breakage_04_random_split_name_cannot_shadow_a_data_column(workspace):
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "random"
    p.data.split.column = "traintest"
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    at.text_input(key=wk(at, "split_name")).set_value("ClaimNb").run()
    assert not at.exception
    assert any("overwrite" in e for e in _errors(at))
    assert at.session_state["_project"].data.split.column == "traintest"
    at.text_input(key=wk(at, "split_name")).set_value("   ").run()
    assert any("needs a name" in e for e in _errors(at))
    assert at.session_state["_project"].data.split.column == "traintest"
    # the engine refuses too, so a hand-edited project cannot slip through
    with pytest.raises(ValueError, match="overwrite"):
        add_split_column(_frame(), Split(mode="random", column="Exposure"))


# --------------------------------------------------------------------------
# crashes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("page", ["pages_variables", "pages_split", "pages_explore"])
def test_breakage_05_broken_pipeline_is_a_message_on_every_page(workspace, page):
    p = Project.from_json(workspace["project"])
    p.data.filters = ["pl.col('Gone') > 0"]  # references a column that does not exist
    p.to_json(workspace["project"])
    at = _run(_script(page, str(workspace["project"])))
    assert any("Gone" in e or "data steps" in e for e in _errors(at)), page


def test_breakage_05_roles_for_missing_columns_are_listed_not_repointed(workspace):
    weird = workspace["dir"] / "weird.parquet"
    _frame().rename({"DrivAge": "1st_age", "Region": "région/zone"}).write_parquet(
        weird
    )
    p = Project.from_json(workspace["project"])
    p.data.source.path = str(weird)
    p.to_json(workspace["project"])
    at = _run(_script("pages_variables", str(workspace["project"])))
    warn = "\n".join(w.value for w in at.warning)
    assert "DrivAge" in warn and "région/zone" not in warn, warn
    kept = at.session_state["_project"]
    assert kept.data.roles["DrivAge"] == "predictor"  # kept, not re-pointed
    assert kept.models["freq"].predictors == ["DrivAge", "BonusMalus", "Region"]
    at = _run(_script("pages_model", str(workspace["project"])))
    assert any("not in the data" in e for e in _errors(at))


def test_breakage_06_rename_onto_an_existing_name_is_refused(workspace):
    p = Project.from_json(workspace["project"])
    rows = [
        {
            "column": c,
            "rename to": "",
            "role": p.data.roles.get(c, "unassigned"),
            "type": "auto",
        }
        for c in _frame().columns
    ]
    rows[4]["rename to"] = "DrivAge"  # BonusMalus -> DrivAge (exists)
    changed, notices = pv.apply_roles_grid(p, _frame().columns, rows)
    assert not changed
    assert any(k == "error" and "already has that name" in t for k, t in notices)
    assert p.data.renames == {} and p.data.roles["BonusMalus"] == "predictor"


def test_breakage_07_cleared_rename_cell_restores_the_original(workspace):
    p = Project.from_json(workspace["project"])
    p.data.renames = {"BonusMalus": "bm"}
    p.rename_column("BonusMalus", "bm")
    assert p.data.roles["bm"] == "predictor" and p.models["freq"].predictors[1] == "bm"
    cols = _frame().columns
    rows = [
        {
            "column": c,
            "rename to": "",
            "role": p.data.roles.get(p.data.renames.get(c, c), "unassigned"),
            "type": "auto",
        }
        for c in cols
    ]
    rows[4]["rename to"] = float("nan")  # the editor returns NaN for a cleared cell
    changed, notices = pv.apply_roles_grid(p, cols, rows)
    assert changed and not any(k == "error" for k, _ in notices)
    assert p.data.renames == {}
    assert p.data.roles["BonusMalus"] == "predictor" and "bm" not in p.data.roles
    assert p.models["freq"].predictors[1] == "BonusMalus"


def test_breakage_08_09_derived_columns_that_cannot_run_are_refused(workspace):
    at = _run(_script("pages_variables", str(workspace["project"])))
    at.text_input(key=wk(at, "derived_name")).set_value("foo").run()
    at.text_input(key=wk(at, "derived_expr")).set_value("pl.col('foo') + 1").run()
    at.button(key=wk(at, "derived_add")).click().run()
    assert not at.exception
    assert any("foo" in e for e in _errors(at))
    assert at.session_state["_project"].data.derived == []
    at.text_input(key=wk(at, "derived_name")).set_value("bad").run()
    at.text_input(key=wk(at, "derived_expr")).set_value("pl.col('Region') / 2").run()
    at.button(key=wk(at, "derived_add")).click().run()
    assert not at.exception
    assert any("fails" in e for e in _errors(at))
    assert at.session_state["_project"].data.derived == []
    # a good one is added, and gets the predictor role
    at.text_input(key=wk(at, "derived_name")).set_value("age2").run()
    at.text_input(key=wk(at, "derived_expr")).set_value("pl.col('DrivAge') * 2").run()
    at.button(key=wk(at, "derived_add")).click().run()
    assert not at.exception
    p = at.session_state["_project"]
    assert [d.name for d in p.data.derived] == ["age2"]
    assert p.data.roles["age2"] == "predictor"


def test_breakage_10_column_split_mode_never_auto_picks_or_demotes(workspace):
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "column"
    p.data.split.column = "gone"  # the indicator is not in the data
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    assert any("not in the data" in e for e in _errors(at))
    assert at.session_state["_project"].data.roles["IDpol"] == "id"  # untouched
    # a text indicator compared with a number: a message, never a traceback
    at.selectbox(key=wk(at, "split_col")).set_value("Region").run()
    assert not at.exception
    assert at.session_state["_project"].data.split.column == "Region"
    assert at.session_state["_project"].data.roles["IDpol"] == "id"
    assert any("No row" in e or "TRAIN" in e for e in _errors(at))
    # string indicators compare as text in the engine
    out = add_split_column(
        _frame(), Split(mode="column", column="Region", train_value="R1")
    )
    assert out["Region"].sum() == (_frame()["Region"] == "R1").sum()


def test_breakage_11_bad_project_files_are_messages(workspace, tmp_path):
    bad = {
        "parquet.json": workspace["data"].read_bytes(),
        "truncated.json": b'{"name": "x", "data": {',
        "v99.json": json.dumps({"version": 99}).encode(),
        "list.json": b"[1, 2, 3]",
        "badtypes.json": json.dumps({"version": 2, "data": {"roles": "oops"}}).encode(),
    }
    for name, content in bad.items():
        (tmp_path / name).write_bytes(content)
    prelude = (
        "import json\nst.session_state['_errs'] = {}\n"
        "from easy_glm.app.pages_project import open_project_file\n"
        + "\n".join(
            f"st.session_state['_errs'][{name!r}] = open_project_file({str(tmp_path / name)!r})"
            for name in bad
        )
    )
    at = _run(
        _script(
            "pages_project",
            str(workspace["project"]),
            prelude=prelude.replace("\n", "\n    "),
        )
    )
    errs = at.session_state["_errs"]
    assert all(
        errs[name] and "Not a valid easy_glm project" in errs[name] for name in bad
    ), errs
    assert at.session_state["_project"].name == "w3"  # untouched
    assert open_project_file(str(tmp_path / "missing.json")).endswith("does not exist")
    assert "folder" in open_project_file(str(tmp_path))


def test_breakage_12_saving_to_a_bad_path_is_a_message(workspace, tmp_path):
    at = _run(_script("pages_project", str(workspace["project"])))
    key = [k for k in at.session_state.filtered_state if k.startswith("proj_path_")][0]
    at.text_input(key=key).set_value("/nonexistent_dir_easy_glm/x.json").run()
    # the "Save project" button has no key: find it by label
    save = [b for b in at.button if b.label == "Save project"][0]
    save.click().run()
    assert not at.exception
    assert any("Could not save" in e for e in _errors(at))
    # autosave to an unwritable path never raises and is reported on every page
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_file = ro_dir / "p.easyglm-project.json"
    ro_file.write_text("{}")
    os.chmod(ro_file, stat.S_IREAD)
    try:
        prelude = f"st.session_state.project_path = {str(ro_file)!r}; st.session_state.project_stamp = None"
        at2 = _run(_script("pages_model", str(workspace["project"]), prelude=prelude))
        at2.text_input(key=wk(at2, "notes_freq")).set_value("x").run()
        assert not at2.exception
        assert any("Autosave failed" in e for e in _errors(at2))
    finally:
        os.chmod(ro_file, stat.S_IREAD | stat.S_IWRITE)


def test_breakage_13_model_names_are_file_safe(workspace):
    p = Project.from_json(workspace["project"])
    for bad, why in [
        ("a/b", "cannot contain"),
        ("", "empty"),
        ("  ", "empty"),
        ("..", "'.'"),
        ("x" * 61, "longer"),
        ("freq", "already"),
    ]:
        assert why in validate_model_name(bad, p.models), bad
    with pytest.raises(ValueError, match="cannot contain"):
        p.new_model("a/b")
    assert validate_model_name("  freq_v2  ", p.models) is None
    p.new_model("  freq_v2  ")
    assert "freq_v2" in p.models
    # a legacy project that already holds a bad name still renders and downloads
    p.models["a/b"] = p.models.pop("freq_v2")
    p.models["a/b"].penalty.alpha = 0.002
    p.models["a/b"].penalty.cv = None
    p.champion = "a/b"
    p.to_json(workspace["project"])
    for page in ("pages_tables", "pages_export"):
        at = _run(
            _script(page, str(workspace["project"]), prelude='S.fit_model("a/b")'),
            240,
        )
        assert not at.exception, page
    assert ui.safe_filename("a/b") == "a_b"
    assert ui.safe_filename("///") == "model"
    assert ui.safe_filename("freq v2") == "freq v2"


# --------------------------------------------------------------------------
# misleading output
# --------------------------------------------------------------------------
def test_misleading_14_target_and_weight_must_be_numeric(workspace):
    prelude = 'S.project().models["freq"].weight = "IDpol"; S.touch()'
    at = _run(_script("pages_model", str(workspace["project"]), prelude=prelude))
    assert any("IDpol" in e and "numeric" in e for e in _errors(at))
    assert (
        at.session_state["_project"].models["freq"].weight == "IDpol"
    )  # kept, not re-pointed


def test_misleading_18_cleared_recode_cell_is_no_mapping():
    rows = [
        {"level": "Regular", "rows": 10, "map to": float("nan")},
        {"level": "Diesel", "rows": 5, "map to": "  "},
        {"level": "Elec", "rows": 1, "map to": "Other"},
        {"level": None, "rows": 1, "map to": "x"},
    ]
    assert pv.recode_mapping(rows) == {"Elec": "Other"}


def test_misleading_19_threshold_message_is_not_all_null(workspace):
    from easy_glm.workflow import build_design

    p = Project.from_json(workspace["project"])
    p.design.defaults.min_level_share = 0.6
    with pytest.raises(ValueError, match="reaches the minimum level share"):
        build_design(p, _frame(), ["Region"])


def test_misleading_22_a_real_other_level_gets_a_distinct_lumped_label():
    assert other_label_for(["A", "B"]) == "Other"
    assert other_label_for(["A", "Other"]) == "Other (lumped)"


def test_misleading_24_non_finite_knots_are_refused():
    from easy_glm.app.pages_design import _parse_numbers

    for text in ("nan", "inf, 40", "1e400, 30", "-1e309"):
        with pytest.raises(ValueError, match="finite|number"):
            _parse_numbers(text)
    assert _parse_numbers("30, 40, 30") == [30.0, 40.0]


def test_misleading_26_huge_percentages_are_dashes():
    assert ui.fmt(-919206693535870.2, pct=True) == "—"
    assert ui.fmt(0.047, pct=True) == "4.7%"


def test_misleading_27_autosave_errors_show_on_every_page(workspace):
    prelude = "st.session_state.errors = ['Autosave failed: Permission denied']"
    for page in ("pages_model", "pages_design", "pages_tables"):
        at = _run(_script(page, str(workspace["project"]), prelude=prelude))
        assert any("Autosave failed" in e for e in _errors(at)), page


def test_misleading_29_target_weight_and_alpha_rules(workspace):
    p = Project.from_json(workspace["project"])
    p.models["freq"].weight = "ClaimNb"
    assert any("same column" in x for x in p.validate("freq"))
    p = Project.from_json(workspace["project"])
    p.models["freq"].penalty.alpha = 0.0
    assert any("alpha must be > 0" in x for x in p.validate("freq"))
    p = Project.from_json(workspace["project"])
    p.models["freq"].predictors.append("ClaimNb")
    assert any("cannot also be a predictor" in x for x in p.validate("freq"))


def test_misleading_31_divide_box_is_unticked_without_a_weight(workspace):
    prelude = 'S.project().models["freq"].weight = None; S.touch()'
    at = _run(_script("pages_model", str(workspace["project"]), prelude=prelude))
    box = at.checkbox(key=wk(at, "div_freq"))
    assert box.value is False and box.disabled


def test_misleading_36_deleting_a_model_removes_its_persisted_fit(workspace):
    at = _run(_script("pages_model", str(workspace["project"]), fit=True), 240)
    runs = workspace["dir"] / "w3.easyglm-runs"
    assert list(runs.glob("*.pkl"))
    delete = [b for b in at.button if b.label == "Delete"][0]
    delete.click().run()
    assert not at.exception
    assert not list(runs.glob("*.pkl"))
    assert "freq" not in at.session_state["_project"].models


def test_misleading_37_monotone_on_a_categorical_is_a_validation_problem(workspace):
    p = Project.from_json(workspace["project"])
    p.design.variables["Region"] = VariableDesign(
        kind="categorical", monotone="increasing"
    )
    assert any(
        "monotone constraints apply to numeric designs only" in x for x in p.validate()
    )


def test_cell_text_rules():
    assert pv._cell_text(None) == "" and pv._cell_text(float("nan")) == ""
    assert pv._cell_text("  x ") == "x" and math.isnan(float("nan"))


# --------------------------------------------------------------------------
# W3 review follow-ups (docs/reviews/w3-hardening.md, S1–S5 and item 32)
# --------------------------------------------------------------------------
def test_s1_autosave_error_clears_once_saving_works_again(workspace):
    """S1: the red banner must go when autosave recovers, or the tool lies."""
    path = workspace["project"]
    at = _run(_script("pages_model", str(path)))
    os.chmod(path, stat.S_IREAD)
    try:
        at.text_input(key=wk(at, "notes_freq")).set_value("while read-only").run()
        assert any("Autosave failed" in e for e in _errors(at))
        assert any("Autosave failed" in e for e in at.session_state["errors"])
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    at.run()  # settle the rerun the failure triggered (AppTest drops the next
    # widget value otherwise); the banner is still there, nothing was saved
    assert any("Autosave failed" in e for e in at.session_state["errors"])
    assert json.loads(path.read_text())["models"]["freq"]["notes"] == ""
    at.text_input(key=wk(at, "notes_freq")).set_value("the share came back").run()
    assert not at.exception
    assert at.session_state["errors"] == []
    assert not any("Autosave" in e for e in _errors(at)), _errors(at)
    saved = json.loads(path.read_text())["models"]["freq"]["notes"]
    assert saved == "the share came back"


def test_s2_rename_follows_into_filters_and_derived_expressions(workspace):
    """S2: a rename rewrites pl.col('old') in row filters and derived columns
    (and nothing else in the expression)."""
    from easy_glm.workflow import Derived, prepare

    p = Project.from_json(workspace["project"])
    p.data.filters = ["pl.col('Exposure') > 0.02", 'pl.col("Exposure") < 0.99']
    p.data.derived = [
        Derived(name="claim_rate", expr="pl.col('ClaimNb') / pl.col('Exposure')"),
        Derived(name="tag", expr="pl.lit('Exposure')"),  # a text literal, not a column
    ]
    assert p.expressions_using("Exposure") == [
        "pl.col('Exposure') > 0.02",
        'pl.col("Exposure") < 0.99',
        "pl.col('ClaimNb') / pl.col('Exposure')",
    ]
    p.data.renames["Exposure"] = "exposure_years"
    assert p.rename_column("Exposure", "exposure_years") == ["freq"]
    assert p.data.filters == [
        "pl.col('exposure_years') > 0.02",
        "pl.col('exposure_years') < 0.99",
    ]
    assert [d.expr for d in p.data.derived] == [
        "pl.col('ClaimNb') / pl.col('exposure_years')",
        "pl.lit('Exposure')",  # untouched: only pl.col(...) references move
    ]
    assert p.data.roles["exposure_years"] == "weight"
    assert p.expressions_using("Exposure") == []
    # and the data steps still run after the rename (they used to fail with
    # "unable to find column Exposure")
    out = prepare(p, _frame())
    assert "exposure_years" in out.columns and out.height > 0


def test_s2_the_roles_grid_says_which_formulas_it_rewrote(workspace):
    from easy_glm.workflow import Derived

    p = Project.from_json(workspace["project"])
    p.data.filters = ["pl.col('Exposure') > 0.02"]
    p.data.derived = [
        Derived(name="rate", expr="pl.col('ClaimNb') / pl.col('Exposure')")
    ]
    cols = _frame().columns
    rows = [
        {
            "column": c,
            "rename to": "exposure_years" if c == "Exposure" else "",
            "role": p.data.roles.get(c, "unassigned"),
            "type": "auto",
        }
        for c in cols
    ]
    changed, notices = pv.apply_roles_grid(p, cols, rows)
    assert changed and not any(kind == "error" for kind, _ in notices)
    said = "\n".join(text for _, text in notices)
    assert "row filter / derived formula(s)" in said and "exposure_years" in said
    assert p.data.filters == ["pl.col('exposure_years') > 0.02"]


def test_s3_conflict_check_compares_content_not_only_the_timestamp(workspace):
    """S3: another session's write with the same mtime (a coarse-timestamp
    share) must still be seen as a conflict."""
    path = workspace["project"]
    p = Project.from_json(path)
    p.models["freq"].notes = "A" * 12
    p.to_json(path)
    at = _run(_script("pages_model", str(path)))
    seen = os.stat(path)
    other = json.loads(path.read_text())
    other["models"]["freq"]["notes"] = "B" * 12  # same number of bytes
    path.write_text(json.dumps(other, indent=2))
    os.utime(path, ns=(seen.st_atime_ns, seen.st_mtime_ns))  # a second-granular mount
    now = os.stat(path)
    assert (now.st_mtime_ns, now.st_size) == (seen.st_mtime_ns, seen.st_size)
    at.text_input(key=wk(at, "notes_freq")).set_value("this tab's edit").run()
    assert not at.exception
    assert any("changed by another browser tab" in w.value for w in at.warning)
    assert json.loads(path.read_text())["models"]["freq"]["notes"] == "B" * 12


@pytest.mark.skipif(os.name == "nt", reason="chmod does not deny the owner on Windows")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anything"
)
def test_s4_an_unreadable_data_file_keeps_the_persisted_fit(workspace):
    """S4: a share that blips must not delete the fit (a cross-validated lasso
    would have to be run again)."""
    script = _script("pages_model", str(workspace["project"]), fit=True)
    script += '\nst.session_state["_run"] = S.get_run("freq") is not None\n'
    at = _run(script, 240)
    assert at.session_state["_run"] is True
    runs = workspace["dir"] / "w3.easyglm-runs"
    persisted = sorted(runs.glob("*.pkl"))
    assert len(persisted) == 1
    stamp = persisted[0].stat().st_mtime_ns
    blind = _script("pages_model", str(workspace["project"]))
    blind += '\nst.session_state["_run"] = S.get_run("freq") is not None\n'
    os.chmod(workspace["data"], 0)
    try:
        at2 = _run(blind)
        assert any("Could not load" in e for e in _errors(at2)), _errors(at2)
        assert at2.session_state["_run"] is False  # nothing claims to be fitted
        assert sorted(runs.glob("*.pkl")) == persisted  # ... but the fit is kept
    finally:
        os.chmod(workspace["data"], stat.S_IREAD | stat.S_IWRITE)
    at3 = _run(blind)
    assert at3.session_state["_run"] is True  # restored, not refitted
    assert persisted[0].stat().st_mtime_ns == stamp


def _other_level_frame(n: int = 6000, seed: int = 11) -> pl.DataFrame:
    """A book whose Area column has a real level called ``Other`` plus rare
    levels that have to be lumped."""
    rng = np.random.default_rng(seed)
    area = rng.choice(["Other", "A", "B"], n, p=[0.4, 0.35, 0.25]).astype(object)
    rare = rng.random(n) < 0.12
    area[rare] = [f"Z{i % 40}" for i in range(int(rare.sum()))]
    expo = rng.uniform(0.2, 1.0, n)
    mu = np.exp(-2.0 + np.where(area == "A", 0.3, 0.0))
    return pl.DataFrame(
        {
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "Area": area,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


def _other_level_project() -> Project:
    p = Project(name="s5")
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "Area": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.design.defaults.min_level_share = 0.05
    p.new_model("m", divide_target_by_weight=True)
    p.models["m"].penalty.alpha = 0.002
    p.models["m"].penalty.cv = None
    return p


def test_s5_the_lumped_bucket_keeps_the_encoders_own_label():
    """S5: with a real level called "Other", the leftovers row must read
    "Other (lumped)" in the rate table, the workbook and the page — not
    "Other / Unknown" next to a level called "Other"."""
    from easy_glm.core.excel import rate_model_tables
    from easy_glm.workflow import run_model

    df = _other_level_frame()
    run = run_model(_other_level_project(), df, "m")
    enc = run.spec["Area"]
    assert enc.other_label == "Other (lumped)"
    labels = run.tables["Area"]["label"].to_list()
    assert labels[-1] == "Other (lumped)"
    assert "Other" in labels  # the real level is still there, and distinct
    assert "Other / Unknown" not in labels
    # the same label reaches the Rate tables page and the Excel export
    cfg = run.rate_model.variables["Area"]
    assert cfg.other_label == "Other (lumped)"
    assert rate_model_tables(run.rate_model)["Area"]["label"].to_list() == labels
    assert level_label(cfg.table[-1], cfg.other_label) == "Other (lumped)"
    # a round trip through a saved model file keeps it
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "s5.easyglm"
        run.rate_model.to_json(target)
        assert RateModel.from_json(target).variables["Area"].other_label == (
            "Other (lumped)"
        )


def test_s5_the_default_bucket_is_still_other_unknown(workspace):
    """The wording only changes when the encoder had to rename its bucket."""
    from easy_glm.core.excel import rate_model_tables
    from easy_glm.workflow import run_model

    p = Project.from_json(workspace["project"])
    p.models["freq"].predictors = ["Region"]
    run = run_model(p, _frame(), "freq")
    assert run.spec["Region"].other_label == "Other"
    assert run.tables["Region"]["label"].to_list()[-1] == "Other / Unknown"
    assert run.rate_model.variables["Region"].other_label is None
    labels = rate_model_tables(run.rate_model)["Region"]["label"].to_list()
    assert labels[-1] == "Other / Unknown"


def test_item_32_the_seed_box_shows_the_seed_the_project_holds(workspace):
    """Item 32: the box used to show a number the project did not hold (and an
    out-of-range seed in the file was a traceback)."""
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.seed = 99_999_999_999
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    assert not at.exception
    assert at.number_input(key=wk(at, "split_seed")).value == 99_999_999_999
    assert at.session_state["_project"].data.split.seed == 99_999_999_999
    assert any("0–10000" in w.value for w in at.warning), _texts(at)
    # a value the box will not take leaves the project's seed alone
    at.number_input(key=wk(at, "split_seed")).set_value(-5).run()
    assert not at.exception
    assert at.session_state["_project"].data.split.seed == 99_999_999_999
    assert at.number_input(key=wk(at, "split_seed")).value == 99_999_999_999
    # a seed in range is accepted, and the box then shows it
    at.number_input(key=wk(at, "split_seed")).set_value(7).run()
    assert at.session_state["_project"].data.split.seed == 7
    assert at.number_input(key=wk(at, "split_seed")).value == 7
    assert not any("0–10000" in w.value for w in at.warning)


def test_project_token_is_stable_while_you_stay_in_one_project(workspace):
    """Widget keys carry the project token: it must not rotate while you
    navigate inside one project (or every page would lose its boxes)."""
    path = str(workspace["project"])
    script = f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({path!r}), {path!r})
    st.session_state._loaded = True
if st.session_state.get("_open"):
    from easy_glm.app.pages_project import open_project_file

    st.session_state["_open_error"] = open_project_file(st.session_state.pop("_open"))
page = st.session_state.get("_page", "pages_variables")
importlib.import_module("easy_glm.app." + page).render()
st.session_state.setdefault("_tokens", []).append(st.session_state["project_token"])
"""
    at = _run(script)
    for page in ("pages_explore", "pages_design", "pages_model", "pages_variables"):
        at.session_state["_page"] = page
        at.run()
        assert not at.exception, page
    # an edit (autosave included) must not rotate it either
    at.session_state["_page"] = "pages_model"
    at.run()
    at.text_input(key=wk(at, "notes_freq")).set_value("a note").run()
    assert len(set(at.session_state["_tokens"])) == 1, at.session_state["_tokens"]
    # ... but loading another project does rotate it, so no widget carries a
    # value from the project that was open before
    other = workspace["dir"] / "other.easyglm-project.json"
    Project(name="other").to_json(other)
    at.session_state["_page"] = "pages_variables"
    at.session_state["_open"] = str(other)
    at.run()
    assert at.session_state["_open_error"] is None
    tokens = at.session_state["_tokens"]
    assert len(set(tokens)) == 2 and tokens[-1] != tokens[0]
