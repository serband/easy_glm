"""W4 — the persisted-run folder and the second breaker session's findings.

One test per finding of ``docs/reviews/w3-breakage-2.md``. The two blocking
findings (1 and 2) are about the runs folder next to the project file: it is
shared by every browser tab, and before W4 a tab that was not in step with the
project on disk could delete the fit that belonged to it. Tests named
``test_breakage2_NN_...`` where ``NN`` is the finding's number in that report
(items 10, 24, 28, 30, 31, 32, 33 and 38 are the old numbers of
``docs/reviews/w2-breakage.md``, as the report uses them).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.app import state as S  # noqa: E402
from easy_glm.workflow import Project, VariableDesign, build_design  # noqa: E402
from easy_glm.workflow.project import validate_model_name  # noqa: E402
from easy_glm.workflow.run import UnusableColumnError  # noqa: E402

N = 2000


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
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region,
            "constant": np.full(n, 3.0),  # one value everywhere (finding 38)
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


def _project(data_path: Path) -> Project:
    p = Project(name="w4")
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Region": "predictor",
        "constant": "ignore",
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
    project = tmp_path / "w4.easyglm-project.json"
    _project(data).to_json(project)
    return {
        "data": data,
        "project": project,
        "dir": tmp_path,
        "runs": tmp_path / "w4.easyglm-runs",
    }


def _script(
    page: str,
    project_path: str,
    *,
    fit: bool = False,
    prelude: str = "",
    body: str = "",
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
page_name = st.session_state.get("_page", {page!r})
importlib.import_module("easy_glm.app." + page_name).render()
{body}
st.session_state["_project"] = S.project()
st.session_state["_runs"] = sorted(st.session_state.runs)
"""


def _run(script: str, timeout: int = 180) -> AppTest:
    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(at: AppTest) -> str:
    parts = [w.value for w in at.warning] + [e.value for e in at.error]
    parts += [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(parts)


def _files(folder: Path) -> list[str]:
    return sorted(f.name for f in folder.glob("*")) if folder.exists() else []


def _snap(folder: Path) -> dict[str, tuple[int, int, str]]:
    """Every file in the folder by name, size, modification time in
    nanoseconds and a hash of the contents — markers included, so "nothing was
    written and nothing was deleted" is checked byte for byte."""
    if not folder.exists():
        return {}
    return {
        f.name: (
            f.stat().st_size,
            f.stat().st_mtime_ns,
            hashlib.sha1(f.read_bytes()).hexdigest(),
        )
        for f in sorted(folder.glob("*"))
    }


def _button(at: AppTest, label: str):
    return [b for b in at.button if b.label == label][0]


def _show_design(at: AppTest) -> None:
    at.session_state["_page"] = "pages_design"
    at.run()


# --------------------------------------------------------------------------
# data loss — the shared runs folder
# --------------------------------------------------------------------------
def test_breakage2_01_a_paused_tab_never_touches_the_runs_folder(workspace):
    """Finding 1: with the conflict notice up, tab B's Fit rewrote the run file
    and tab B's Delete removed the fit of the model the *project on disk* still
    contains — silently. A paused tab may now fit (for itself) but may not
    write to or delete from the folder."""
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path, fit=True), 240)
    tab_b = _run(_script("pages_model", path))
    assert "Fitted and up to date" in _texts(tab_b)
    before = _files(workspace["runs"])
    assert [f for f in before if f.endswith(".pkl")]

    # A edits and autosaves; B's next edit hits the conflict notice
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("B loses").run()
    assert any("changed by another browser tab" in w.value for w in tab_b.warning)

    # ... Fit in the paused tab: a result for this tab, nothing on disk
    tab_b.button(key=wk(tab_b, "fit_freq")).click().run()
    assert not tab_b.exception
    assert _files(workspace["runs"]) == before
    assert "shown in this tab only" in _texts(tab_b)

    # ... Delete in the paused tab: this tab's project only, files kept
    _show_design(tab_b)
    _button(tab_b, "Delete").click().run()
    assert not tab_b.exception
    assert _files(workspace["runs"]) == before
    assert "removed from this tab only" in _texts(tab_b)
    assert "freq" not in tab_b.session_state["_project"].models
    assert "freq" in json.loads(workspace["project"].read_text())["models"]

    # a fresh session still finds the fit the project on disk points at
    fresh = _run(_script("pages_model", path))
    assert "Fitted and up to date" in _texts(fresh)


def test_breakage2_02_a_stale_tabs_fit_keeps_the_run_of_the_saved_project(workspace):
    """Finding 2: tab B, one edit behind, used to delete tab A's run — the only
    one matching the project on disk — so the saved project said "Not fitted
    yet." Pruning now spares the saved project's run and this session's own."""
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path))
    tab_b = _run(_script("pages_model", path))

    tab_a.number_input(key=wk(tab_a, "alpha_freq")).set_value(0.004).run()
    tab_a.button(key=wk(tab_a, "fit_freq")).click().run()
    assert not tab_a.exception
    after_a = [f for f in _files(workspace["runs"]) if f.endswith(".pkl")]
    assert len(after_a) == 1
    assert json.loads(workspace["project"].read_text())["models"]["freq"]["penalty"][
        "alpha"
    ] == pytest.approx(0.004)

    # B is one edit behind (its alpha is still 0.002) and fits
    tab_b.button(key=wk(tab_b, "fit_freq")).click().run()
    assert not tab_b.exception
    both = [f for f in _files(workspace["runs"]) if f.endswith(".pkl")]
    assert len(both) == 2 and after_a[0] in both

    # a third, fresh session opens the saved project and finds its fit
    third = _run(_script("pages_model", path))
    assert "Fitted and up to date" in _texts(third)
    assert third.session_state["_project"].models["freq"].penalty.alpha == 0.004

    # and the rule itself: a session whose own spec is unsaved prunes the file
    # that belongs to neither it nor the project on disk, and only that one
    prelude = 'S.project().models["freq"].penalty.alpha = 0.006'
    fourth = _run(_script("pages_model", path, prelude=prelude))
    fourth.button(key=wk(fourth, "fit_freq")).click().run()
    assert not fourth.exception
    left = [f for f in _files(workspace["runs"]) if f.endswith(".pkl")]
    assert len(left) == 2 and after_a[0] in left  # the saved project's run kept
    assert sorted(set(both) - set(left)) == sorted(set(both) - {after_a[0]})


def test_breakage2_02_pruning_keeps_the_saved_projects_key(workspace):
    """The same rule at the level it is written: which keys ``_prune_runs``
    protects, without a browser."""
    body = """
S.fit_model("freq")
out = {}
folder = S.runs_dir()
out["after_fit"] = sorted(f.name for f in folder.glob("*.pkl"))
# a spec this session has not saved: its own fit and the saved one both stay
p.models["freq"].penalty.alpha = 0.009
S.fit_model("freq")
out["two"] = sorted(f.name for f in folder.glob("*.pkl"))
out["paused_writes"] = S.runs_write_paused()
st.session_state.conflict = st.session_state.project_path
out["paused_now"] = S.runs_write_paused() and S.runs_delete_paused()
out["persist_while_paused"] = S.persist_run("freq", S.get_run("freq"))
out["files_while_paused"] = sorted(f.name for f in folder.glob("*.pkl"))
st.session_state["out"] = out
"""
    script = f"""
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project
S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(workspace["project"])!r}), {str(workspace["project"])!r})
    st.session_state._loaded = True
p = S.project()
{body}
"""
    at = _run(script, 240)
    o = at.session_state["out"]
    assert len(o["after_fit"]) == 1
    assert len(o["two"]) == 2 and o["after_fit"][0] in o["two"]
    assert o["paused_writes"] is False and o["paused_now"] is True
    assert o["persist_while_paused"] is None
    assert o["files_while_paused"] == o["two"]


# --------------------------------------------------------------------------
# misleading
# --------------------------------------------------------------------------
def test_breakage2_03_create_switches_to_the_new_model(workspace):
    """Finding 3: "Model 'freq_v2' created" while the page below kept editing
    and fitting the champion."""
    path = str(workspace["project"])
    at = _run(_script("pages_design", path))
    at.text_input(key=wk(at, "model_new_name")).set_value("freq_v2").run()
    _button(at, "Create").click().run()
    assert not at.exception
    assert at.selectbox(key=wk(at, "model_select")).value == "freq_v2"
    assert at.session_state["model_current"] == "freq_v2"
    assert at.text_input(key=wk(at, "model_new_name")).value == ""

    # What is typed and fitted next belongs to the new model, not the champion.
    at.session_state["_page"] = "pages_model"
    at.run()
    assert at.selectbox(key=wk(at, "model_select")).value == "freq_v2"
    at.text_input(key=wk(at, "notes_freq_v2")).set_value("for v2").run()
    p = at.session_state["_project"]
    assert p.models["freq_v2"].notes == "for v2" and p.models["freq"].notes == ""

    # ... and so does Fit
    at.radio(key=wk(at, "pmode_freq_v2")).set_value("fixed").run()
    at.button(key=wk(at, "fit_freq_v2")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_runs"] == ["freq_v2"]


def test_breakage2_04_the_persist_banner_clears_once_saving_works(workspace):
    """Finding 4: one transient failure said "Could not persist the fit" on
    every page for the rest of the session, while fits were being saved."""
    prelude = "st.session_state.errors = ['Could not persist the fit: [Errno 13] no']"
    at = _run(_script("pages_model", str(workspace["project"]), prelude=prelude))
    assert any("Could not persist" in e.value for e in at.error)
    at.button(key=wk(at, "fit_freq")).click().run()
    assert not at.exception
    assert not any("Could not persist" in e.value for e in at.error)
    assert [f for f in _files(workspace["runs"]) if f.endswith(".pkl")]


def test_breakage2_05_an_out_of_range_alpha_is_refused_not_left_on_screen(workspace):
    """Finding 5: a pasted 1e9 stayed in the alpha box (as 190.00100) while the
    fit used 0.001 — the page named a penalty the model did not use."""
    at = _run(_script("pages_model", str(workspace["project"])))
    at.number_input(key=wk(at, "alpha_freq")).set_value(1e9).run()
    assert not at.exception
    assert any("alpha must be between 0 and 10" in e.value for e in at.error)
    assert at.number_input(key=wk(at, "alpha_freq")).value == pytest.approx(0.002)
    assert at.session_state["_project"].models["freq"].penalty.alpha == 0.002


def test_breakage2_07_the_status_chips_agree_with_the_refit_banner(workspace):
    """Finding 7: after a spec change the chips still said "✓ Fitted" next to
    "results below are from the previous fit"."""
    at = _run(_script("pages_model", str(workspace["project"]), fit=True), 240)
    chips = [m.value for m in at.markdown if "Fitted]" in m.value]
    assert chips and "✓ Fitted" in chips[0]
    at.session_state["_page"] = "pages_design"
    at.run()
    at.selectbox(key=wk(at, "fam_freq")).set_value("gamma").run()
    at.session_state["_page"] = "pages_model"
    at.run()
    assert not at.exception
    chips = [m.value for m in at.markdown if "Fitted]" in m.value]
    assert chips and "○ Fitted" in chips[0], chips
    assert "Spec changed since the last fit" in _texts(at)


def test_breakage2_09_an_orphaned_sidecar_is_removed(workspace):
    """Finding 9: deleting the pickle by hand left its .json next to it for
    ever."""
    path = str(workspace["project"])
    _run(_script("pages_model", path, fit=True), 240)
    pkl = next(iter(workspace["runs"].glob("*.pkl")))
    sidecar = pkl.with_suffix(".json")
    pkl.unlink()
    assert sidecar.exists()
    at = _run(_script("pages_model", path))
    assert "Not fitted yet." in _texts(at)
    assert not sidecar.exists()


def test_breakage2_10c_the_caption_names_the_missing_parent(workspace):
    """Finding 10 (cosmetic): the caption named the interaction, not the parent
    that had left the predictor list."""
    p = Project.from_json(workspace["project"])
    from easy_glm.workflow import Interaction

    p.models["freq"].interactions.append(Interaction(a="DrivAge", b="Region"))
    p.to_json(workspace["project"])
    at = _run(_script("pages_design", str(workspace["project"])))
    at.multiselect(key=wk(at, "preds_freq")).set_value(["DrivAge", "BonusMalus"]).run()
    assert not at.exception
    messages = _texts(at)
    assert "no longer among the predictors: Region" in messages, messages


# --------------------------------------------------------------------------
# the four findings the first session left open
# --------------------------------------------------------------------------
def test_breakage2_24_a_knot_above_the_data_is_accepted_and_flagged(workspace):
    """Item 24 (caveat): a knot above the largest training value was accepted
    in silence, and its bin has no training rows."""
    at = _run(_script("pages_design", str(workspace["project"])))
    at.selectbox(key=wk(at, "design_detail_var")).set_value("DrivAge").run()
    at.text_area(key=wk(at, "knots_DrivAge")).set_value("30, 40, 999999").run()
    at.button(key=wk(at, "apply_knots_DrivAge")).click().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "999999" in warnings and "no training rows" in warnings, warnings
    assert at.session_state["_project"].design.variables["DrivAge"].knots == [
        30.0,
        40.0,
        999999.0,
    ]
    assert at.session_state["_project"].design.variables["DrivAge"].n_bins == 4
    assert any("Custom knots are active" in i.value for i in at.info)


def test_quantile_knot_source_and_actual_bin_count_are_explained(workspace, tmp_path):
    """A requested bin count may yield fewer bins when quantiles repeat; the
    page names both numbers before the user decides whether to override them."""
    p = Project.from_json(workspace["project"])
    p.design.variables["DrivAge"] = VariableDesign(
        kind="step", knots="quantile", n_bins=50
    )
    path = tmp_path / "quantile-note.easyglm-project.json"
    p.to_json(path)
    at = _run(_script("pages_design", str(path)))
    at.selectbox(key=wk(at, "design_detail_var")).set_value("DrivAge").run()
    notes = " ".join(i.value for i in at.info)
    assert "requested 50 quantile bins" in notes
    assert "actual bins" in notes


def test_breakage2_28_an_interrupted_fit_is_reported_not_forgotten(workspace):
    """Item 28: a fit interrupted by a page reload left nothing behind — the
    next session simply said "Not fitted yet."."""
    path = str(workspace["project"])
    # a fit that is started and never saved leaves a marker ...
    body = 'S._mark_fit_started("freq", S.run_key(S.project(), "freq"))'
    started = _run(_script("pages_model", path, body=body))
    markers = list(workspace["runs"].glob("*.fitting"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_text())["model"] == "freq"
    assert "Not fitted yet." in _texts(started)

    # ... which the next session turns into a sentence
    at = _run(_script("pages_model", path))
    assert "was interrupted" in _texts(at)
    # the marker is left where it is: from here it is indistinguishable from a
    # fit running in another tab, and taking it away would cost that tab its
    # own warning (W4 review S3). It is still true, so it is said again ...
    assert list(workspace["runs"].glob("*.fitting"))
    again = _run(_script("pages_model", path))
    assert "was interrupted" in _texts(again)

    # ... until the fit is actually done: that clears this session's marker and
    # tidies the stale one, whose result is now on disk
    done = _run(_script("pages_model", path, fit=True), 240)
    assert not list(workspace["runs"].glob("*.fitting"))
    assert "was interrupted" not in _texts(done)
    after = _run(_script("pages_model", path))
    assert "was interrupted" not in _texts(after)


def test_breakage2_30_windows_device_names_are_refused(workspace):
    """Item 30 (caveat): CON / NUL / PRN were accepted, and their downloads
    cannot be written on Windows."""
    for bad in ("CON", "NUL", "prn", "AUX", "COM1", "LPT9", "CON.xlsx"):
        assert "reserved by Windows" in (validate_model_name(bad) or ""), bad
    for ok in ("CONS", "COM10", "freq_CON", "NULL_model"):
        assert validate_model_name(ok) is None, ok
    at = _run(_script("pages_design", str(workspace["project"])))
    at.text_input(key=wk(at, "model_new_name")).set_value("CON").run()
    assert _button(at, "Create").disabled
    assert any("reserved by Windows" in c.value for c in at.caption)
    assert "CON" not in at.session_state["_project"].models


def test_breakage2_31_the_divide_box_is_never_ticked_while_disabled(workspace):
    """Item 31: with the weight cleared the box stayed ticked (Streamlit keeps
    a widget key's value) while the project held False."""
    at = _run(_script("pages_design", str(workspace["project"])))
    assert at.checkbox(key=wk(at, "div_freq")).value is True
    at.selectbox(key=wk(at, "wgt_freq")).set_value("(none)").run()
    assert not at.exception
    box = at.checkbox(key=wk(at, "div_freq"))
    assert box.value is False and box.disabled
    assert at.session_state["_project"].models["freq"].divide_target_by_weight is False


def test_breakage2_32_the_seed_box_never_names_a_seed_the_split_lacks(workspace):
    """Item 32: a typed 99999 (or -5) stayed in the box while the project kept
    the old seed, with no message."""
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.seed = 99_999
    p.to_json(workspace["project"])
    path = str(workspace["project"])

    # the seed in the file is shown, and used
    at = _run(_script("pages_split", path))
    assert at.number_input(key=wk(at, "split_seed")).value == 99_999
    assert at.session_state["_project"].data.split.seed == 99_999
    assert any("0–10000" in w.value for w in at.warning), _texts(at)

    # a typed seed outside the usual range is taken, and said to be unusual
    at.number_input(key=wk(at, "split_seed")).set_value(7).run()
    assert at.session_state["_project"].data.split.seed == 7
    at.number_input(key=wk(at, "split_seed")).set_value(99_999).run()
    assert not at.exception
    assert at.session_state["_project"].data.split.seed == 99_999
    assert at.number_input(key=wk(at, "split_seed")).value == 99_999

    # a seed no split can use is refused, said so, and the box put back
    at.number_input(key=wk(at, "split_seed")).set_value(-5).run()
    assert not at.exception
    assert any("seed must be 0 or more" in e.value for e in at.error), _texts(at)
    assert at.session_state["_project"].data.split.seed == 99_999
    assert at.number_input(key=wk(at, "split_seed")).value == 99_999

    # ... and a negative seed hand-edited into the file is repaired, not used
    p.data.split.seed = -5
    p.to_json(workspace["project"])
    at2 = _run(_script("pages_split", path))
    assert at2.session_state["_project"].data.split.seed == 42
    assert any("negative" in w.value for w in at2.warning), _texts(at2)


def test_breakage2_33_the_clamped_fraction_says_what_it_changed(workspace):
    """Item 33 (caveat): the file was rewritten from 1.0 to 0.95 and the
    warning that explained it was lost in the rerun."""
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.fraction = 1.0
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "outside the 0.50–0.95 range" in warnings and "changed to 0.95" in warnings
    assert at.session_state["_project"].data.split.fraction == 0.95
    on_disk = json.loads(workspace["project"].read_text())["data"]["split"]["fraction"]
    assert on_disk == 0.95


def test_breakage2_38_a_constant_predictor_is_dropped_not_a_blocked_fit(workspace):
    """Item 38: one column with a single value stopped the whole fit. It is now
    left out of the design and named."""
    p = Project.from_json(workspace["project"])
    p.data.roles["constant"] = "predictor"
    p.models["freq"].predictors.append("constant")
    p.to_json(workspace["project"])
    at = _run(_script("pages_model", str(workspace["project"])), 240)
    at.button(key=wk(at, "fit_freq")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Fitted and up to date" in _texts(at)
    assert "constant" in _texts(at) and "all-null" in _texts(at)
    assert at.session_state["_runs"] == ["freq"]

    # the rule, without a browser: dropping is opt-in, and every predictor
    # being unusable is still an error
    train = _frame().filter(pl.col("traintest") == 1)
    with pytest.raises(UnusableColumnError, match="constant"):
        build_design(p, train, ["constant"])
    dropped: list[str] = []
    spec = build_design(p, train, ["DrivAge", "constant"], dropped=dropped)
    assert dropped == ["constant"] and list(spec.encoders) == ["DrivAge"]
    with pytest.raises(ValueError, match="Every predictor"):
        build_design(p, train, ["constant"], dropped=[])


def test_breakage2_10_a_modelling_column_as_indicator_needs_confirmation(workspace):
    """Item 10 (caveat): picking a predictor as the train/holdout indicator
    took its predictor role away without a word."""
    p = Project.from_json(workspace["project"])
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    at.selectbox(key=wk(at, "split_col")).set_value("Region").run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "currently the **predictor**" in warnings and "freq" in warnings
    project = at.session_state["_project"]
    assert project.data.split.column == "traintest"
    assert project.data.roles["Region"] == "predictor"
    at.button(key=wk(at, "split_role_btn_Region")).click().run()
    assert not at.exception
    project = at.session_state["_project"]
    assert project.data.split.column == "Region"
    assert project.data.roles["Region"] == "split"


def test_breakage2_11_a_typo_in_the_path_does_not_create_folders(workspace):
    """Finding 11: a mistyped project path silently created a folder tree and
    moved the project into it."""
    body = f"""
missing = {str(workspace["dir"] / "typo_dir" / "deep" / "x.json")!r}
st.session_state["_err"] = S.save_project(missing)
st.session_state["_path_now"] = st.session_state.get("project_path")
"""
    at = _run(_script("pages_project", str(workspace["project"]), body=body))
    err = at.session_state["_err"]
    assert err and "does not exist" in err and "typo_dir" in err
    assert not (workspace["dir"] / "typo_dir").exists()
    assert at.session_state["_path_now"] == str(workspace["project"])


def test_breakage2_14_row_counts_and_the_prepared_chip(workspace):
    """Finding 14: "sample of 1 rows", and ✓ Prepared for a frame with no
    rows."""
    p = Project.from_json(workspace["project"])
    p.data.sample_rows = 1
    p.to_json(workspace["project"])
    at = _run(_script("pages_explore", str(workspace["project"])))
    captions = " ".join(c.value for c in at.caption)
    assert "sample of 1 row " in captions and "1 rows" not in captions

    p = Project.from_json(workspace["project"])
    p.data.sample_rows = None
    p.data.filters = ["pl.col('ClaimNb') < -1"]  # keeps nothing
    p.to_json(workspace["project"])
    at = _run(_script("pages_model", str(workspace["project"])))
    chips = [m.value for m in at.markdown if "Prepared]" in m.value]
    assert chips and "○ Prepared" in chips[0], chips


def test_w4_state_helpers_are_documented_rules():
    """The three folder rules live in one place each, so the check page and the
    reviewer can point at them."""
    assert "conflict" in (S.runs_write_paused.__doc__ or "")
    assert "in step with" in (S.runs_delete_paused.__doc__ or "")
    assert "saved on disk" in (S._prune_runs.__doc__ or "")


# --------------------------------------------------------------------------
# the W4 review's should-fix items and missing tests
# --------------------------------------------------------------------------
def _foreign_marker(workspace, session: str = "deadbeef") -> Path:
    """A "fit in progress" marker as another browser session would leave it —
    a key nothing was ever saved under, so it reads as a live fit."""
    workspace["runs"].mkdir(exist_ok=True)
    marker = workspace["runs"] / f"{S._model_tag('freq')}-{'a' * 16}-{session}.fitting"
    marker.write_text(json.dumps({"model": "freq", "session": session, "pid": 1}))
    return marker


def test_breakage2_s1_markers_obey_the_same_pause_rules(workspace):
    """W4 review S1: a paused tab unlinked another tab's "fit in progress"
    marker, so the folder moved after all — and the tab that really was
    fitting lost its own warning."""
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path, fit=True), 240)
    marker = _foreign_marker(workspace)
    tab_b = _run(_script("pages_model", path))

    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("B loses").run()
    assert any("changed by another browser tab" in w.value for w in tab_b.warning)
    before = _snap(workspace["runs"])  # everything tab A did is now in place
    tab_b.button(key=wk(tab_b, "fit_freq")).click().run()
    assert not tab_b.exception
    assert _snap(workspace["runs"]) == before  # markers included

    # a tab that *is* in step reports the fit it can see, but still leaves a
    # marker that may belong to a fit running somewhere else
    fresh = _run(_script("pages_model", path))
    assert "was interrupted" in _texts(fresh)
    assert marker.exists()

    # ... once it is older than the grace period, it is tidied away
    old = time.time() - S.MARKER_GRACE_SECONDS - 10
    os.utime(marker, (old, old))
    later = _run(_script("pages_model", path))
    assert "was interrupted" in _texts(later)
    assert not marker.exists()


def test_breakage2_s1_a_finished_fit_leaves_only_its_own_marker_behind(workspace):
    """A fit clears the marker it wrote, not every marker of that model."""
    path = str(workspace["project"])
    marker = _foreign_marker(workspace)
    at = _run(_script("pages_model", path), 240)
    at.button(key=wk(at, "fit_freq")).click().run()
    assert not at.exception
    assert marker.exists()  # another session's marker is not this fit's to clear
    mine = [
        f
        for f in workspace["runs"].glob("*.fitting")
        if f != marker  # this session's own marker went with the saved run
    ]
    assert mine == []


def test_breakage2_s2_delete_says_so_before_the_conflict_notice_is_up(workspace):
    """W4 review S2: with the file changed but no notice showing yet, Delete
    was refused on disk (right) and said nothing (wrong) — touch() reran before
    the flash was queued."""
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path, fit=True), 240)
    tab_b = _run(_script("pages_model", path))
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    before = _snap(workspace["runs"])  # tab B has no conflict notice yet

    _show_design(tab_b)
    _button(tab_b, "Delete").click().run()
    assert not tab_b.exception
    assert _snap(workspace["runs"]) == before
    assert "removed from this tab only" in _texts(tab_b)
    assert any("changed by another browser tab" in w.value for w in tab_b.warning)
    assert "freq" in json.loads(workspace["project"].read_text())["models"]


@pytest.mark.parametrize("resolution", ["conflict_reload", "conflict_overwrite"])
def test_breakage2_s3_deleting_resumes_once_the_conflict_is_resolved(
    workspace, resolution
):
    """The other half of the rule: the pause must lift. After Reload or
    Overwrite a Delete really removes the model and its fit."""
    path = str(workspace["project"])
    tab_a = _run(_script("pages_model", path, fit=True), 240)
    tab_b = _run(_script("pages_model", path))
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("B loses").run()
    assert any("changed by another browser tab" in w.value for w in tab_b.warning)

    tab_b.button(key=resolution).click().run()
    assert not tab_b.exception
    assert not any("changed by another" in w.value for w in tab_b.warning)
    _show_design(tab_b)
    _button(tab_b, "Delete").click().run()
    assert not tab_b.exception
    assert "removed from this tab only" not in _texts(tab_b)
    assert not [f for f in _files(workspace["runs"]) if f.endswith(".pkl")]
    assert json.loads(workspace["project"].read_text())["models"] == {}


def test_breakage2_03b_delete_moves_the_picker_too(workspace):
    """The companion of finding 3: after Delete the picker must be on a model
    that still exists (and on nothing when the last one goes)."""
    path = str(workspace["project"])
    at = _run(_script("pages_design", path))
    at.text_input(key=wk(at, "model_new_name")).set_value("freq_v2").run()
    _button(at, "Create").click().run()
    assert at.selectbox(key=wk(at, "model_select")).value == "freq_v2"

    _button(at, "Delete").click().run()  # deletes the selected freq_v2
    assert not at.exception
    assert at.session_state["model_current"] == "freq"
    assert at.selectbox(key=wk(at, "model_select")).value == "freq"
    assert list(at.session_state["_project"].models) == ["freq"]

    _button(at, "Delete").click().run()  # ... and the last one
    assert not at.exception
    assert at.session_state["model_current"] is None
    assert at.session_state["_project"].models == {}
    assert "Define a model to start" in _texts(at)


def test_breakage2_nit_number_boxes_take_whole_seeds_and_refuse_negatives(workspace):
    """W4 review nit 1 and missing test 5: a fractional seed keeps the box and
    the project in step (whole numbers only, said in the box's help), and a
    negative base-rate override is refused like a negative seed."""
    p = Project.from_json(workspace["project"])
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.seed = 7
    p.to_json(workspace["project"])
    at = _run(_script("pages_split", str(workspace["project"])))
    assert "whole number" in at.number_input(key=wk(at, "split_seed")).help
    at.number_input(key=wk(at, "split_seed")).set_value(2.5).run()
    assert not at.exception
    shown = at.number_input(key=wk(at, "split_seed")).value
    assert shown == at.session_state["_project"].data.split.seed == 2

    at2 = _run(_script("pages_model", str(workspace["project"])))
    at2.number_input(key=wk(at2, "bro_freq")).set_value(-1.0).run()
    assert not at2.exception
    assert any("must be 0 or more" in e.value for e in at2.error), _texts(at2)
    assert at2.number_input(key=wk(at2, "bro_freq")).value == 0.0
    assert at2.session_state["_project"].models["freq"].base_rate_override is None
