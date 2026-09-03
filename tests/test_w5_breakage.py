"""W5 / breaker session 3 — one test per finding of
``docs/reviews/w5-breakage-3.md``, on the surfaces added since W4: Compare,
the HTML report, the Rate tables Tools/Undo/Redo/Snapshots/Rebalance panel,
the rate-change flow, penalty weights, Tweedie power/binomial, cells alpha,
the CLI and the compact-matrix path (>200k rows).

Most findings are a hand-edited project file putting the wrong type in a
numeric field (``"alpha": "abc"``): ``Project.validate`` and several page
widgets assumed a real number and raised straight out of a comparison or a
format spec instead of reporting one more problem — a crash reachable simply
by opening the Model or Design page on such a file, or running
``easy-glm validate/run/export`` on it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.workflow import Interaction, Project  # noqa: E402

N = 3000


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
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


def _project(data_path) -> Project:
    p = Project(name="w5")
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
    p.new_model("m1", divide_target_by_weight=True)
    p.models["m1"].predictors = ["DrivAge", "Region"]
    p.models["m1"].penalty.alpha = 0.002
    p.models["m1"].penalty.cv = None
    p.models["m1"].interactions = [Interaction("DrivAge", "Region")]
    return p


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "policies.parquet"
    _frame().write_parquet(data)
    project_path = tmp_path / "w5.easyglm-project.json"
    _project(data).to_json(project_path)
    return {"data": data, "project": project_path, "dir": tmp_path}


def _script(page: str, project_path, *, autosave: bool = False) -> str:
    # ``autosave=True`` gives the project a path, the way a real session does,
    # so ``S.touch()`` actually writes the repaired value back — the read-only
    # smoke tests elsewhere in this suite pass ``None`` on purpose to keep the
    # file untouched.
    path_arg = repr(str(project_path)) if autosave else "None"
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(project_path)!r}), {path_arg})
    st.session_state._loaded = True
importlib.import_module("easy_glm.app." + {page!r}).render()
"""


def _edit(path, mutate) -> None:
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))


# --------------------------------------------------------------------------
# finding 1: Project.validate() crashes on a non-numeric field
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("alpha", "abc"),
        lambda raw: (
            raw["models"]["m1"].__setitem__("family", "tweedie"),
            raw["models"]["m1"].__setitem__("tweedie_power", "abc"),
        ),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__(
            "penalty_weight", "abc"
        ),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__(
            "min_cell_exposure", "abc"
        ),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__("alpha", "abc"),
        lambda raw: raw["design"]["variables"].__setitem__(
            "DrivAge", {"clamp": ["abc", 10]}
        ),
        lambda raw: (
            raw["data"]["split"].__setitem__("mode", "random"),
            raw["data"]["split"].__setitem__("fraction", "abc"),
        ),
    ],
    ids=[
        "alpha",
        "tweedie_power",
        "interaction_penalty_weight",
        "interaction_min_cell_exposure",
        "interaction_alpha",
        "clamp",
        "split_fraction",
    ],
)
def test_finding_1_validate_reports_a_bad_type_instead_of_raising(workspace, mutate):
    _edit(workspace["project"], mutate)
    p = Project.from_json(workspace["project"])
    # this used to raise TypeError/ValueError straight out of validate()
    problems = p.validate("m1")
    assert isinstance(problems, list) and problems


# --------------------------------------------------------------------------
# finding 2: the Model and Design pages crash rendering the same bad fields
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("alpha", "abc"),
        lambda raw: (
            raw["models"]["m1"].__setitem__("family", "tweedie"),
            raw["models"]["m1"].__setitem__("tweedie_power", "abc"),
        ),
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("l1_ratio", "abc"),
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("cv", "abc"),
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("n_alphas", "abc"),
        lambda raw: raw["models"]["m1"].__setitem__("base_rate_override", "abc"),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__("alpha", "abc"),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__(
            "penalty_weight", "abc"
        ),
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__(
            "min_cell_exposure", "abc"
        ),
    ],
    ids=[
        "alpha",
        "tweedie_power",
        "l1_ratio",
        "cv",
        "n_alphas",
        "base_rate_override",
        "interaction_alpha",
        "interaction_penalty_weight",
        "interaction_min_cell_exposure",
    ],
)
@pytest.mark.parametrize("page", ["pages_model", "pages_design"])
def test_finding_2_model_and_design_pages_render_with_a_bad_field(
    workspace, mutate, page
):
    _edit(workspace["project"], mutate)
    at = AppTest.from_string(_script(page, workspace["project"]), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_finding_2b_bad_clamp_does_not_crash_the_design_page(workspace):
    _edit(
        workspace["project"],
        lambda raw: raw["design"]["variables"].__setitem__(
            "DrivAge", {"kind": "linear", "clamp": ["abc", 10]}
        ),
    )
    at = AppTest.from_string(
        _script("pages_design", workspace["project"]), default_timeout=120
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------
# finding 4: repairing a bad field must say so, not silently autosave over it
# --------------------------------------------------------------------------
def test_finding_4_repairing_alpha_is_explained_not_silent(workspace):
    """Once the Model page can render past a non-numeric ``alpha`` (finding
    2), it also reconciles every widget's value back into the project on the
    very same run — including this one, now a fallback number — and
    autosaves. Without an explanation that replaces the user's mistake with a
    made-up number and no trace of what happened, exactly the "autosave of a
    bad state" class the plan's break-it catalogue warns about, just with a
    repaired number instead of a crash."""
    _edit(
        workspace["project"],
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("alpha", "abc"),
    )
    at = AppTest.from_string(
        _script("pages_model", workspace["project"], autosave=True),
        default_timeout=120,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(
        "alpha" in w.value and "'abc'" in w.value and "0.001" in w.value
        for w in at.warning
    )
    saved = json.loads(workspace["project"].read_text())
    assert saved["models"]["m1"]["penalty"]["alpha"] == 0.001


def test_finding_4b_repairing_an_interaction_alpha_is_explained(workspace):
    _edit(
        workspace["project"],
        lambda raw: raw["models"]["m1"]["interactions"][0].__setitem__("alpha", "xyz"),
    )
    at = AppTest.from_string(
        _script("pages_design", workspace["project"], autosave=True),
        default_timeout=120,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("DrivAge×Region" in w.value and "'xyz'" in w.value for w in at.warning)
    saved = json.loads(workspace["project"].read_text())
    assert saved["models"]["m1"]["interactions"][0].get("alpha") is None


def test_finding_4c_a_legitimate_none_alpha_is_never_reported_as_a_problem(workspace):
    """``base_rate_override``/``penalty.alpha``/an interaction's ``alpha`` are
    all optional (``None`` = "not set"); the repair message must fire only for
    an actually-unusable value, never for the ordinary unset case."""
    at = AppTest.from_string(
        _script("pages_model", workspace["project"], autosave=True),
        default_timeout=120,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("is not a usable number" in w.value for w in at.warning)


# --------------------------------------------------------------------------
# finding 5 (review round 2, blocking B1): a real number outside the
# widget's UI range must be repaired-and-explained, not silently autosaved
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate,field,bad,repaired",
    [
        (
            lambda raw: raw["models"]["m1"]["penalty"].__setitem__("l1_ratio", 5.0),
            "l1_ratio",
            "5",
            1.0,
        ),
        (
            lambda raw: (
                raw["models"]["m1"]["penalty"].__setitem__("alpha", None),
                raw["models"]["m1"]["penalty"].__setitem__("cv", 5),
                raw["models"]["m1"]["penalty"].__setitem__("n_alphas", -5),
            ),
            "n_alphas",
            "-5",
            3,
        ),
        (
            lambda raw: (
                raw["models"]["m1"]["penalty"].__setitem__("alpha", None),
                raw["models"]["m1"]["penalty"].__setitem__("cv", 999),
            ),
            "cv",
            "999",
            10,
        ),
    ],
    ids=["l1_ratio_above_1", "n_alphas_negative", "cv_above_widget_range"],
)
def test_finding_5_an_out_of_range_penalty_field_is_explained_not_silent(
    workspace, mutate, field, bad, repaired
):
    """A hand-edited ``l1_ratio``/``n_alphas``/``cv`` that is a real, finite
    number but outside the widget's declared UI range used to be clamped by a
    bare ``min(hi, max(lo, ...))`` with no message, and the reconciliation
    loop then autosaved that clamped value over the original — silent data
    loss, and for ``l1_ratio`` specifically a regression from release-0.4
    (``st.slider`` never enforced its own bounds, so the hand-edited value
    used to round-trip unchanged)."""
    _edit(workspace["project"], mutate)
    at = AppTest.from_string(
        _script("pages_model", workspace["project"], autosave=True),
        default_timeout=120,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(
        field in w.value and bad in w.value and f"{repaired:g}" in w.value
        for w in at.warning
    ), [w.value for w in at.warning]
    saved = json.loads(workspace["project"].read_text())
    assert saved["models"]["m1"]["penalty"][field] == repaired


def test_finding_5_validate_reports_an_out_of_range_penalty_field(workspace):
    """``Project.validate()`` itself must catch these three, independent of
    what any page happens to do with them (a project validated by the CLI, or
    a page that stops reading a field through the repairing helper, must
    still refuse to fit)."""
    p = Project.from_json(workspace["project"])
    p.models["m1"].penalty.l1_ratio = 5.0
    assert any("l1_ratio" in m for m in p.validate("m1"))
    p.models["m1"].penalty.l1_ratio = 1.0

    p.models["m1"].penalty.n_alphas = -5
    assert any("n_alphas" in m for m in p.validate("m1"))
    p.models["m1"].penalty.n_alphas = 20

    p.models["m1"].penalty.cv = 1
    assert any("cv" in m for m in p.validate("m1"))
    p.models["m1"].penalty.cv = None

    assert p.validate("m1") == []


# --------------------------------------------------------------------------
# finding 6 (review round 2, should-fix S1): ui.number_in_range must not
# loop forever when the *stored* value is a legal number outside lo/hi
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate,field",
    [
        (
            lambda raw: raw["models"]["m1"]["penalty"].__setitem__("alpha", 1e9),
            "alpha",
        ),
        (
            lambda raw: raw["models"]["m1"].__setitem__("base_rate_override", -1.0),
            "base rate override",
        ),
    ],
    ids=["alpha", "base_rate_override"],
)
def test_finding_6_number_in_range_repairs_an_out_of_range_stored_value_once(
    workspace, mutate, field
):
    """``ui.number_in_range`` used to rebuild the widget around the stored
    value on every rerun, including a repair rerun, so a legal-but-out-of-
    range stored value (``alpha: 1e9``, ``lo=0, hi=10``) reproposed the same
    invalid value forever — an infinite rerun loop, confirmed by an
    ``AppTest`` that never returns within a short timeout. A short
    ``default_timeout`` here is the point: this test times out (raises)
    before the fix and returns well within it after."""
    _edit(workspace["project"], mutate)
    at = AppTest.from_string(
        _script("pages_model", workspace["project"], autosave=True),
        default_timeout=8,
    )
    at.run()  # must return, not raise "AppTest script run timed out"
    assert not at.exception, [e.value for e in at.exception]
    assert any(field in w.value for w in at.warning)


# --------------------------------------------------------------------------
# finding 3: the CLI must never let this reach the user as a traceback
# --------------------------------------------------------------------------
def test_finding_3_cli_validate_reports_a_message_not_a_traceback(workspace):
    import subprocess
    import sys

    _edit(
        workspace["project"],
        lambda raw: raw["models"]["m1"]["penalty"].__setitem__("alpha", "abc"),
    )
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, "-m", "easy_glm.cli", "validate", str(workspace["project"])],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip()
