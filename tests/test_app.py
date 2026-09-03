"""Smoke tests for the workbench pages using Streamlit's AppTest.

Each page is rendered against a small synthetic project (data file on disk)
and must not raise. A fitted model is created through the same state helpers
the pages use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm.workflow import Project  # noqa: E402


@pytest.fixture(scope="module")
def project_file(tmp_path_factory) -> str:
    rng = np.random.default_rng(5)
    n = 3000
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
    df = pl.DataFrame(
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
    folder = tmp_path_factory.mktemp("app")
    data = folder / "policies.parquet"
    df.write_parquet(data)
    p = Project(name="apptest")
    p.data.source.type = "parquet"
    p.data.source.path = str(data)
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
    path = folder / "apptest.easyglm-project.json"
    p.to_json(path)
    return str(path)


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


def _script(page: str, project_path: str, fit: bool) -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), None)
    st.session_state._loaded = True
if {fit!r} and S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app." + {page!r}).render()
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_a_fit(page, project_file):
    at = AppTest.from_string(
        _script(page, project_file, fit=False), default_timeout=120
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize(
    "page", ["pages_model", "pages_diagnostics", "pages_tables", "pages_export"]
)
def test_page_renders_with_a_fit(page, project_file):
    at = AppTest.from_string(_script(page, project_file, fit=True), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "pages_export":
        code = "\n".join(c.value for c in at.code)
        assert "StepEncoder('DrivAge'" in code and "fit_glm(" in code
    if page == "pages_model":
        assert any("Fitted" in m.value for m in at.success)


@pytest.mark.parametrize(
    "page",
    [
        "pages_design",
        "pages_model",
        "pages_diagnostics",
        "pages_tables",
        "pages_export",
    ],
)
def test_pages_render_with_an_interaction(page, project_file, tmp_path):
    from easy_glm.workflow import Interaction

    p = Project.from_json(project_file)
    p.models["freq"].interactions = [
        Interaction("DrivAge", "Region", min_cell_exposure=0.02)
    ]
    path = tmp_path / "with_interaction.json"
    p.to_json(path)
    at = AppTest.from_string(_script(page, str(path), fit=True), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "pages_export":
        code = "\n".join(c.value for c in at.code)
        assert "InteractionEncoder(" in code


def test_main_entry_point_renders(project_file):
    import sys

    argv = sys.argv
    sys.argv = ["main.py", f"--project={project_file}"]
    try:
        import easy_glm.app as app_pkg

        main_py = str(Path(app_pkg.__file__).with_name("main.py"))
        at = AppTest.from_file(main_py, default_timeout=120)
        at.run()
    finally:
        sys.argv = argv
    assert not at.exception, [e.value for e in at.exception]
    assert any("apptest" in m.value for m in at.sidebar.markdown)


def test_leakage_page_actions(project_file):
    script = _script("pages_explore", project_file, fit=False)
    at = AppTest.from_string(script, default_timeout=180)
    at.run()
    assert not at.exception
    # run the scan
    buttons = [b for b in at.button if "scan" in b.label.lower()]
    assert buttons
    buttons[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.dataframe  # the report table rendered
