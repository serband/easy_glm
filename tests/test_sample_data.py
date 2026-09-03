"""French-motor sample entry point on the Project & data page."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from easy_glm import DesignSpec, fit_two_stage, rate_tables  # noqa: E402
from easy_glm.workflow import Project, double_lift, totals  # noqa: E402
from easy_glm.workflow.project import ModelConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"


def _script(loader: str) -> str:
    loader = loader.replace("\n", "\n    ")
    return f"""
import streamlit as st
from easy_glm.app import pages_design, pages_model, pages_project
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project(name="untitled"), None)
    {loader}
    st.session_state._loaded = True
if st.session_state.get("_page") == "model":
    pages_model.render()
elif st.session_state.get("_page") == "design":
    pages_design.render()
else:
    pages_project.render()
st.session_state["_project"] = S.project()
"""


def _run(loader: str) -> AppTest:
    at = AppTest.from_string(_script(loader), default_timeout=180)
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    return at


def _sample_button(at: AppTest):
    return next(
        button for button in at.button if button.label == "Use the French motor sample"
    )


def test_empty_project_explains_when_to_open_an_easyglm_project_file():
    at = _run("pass")
    uploader = next(
        field
        for field in at.file_uploader
        if field.label == "Open an existing EasyGLM project"
    )
    assert uploader
    text = " ".join(item.value for items in (at.info, at.caption) for item in items)
    assert "Resuming previous work" in text
    assert "data-file location" in text
    assert "column roles and types" in text
    assert "recodes, derived columns and filters" in text
    assert "train/test split" in text
    assert "model definitions and rate-table adjustments" in text
    assert "does not contain the data itself or fitted results" in text
    assert "French motor sample" in text


def test_french_motor_interaction_uses_its_own_cv_path_and_changes_predictions():
    """A real book guards against cells silently inheriting a main-stage alpha.

    The full model has a detectable BonusMalus × Area residual effect.  The
    stage-two path must retain it, so its predictions and double-lift A/E line
    differ from the frozen mains.
    """
    train = pl.read_parquet(FIXTURE).sample(fraction=0.7, seed=42)
    predictors = ["DrivAge", "Region", "BonusMalus", "Density", "Area"]
    spec = DesignSpec.from_data(
        train,
        predictors,
        weight_col="Exposure",
        interactions=[("BonusMalus", "Area")],
        min_cell_exposure=0.0,
    )
    fit = fit_two_stage(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        cv=5,
        n_alphas=20,
    )
    assert fit.stage1.model.cv == fit.stage2.model.cv == 5
    assert fit.stage1.model.n_alphas == fit.stage2.model.n_alphas == 20
    assert (fit.stage2.coef != 0).sum() > 0
    cells = rate_tables(fit)["BonusMalus×Area"]
    assert cells["relativity"].max() > 1.1
    assert cells["relativity"].min() < 0.95

    cfg = ModelConfig(target="ClaimNb", weight="Exposure", divide_target_by_weight=True)
    actual, expected_cells, weight = totals(train, cfg, fit.predict(train))
    expected_mains = totals(train, cfg, fit.stage1.predict(train))[1]
    assert not (expected_cells == expected_mains).all()
    double = double_lift(actual, expected_cells, expected_mains, weight)
    assert (double["ae_a"] - double["ae_b"]).abs().max() > 0.01


def _button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _empty_model_script() -> str:
    return f"""
import streamlit as st
from easy_glm.app import pages_design, pages_model
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    project = Project(name="untitled")
    project.data.source.path = {str(FIXTURE)!r}
    project.data.source.type = "parquet"
    project.data.roles = {{
        "ClaimNb": "target",
        "Exposure": "weight",
        "DrivAge": "predictor",
    }}
    project.data.split.mode = "random"
    S.set_project(project, None)
    st.session_state._loaded = True
if st.session_state.get("_page") == "design":
    pages_design.render()
else:
    pages_model.render()
st.session_state["_project"] = S.project()
"""


@pytest.fixture
def two_model_project(tmp_path) -> str:
    data = tmp_path / "french_motor_small.parquet"
    pl.read_parquet(FIXTURE).head(1_000).write_parquet(data)
    project = Project(name="two-models")
    project.data.source.path = str(data)
    project.data.source.type = "parquet"
    project.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "Region": "predictor",
    }
    project.data.split.mode = "random"
    project.data.split.column = "traintest"
    project.data.split.fraction = 0.7
    project.data.split.seed = 42
    path = tmp_path / "two-models.easyglm-project.json"
    project.to_json(path)
    return str(path)


def _two_model_script(project_path: str) -> str:
    return f"""
import streamlit as st
from easy_glm.app import pages_design, pages_diagnostics, pages_model
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), {project_path!r})
    st.session_state._loaded = True
if st.session_state.get("_page") == "diagnostics":
    pages_diagnostics.render()
elif st.session_state.get("_page") == "design":
    pages_design.render()
else:
    pages_model.render()
"""


def _key(at: AppTest, name: str) -> str:
    return f"{name}_{at.session_state['project_token']}"


def test_sample_data_button_loads_a_ready_french_motor_project():
    at = _run(
        "import easy_glm\n"
        "import polars as pl\n"
        f"easy_glm.load_external_dataframe = lambda: pl.read_parquet({str(FIXTURE)!r})"
    )

    _sample_button(at).click().run()
    assert not at.exception, [exception.value for exception in at.exception]
    project = at.session_state["_project"]
    source = Path(project.data.source.path)
    assert source.name == "french_motor_sample.parquet" and source.is_file()
    assert project.data.source.type == "parquet"
    assert project.data.roles == {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "Region": "predictor",
        "BonusMalus": "predictor",
        "Density": "predictor",
    }
    assert project.data.split.mode == "random"
    assert project.data.split.column == "traintest"
    assert project.data.split.fraction == 0.7
    assert project.data.split.seed == 42
    assert list(project.models) == ["frequency"]
    starter = project.models["frequency"]
    assert starter.family == "poisson"
    assert starter.target == "ClaimNb" and starter.weight == "Exposure"
    assert starter.divide_target_by_weight is True
    assert starter.predictors == ["DrivAge", "Region", "BonusMalus", "Density"]
    assert at.session_state["raw"][1].height == 50_000
    assert at.dataframe

    at.session_state["_page"] = "design"
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    assert (
        next(select.value for select in at.selectbox if select.label == "Model")
        == "frequency"
    )
    assert at.selectbox(key=_key(at, "fam_frequency")).value == "poisson"

    at.session_state["_page"] = "model"
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    assert (
        next(select.value for select in at.selectbox if select.label == "Model")
        == "frequency"
    )
    assert not _button(at, "Fit model").disabled
    assert any("ready to fit" in info.value for info in at.info)


def test_sample_data_loader_failure_is_a_page_message():
    at = _run(
        "import easy_glm\n"
        "def unavailable():\n"
        "    raise OSError('offline')\n"
        "easy_glm.load_external_dataframe = unavailable"
    )

    _sample_button(at).click().run()
    assert not at.exception, [exception.value for exception in at.exception]
    assert any(
        "Could not load the French motor sample" in error.value for error in at.error
    )


def test_model_without_a_definition_can_be_created_in_the_single_workflow():
    at = AppTest.from_string(_empty_model_script(), default_timeout=180)
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    create = _button(at, "Create")
    assert create.disabled
    name_input = next(
        field for field in at.text_input if field.label.startswith("New model name")
    )
    assert "required to enable Create" in name_input.label
    assert any("enter a valid model name" in info.value.lower() for info in at.info)

    name_input.set_value("frequency").run()
    assert not _button(at, "Create").disabled
    _button(at, "Create").click().run()
    assert not at.exception, [exception.value for exception in at.exception]
    assert not _button(at, "Fit model").disabled


def test_model_combines_definition_factor_design_and_fit_controls():
    at = AppTest.from_string(_empty_model_script(), default_timeout=180)
    at.run()
    at.text_input(key=_key(at, "model_new_name")).set_value("frequency").run()
    _button(at, "Create").click().run()

    assert at.selectbox(key=_key(at, "fam_frequency")).value == "poisson"
    assert at.selectbox(key=_key(at, "tgt_frequency")).value == "ClaimNb"
    assert at.selectbox(key=_key(at, "wgt_frequency")).value == "Exposure"
    assert at.selectbox(key=_key(at, "off_frequency")).value == "(none)"
    assert at.checkbox(key=_key(at, "div_frequency")).value is False
    assert "DrivAge" in at.multiselect(key=_key(at, "preds_frequency")).value

    at.selectbox(key=_key(at, "fam_frequency")).set_value("gamma").run()
    at.selectbox(key=_key(at, "off_frequency")).set_value("VehAge").run()
    at.checkbox(key=_key(at, "div_frequency")).set_value(True).run()
    at.checkbox(key=_key(at, "div_frequency")).set_value(False).run()
    at.multiselect(key=_key(at, "preds_frequency")).set_value(["DrivAge"]).run()

    assert at.radio(key=_key(at, "pmode_frequency"))
    assert at.radio(key=_key(at, "base_frequency"))
    config = at.session_state["_project"].models["frequency"]
    assert config.family == "gamma"
    assert config.offset == "VehAge"
    assert config.divide_target_by_weight is False
    assert config.predictors == ["DrivAge"]

    assert at.selectbox(key=_key(at, "fam_frequency")).value == "gamma"
    assert at.selectbox(key=_key(at, "off_frequency")).value == "VehAge"


def test_second_fitted_model_is_immediately_available_in_diagnostics(
    two_model_project: str,
):
    """Diagnostics distinguishes a definition from a usable fitted challenger.

    A second model must not appear as a comparison choice until its fit exists;
    once fitted it is available immediately and after its persisted run is
    re-resolved.  Keep frequency selected throughout: creating or fitting v2
    must not silently change the current diagnostic model.
    """
    at = AppTest.from_string(_two_model_script(two_model_project), default_timeout=180)
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]

    at.text_input(key=_key(at, "model_new_name")).set_value("frequency").run()
    _button(at, "Create").click().run()
    at.radio(key=_key(at, "pmode_frequency")).set_value("fixed").run()
    at.number_input(key=_key(at, "alpha_frequency")).set_value(0.002).run()
    at.button(key=_key(at, "fit_frequency")).click().run()
    assert not at.exception, [exception.value for exception in at.exception]

    # Creating v2 is not enough to compare it: state makes that limitation
    # visible instead of quietly pretending the new definition is a model run.
    at.text_input(key=_key(at, "model_new_name")).set_value("v2").run()
    _button(at, "Create").click().run()
    at.session_state["_page"] = "diagnostics"
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    assert at.selectbox(key=_key(at, "diag_run")).value == "frequency"
    challenger = at.selectbox(key=_key(at, "diag_chal_None"))
    assert challenger.options == ["(none)"]
    assert any(
        "v2 is defined but not fitted" in caption.value for caption in at.caption
    )

    # Finish the same model, then it becomes an immediate challenger without
    # replacing the selected current model.
    at.session_state["_page"] = "model"
    at.run()
    assert at.selectbox(key=_key(at, "model_select")).value == "v2"
    at.radio(key=_key(at, "pmode_v2")).set_value("fixed").run()
    at.number_input(key=_key(at, "alpha_v2")).set_value(0.002).run()
    at.button(key=_key(at, "fit_v2")).click().run()
    assert not at.exception, [exception.value for exception in at.exception]

    assert set(at.session_state["runs"]) == {"frequency", "v2"}
    at.session_state["_page"] = "diagnostics"
    at.run()
    assert not at.exception, [exception.value for exception in at.exception]
    selector = at.selectbox(key=_key(at, "diag_run"))
    assert set(selector.options) == {"frequency", "v2"}
    selector.set_value("frequency").run()
    assert at.selectbox(key=_key(at, "diag_run")).value == "frequency"
    challenger = at.selectbox(key=_key(at, "diag_chal_None"))
    assert set(challenger.options) == {"(none)", "v2"}
    challenger.set_value("v2").run()
    assert any("same predictions" in message.value for message in at.info), [
        message.value for message in at.info
    ]

    # Navigation/recovery may leave no in-memory runs; Diagnostics must resolve
    # both successful fits from the normal persisted-run cache without losing a
    # still-valid page-level selection.
    at.session_state["runs"] = {}
    at.run()
    selector = at.selectbox(key=_key(at, "diag_run"))
    assert set(selector.options) == {"frequency", "v2"}
    assert selector.value == "frequency"
    assert at.selectbox(key=_key(at, "diag_chal_None")).value == "v2"
