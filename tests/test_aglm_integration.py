"""Independent integration checks for the opt-in AGLM numeric bases."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, EasyGLM, fit_glm, to_rate_model
from easy_glm.core.design import StepEncoder
from easy_glm.core.excel import rate_model_tables
from easy_glm.core.fit import design_row_key
from easy_glm.engine.rate_model import RateModel
from easy_glm.pricing import PricingSession
from easy_glm.workflow import prepare, run_model, to_script
from easy_glm.workflow.project import Project, VariableDesign


def _training_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.5, 3.0],
        }
    )


@pytest.mark.parametrize("kind", ["aglm", "aglm_l"])
@pytest.mark.parametrize("extrapolation", ["extend", "flat"])
def test_aglm_compiled_scorer_and_table_roundtrip_keep_tail_policy(
    kind: str, extrapolation: str
) -> None:
    train = _training_data()
    spec = DesignSpec.from_data(
        train,
        ["x"],
        numeric_kinds={"x": kind},
        breaks={"x": [0.0, 2.0, 5.0]},
        extrapolation={"x": extrapolation},
    )
    fitted = fit_glm(train, spec, "y", family="gaussian", alpha=0.01)
    scorer = to_rate_model(fitted)
    values = pl.DataFrame({"x": [-3.0, 0.0, 2.0, 5.0, 10.0]})

    np.testing.assert_allclose(
        fitted.predict(values), scorer.predict(values), atol=1e-12, rtol=0.0
    )

    tables = rate_model_tables(scorer)
    assert tables["x"]["curve_kind"].unique().to_list() == [kind]
    assert tables["x"]["extrapolation"].unique().to_list() == [extrapolation]
    restored = RateModel.from_rate_tables(tables, scorer.base_rate)
    assert restored.variables["x"].curve_kind == kind
    assert restored.variables["x"].extrapolation == extrapolation
    np.testing.assert_allclose(
        scorer.predict(values), restored.predict(values), atol=1e-12, rtol=0.0
    )
    with pytest.raises(ValueError, match="infinite"):
        restored.predict(pl.DataFrame({"x": [float("inf")]}))


def test_aglm_grid_uses_training_values_and_aggregation_keeps_real_x() -> None:
    train = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    spec = DesignSpec.from_data(
        train,
        ["x"],
        numeric_kinds={"x": "aglm"},
        n_breaks=3,
    )
    encoder = spec["x"]
    assert max(encoder.breaks) == 3.0

    # Both values are in the same interval, but their raw-x basis columns differ.
    rows = pl.DataFrame({"x": [0.2, 0.8]})
    keys = design_row_key(spec, rows)
    assert keys["c:x"].n_unique() == 1
    assert keys["x:x"].n_unique() == 2


@pytest.mark.parametrize("kind", ["aglm", "aglm_l"])
@pytest.mark.parametrize("extrapolation", ["extend", "flat"])
def test_aglm_without_null_indicator_never_exports_or_scores_a_null_row(
    kind: str, extrapolation: str
) -> None:
    train = _training_data()
    spec = DesignSpec.from_data(
        train,
        ["x"],
        numeric_kinds={"x": kind},
        breaks={"x": [0.0, 2.0, 5.0]},
        extrapolation={"x": extrapolation},
        null_indicator=False,
    )
    encoder = spec["x"]
    assert encoder.rows() == encoder.bins()
    fitted = fit_glm(train, spec, "y", family="gaussian", alpha=0.01)
    assert len(fitted.row_exposure["x"]) == len(encoder.rows())
    scorer = to_rate_model(fitted)
    assert all(
        not (row.from_ is None and row.to_ is None)
        for row in scorer.variables["x"].table
    )
    assert scorer.variables["x"].null_relativity is None
    missing = pl.DataFrame({"x": [None]})
    with pytest.raises(ValueError, match="null_indicator=False"):
        encoder.row_index(missing["x"])
    with pytest.raises(ValueError, match="null_indicator=False"):
        fitted.predict(missing)
    with pytest.raises(ValueError, match="NaN|band"):
        scorer.predict(missing)


def test_legacy_variable_design_json_roundtrips_without_aglm_null_keys() -> None:
    project = Project()
    project.design.variables["age"] = VariableDesign(kind="step", n_bins=8)
    legacy = project.to_dict()
    for key in ("breaks", "n_breaks", "extrapolation"):
        assert key not in legacy["design"]["variables"]["age"]

    # A JSON document written before AGLM must retain the same wire form.
    restored = Project.from_dict(json.loads(json.dumps(legacy)))
    assert restored.to_dict() == legacy


def test_public_easyglm_and_pricing_session_routes_build_aglm_from_train() -> None:
    data = pl.DataFrame(
        {
            "policy": list(range(12)),
            "split": [1] * 8 + [0] * 4,
            "x": [float(value) for value in range(8)] + [100.0] * 4,
            "claims": [0, 0, 1, 0, 1, 1, 2, 2, 3, 3, 3, 4],
            "exposure": [1.0] * 12,
        }
    )
    fitted = EasyGLM.fit(
        data,
        "claims",
        "poisson",
        ["x"],
        train_test_col="split",
        divide_target_by_weight=True,
        weight_col="exposure",
        alpha=0.01,
        numeric_kinds={"x": "aglm"},
        n_breaks=3,
    )
    assert max(fitted.spec["x"].breaks) == 7.0

    work = PricingSession(
        data,
        claims="claims",
        exposure="exposure",
        id="policy",
        split="split",
    )
    preview = work.numeric("x", kind="aglm_l", n_breaks=3, extrapolation="flat")
    assert preview.table["exposure"].sum() == 8.0
    model = work.fit_glm("curve", factors=["x"], alpha=0.01)
    assert model._run.spec["x"].kind == "aglm_l"
    assert model._run.spec["x"].extrapolation == "flat"


def test_pricing_numeric_validation_is_atomic_and_does_not_mutate_a_saved_fit() -> None:
    data = _training_data().with_columns(
        pl.Series("policy", list(range(8))),
        pl.Series("claims", [0, 0, 1, 0, 1, 1, 2, 2]),
        pl.Series("exposure", [1.0] * 8),
    )
    work = PricingSession(
        data,
        claims="claims",
        exposure="exposure",
        id="policy",
        seed=3,
    )
    work.bands("x", number=3)
    original = work.fit_glm("bands", factors=["x"], alpha=0.01)
    before = work.settings()

    with pytest.raises(ValueError, match="increasing"):
        work.numeric("x", breaks=[0.0, 0.0, 5.0])
    assert work.settings() == before

    work.numeric("x", breaks=[0.0, 2.0, 7.0])
    assert isinstance(original._run.spec["x"], StepEncoder)
    assert work._project.design.variables["x"].kind == "aglm"


def test_project_rejects_invalid_aglm_schema_and_roundtrips_valid_controls() -> None:
    project = Project()
    project.design.variables["x"] = VariableDesign(
        kind="aglm", breaks=[0.0, 2.0, 5.0], extrapolation="flat"
    )
    encoded = project.to_dict()
    restored = Project.from_dict(encoded)
    assert restored.to_dict() == encoded

    project.design.variables["x"] = VariableDesign(kind="aglm", breaks=[0.0, 0.0])
    assert any("increasing endpoints" in message for message in project.validate())
    project.design.variables["x"] = VariableDesign(kind="aglm_l", n_bins=8)
    assert any("breaks/n_breaks" in message for message in project.validate())


def test_aglm_project_exported_script_rebuilds_the_reviewed_curve(tmp_path) -> None:
    """The public project export retains the exact AGLM class, grid and tails."""
    data = _training_data().with_columns(
        pl.Series("policy", list(range(8))),
        pl.Series("split", [1, 1, 1, 1, 1, 1, 0, 0]),
        pl.Series("exposure", [1.0] * 8),
    )
    source = tmp_path / "curve.parquet"
    data.write_parquet(source)
    project = Project(name="AGLM export")
    project.data.source.type = "parquet"
    project.data.source.path = str(source)
    project.data.roles = {
        "policy": "id",
        "split": "split",
        "x": "predictor",
        "y": "target",
        "exposure": "weight",
    }
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.data.split.train_value = 1
    project.data.split.holdout_value = 0
    project.design.variables["x"] = VariableDesign(
        kind="aglm_l", breaks=[0.0, 2.0, 5.0], extrapolation="flat"
    )
    project.new_model("curve", family="gaussian", predictors=["x"])
    project.models["curve"].penalty.alpha = 0.01

    prepared = prepare(project)
    fitted = run_model(project, prepared, "curve")
    script = tmp_path / "rebuild.py"
    script.write_text(to_script(project, "curve", run=fitted, output_prefix="rebuilt"))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    rebuilt = RateModel.from_json(tmp_path / "rebuilt.easyglm")
    assert rebuilt.variables["x"].curve_kind == "aglm_l"
    assert rebuilt.variables["x"].extrapolation == "flat"
    np.testing.assert_allclose(
        fitted.predict(prepared),
        rebuilt.predict(prepared, exposure_col=None),
        atol=1e-12,
        rtol=0.0,
    )
