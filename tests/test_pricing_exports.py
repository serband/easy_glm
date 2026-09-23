from __future__ import annotations

import json
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest
from test_pricing_views import ExampleModel

from easy_glm.pricing import PricingModel
from easy_glm.pricing.excel import export_excel
from easy_glm.pricing.persistence import save_model


def test_pricing_json_contains_exact_scorer_settings_and_no_policy_rows(
    tmp_path,
) -> None:
    model = ExampleModel()
    model._history = [{"action": "fit", "model": model.name}]
    model._evidence = {"review": "training only"}
    path = save_model(model, tmp_path / "candidate.json")
    raw = json.loads(path.read_text())
    assert raw["format"] == "easy-glm-pricing-model"
    assert raw["project"]["models"][model.name]["predictors"] == ["x"]
    assert raw["rate_model"]["pair_tables"][0]["stage_id"] == "pair-1"
    assert "claims" not in raw


def test_pricing_excel_contains_current_pairs_and_audit_sheets(tmp_path) -> None:
    model = ExampleModel()
    model._run.metrics = {"train": {"ae": 0.98}}
    model._evidence = {
        "pair decision": "reviewed",
        "validation": {"holdout": {"ae": 1.02, "rows": 2}},
    }
    path = export_excel(model, tmp_path / "candidate.xlsx")
    assert path.exists()
    with ZipFile(path) as workbook:
        names = workbook.read("xl/workbook.xml").decode()
    for sheet in (
        "Summary",
        "Pair stages",
        "Pair 1 cells",
        "Amendments",
        "Evidence",
        "Validation",
        "History",
        "Scoring rules",
    ):
        assert f'name="{sheet}"' in names
    validation = pl.read_excel(path, sheet_name="Validation")
    assert set(validation["subset"]) == {"train", "holdout"}
    assert (
        validation.filter(
            (pl.col("subset") == "holdout") & (pl.col("metric") == "ae")
        ).height
        == 1
    )


def test_loaded_model_scores_without_refit_and_attaches_analysis_data(tmp_path) -> None:
    model = ExampleModel()
    path = save_model(model, tmp_path / "candidate.json")
    expected = model._run.rate_model.predict(model._data, exposure_col=None)

    frozen = PricingModel.load(path)
    np.testing.assert_allclose(frozen.predict(model._data), expected)
    frozen.to_excel(tmp_path / "frozen.xlsx")

    attached = PricingModel.load(path, data=model._data)
    np.testing.assert_allclose(attached.predict(model._data), expected)
    assert attached.ae("x").table["exposure"].sum() == 6.0
    assert attached._run.pair_stages[0].table_cv_loss == 1.1
    assert attached._run.pair_stages[0].table.stage_id == "pair-1"

    changed = model._data.with_columns((pl.col("claims") + 1).alias("claims"))
    with pytest.raises(ValueError, match="does not match"):
        PricingModel.load(path, data=changed)
