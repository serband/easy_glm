from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from easy_glm.engine.models import (
    FromToRow,
    ModelMetadata,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.engine.rate_model import RateModel
from easy_glm.pricing.search import find_interactions, find_missing_factors
from easy_glm.pricing.session import _data_fingerprint
from easy_glm.pricing.views import ae, relativities
from easy_glm.workflow.pair_stages import PairStageArtifact
from easy_glm.workflow.project import ModelConfig, Project, VariableDesign


class ExampleModel:
    def __init__(self) -> None:
        self.name = "candidate"
        self._data = pl.DataFrame(
            {
                "claims": [0.0, 1.0, 0.0, 2.0, 1.0, 0.0],
                "exposure": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
                "split": [1, 1, 1, 1, 0, 0],
                "x": [0.0, 0.5, 1.5, 2.0, 0.2, 1.8],
                "z": ["a", "b", "a", "b", "a", "b"],
                "omitted": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                "numeric_omitted": [0.0, 0.5, 1.5, 2.5, 0.2, 2.0],
            }
        )
        fingerprint = _data_fingerprint(self._data, None)
        self._session = SimpleNamespace(_data=self._data, _data_fingerprint=fingerprint)
        x_axis = VariableConfig(
            "numeric",
            [
                FromToRow(None, 1.0, 1.0),
                FromToRow(1.0, None, 1.0),
                FromToRow(None, None, 1.0),
            ],
        )
        z_axis = VariableConfig(
            "categorical",
            [FromToRow("a", "a", 1.0), FromToRow(None, None, 1.0)],
        )
        pair = PairTableConfig(
            "pair-1",
            ("x", "z"),
            (x_axis, z_axis),
            [
                PairCellRow(0, 0, 1.0, 2, 3.0, 0.375, "below support floor"),
                PairCellRow(0, 1, 1.1, 1, 2.0, 0.25),
                PairCellRow(1, 0, 1.2, 1, 1.0, 0.125),
                PairCellRow(1, 1, 0.8, 2, 2.0, 0.25),
            ],
        )
        metadata = ModelMetadata(
            target="claims",
            weight_col="exposure",
            exposure_col="exposure",
            train_test_col="split",
            divide_target_by_weight=True,
            model_type="poisson",
            link="log",
        )
        scorer = RateModel(
            0.2,
            {"x": x_axis},
            metadata=metadata,
            pair_tables=[pair],
        )
        config = ModelConfig(
            family="poisson",
            target="claims",
            weight="exposure",
            divide_target_by_weight=True,
            predictors=["x"],
        )
        project = Project(name="views")
        project.data.roles = {
            "claims": "target",
            "exposure": "weight",
            "split": "split",
            "x": "predictor",
        }
        project.data.split.column = "split"
        # A numeric code deliberately reviewed as categorical. Searches must
        # use this saved design rather than infer a numeric quantile treatment.
        project.design.variables["omitted"] = VariableDesign(
            kind="categorical", levels=["0.0", "1.0"]
        )
        project.design.variables["numeric_omitted"] = VariableDesign(
            kind="step", knots=[1.0, 2.0]
        )
        project.models["candidate"] = config
        self._project = project
        self._run = SimpleNamespace(
            config=config,
            rate_model=scorer,
            pair_stages=[
                PairStageArtifact(
                    "pair-1",
                    ("x", "z"),
                    pair,
                    None,
                    "test-prefix",
                    prefix_cv_loss=1.2,
                    table_cv_loss=1.1,
                )
            ],
            metrics={},
            created_at="",
            train_rows=4,
            holdout_rows=2,
            dropped_predictors=[],
        )
        self._history = []
        self._evidence = {"data_fingerprint": fingerprint}
        self._frozen_stages = ["pair-1"]

    def _frame(self, subset: str = "train") -> pl.DataFrame:
        value = 1 if subset == "train" else 0
        return self._data.filter(pl.col("split") == value)


def test_ae_and_relativity_views_use_complete_scorer_and_exact_pair_axes() -> None:
    model = ExampleModel()
    one_way = ae(model, "x")
    expected = (
        model._run.rate_model.predict(model._frame("train"), exposure_col=None)
        * model._frame("train")["exposure"].to_numpy()
    )
    assert one_way.table["expected"].sum() == np.sum(expected)
    assert one_way.figure is not None
    assert "<br>" in one_way.figure.layout.title.text
    assert one_way.figure.layout.title.x == 0.02
    assert one_way.figure.layout.title.y == 0.94
    assert one_way.figure.layout.legend.y > 1
    assert one_way.figure.layout.margin.t >= 120

    pair = ae(model, "x", "z")
    assert pair.table.height == 6
    assert pair.table["exposure"].sum() == 6.0
    assert pair.figure is not None

    deployed = relativities(model, "x", "z")
    assert deployed.table.height == 6
    assert set(deployed.table["support_source"]) == {"fitting_weight"}
    assert deployed.figure is not None
    assert "<br>" in deployed.figure.layout.title.text
    assert deployed.figure.layout.title.x == 0.02
    assert deployed.figure.layout.title.y == 0.94
    assert deployed.figure.layout.margin.t >= 100
    assert "plotly" in deployed._repr_html_().lower()
    assert any("*" in label for row in deployed.figure.data[0].text for label in row)
    reversed_pair = relativities(model, "z", "x")
    assert reversed_pair.table["parent_a"].unique().to_list() == ["z"]
    assert reversed_pair.table["parent_b"].unique().to_list() == ["x"]
    assert ae(model, "z", "x").table.height == 6

    model.name = "A very long reviewed pricing model name " * 4
    bounded = ae(model, "x")
    assert "…" in bounded.figure.layout.title.text


def test_unassigned_factor_views_and_searches_stay_on_training_rows() -> None:
    model = ExampleModel()
    omitted = ae(model, "omitted")
    assert omitted.table["exposure"].sum() == 6.0
    assert omitted.table["label"].to_list() == ["0.0", "1.0", "Other"]
    numeric = ae(model, "numeric_omitted")
    assert numeric.table.height == 4

    factors = find_missing_factors(model, min_expected=0.01)
    assert "omitted" in factors.table["variable"].to_list()
    assert "claims" not in factors.table["variable"].to_list()
    assert "exposure" not in factors.table["variable"].to_list()

    pairs = find_interactions(model, ["x", "z", "omitted"], min_expected=0.01)
    assert "x × z" not in pairs.table["pair"].to_list()
