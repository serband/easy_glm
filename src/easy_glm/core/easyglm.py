from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

from easy_glm.engine.rate_model import RateModel

from .all_ratetables import generate_all_ratetables
from .blueprint import generate_blueprint
from .model import (
    TRAIN_FLAG,
    fit_lasso_glm,
    predict_with_model,
    validate_train_test_column,
)
from .prepare import prepare_data


class EasyGLM:
    """End-to-end insurance GLM pipeline (recommended entry point).

    Use :meth:`fit` to run blueprint → prepare → LASSO GLM → rate tables →
    :class:`~easy_glm.engine.RateModel`. For manual control over each stage,
    use the step functions documented in :mod:`easy_glm`.
    """

    def __init__(
        self,
        blueprint: dict[str, Any],
        model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV,
        rate_model: RateModel,
        predictors: list[str],
        all_tables: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self.blueprint = blueprint
        self.model = model
        self.rate_model = rate_model
        self.predictors = predictors
        self._all_tables = all_tables or {}

    @classmethod
    def fit(
        cls,
        data: pl.DataFrame,
        target: str,
        model_type: str,
        predictors: list[str],
        *,
        weight_col: str | None = None,
        train_test_col: str = "traintest",
        divide_target_by_weight: bool = False,
        use_cv: bool = True,
        cv_params: dict | None = None,
        base_rate: float = 1.0,
        exposure_col: str | None = None,
        random_seed: int = 42,
    ) -> EasyGLM:
        """Fit on training rows only (``train_test_col == 1``).

        Args:
            data: Full dataset (train + holdout). Must include a split column.
            train_test_col: Name of the train/holdout column in ``data``
                (default ``"traintest"``). **1 = train**, **0 = holdout**.
        """
        validate_train_test_column(data, train_test_col)

        train_data = data.filter(pl.col(train_test_col) == TRAIN_FLAG)

        blueprint = generate_blueprint(train_data)

        additional_cols = [target, train_test_col]
        if weight_col:
            additional_cols.append(weight_col)

        prepped = prepare_data(
            df=data,
            modelling_variables=predictors,
            additional_columns=additional_cols,
            formats=blueprint,
            traintest_column=None,
            table_name="line_prepped",
        )

        model = fit_lasso_glm(
            dataframe=prepped,
            target=target,
            train_test_col=train_test_col,
            model_type=model_type,
            weight_col=weight_col,
            divide_target_by_weight=divide_target_by_weight,
            use_cv=use_cv,
            cv_params=cv_params,
        )

        effective_exposure = exposure_col if exposure_col is not None else weight_col

        all_tables = generate_all_ratetables(
            model=model,
            dataset=data,
            predictor_variables=predictors,
            blueprint=blueprint,
            random_seed=random_seed,
        )

        rate_model = RateModel.from_rate_tables(
            all_tables=all_tables,
            blueprint=blueprint,
            base_rate=base_rate,
            model_type=model_type,
            target=target,
            weight_col=weight_col,
            exposure_col=effective_exposure,
            train_test_col=train_test_col,
            predictor_variables=predictors,
        )

        return cls(
            blueprint=blueprint,
            model=model,
            rate_model=rate_model,
            predictors=predictors,
            all_tables=all_tables,
        )

    def predict(self, raw_data: pl.DataFrame) -> pl.Series:
        """GLM predictions on raw data (prepares features using stored blueprint)."""
        prepped = prepare_data(
            df=raw_data,
            modelling_variables=self.predictors,
            formats=self.blueprint,
            table_name="line_prepped",
        )
        return predict_with_model(self.model, prepped, return_polars=True)

    @property
    def relativities(self) -> dict[str, pl.DataFrame]:
        return dict(self._all_tables)

    @property
    def base_rate(self) -> float:
        return self.rate_model.base_rate

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, str(path / "glm_model.joblib"))

        blueprint_copy = copy.deepcopy(self.blueprint)
        for k, v in blueprint_copy.items():
            if not isinstance(v, list):
                continue
            sanitised = []
            for item in v:
                if isinstance(item, float | int | str | bool | type(None)):
                    sanitised.append(item)
                elif isinstance(item, bytes):
                    sanitised.append(item.decode("utf-8", errors="replace"))
                else:
                    sanitised.append(str(item))
            blueprint_copy[k] = sanitised

        (path / "blueprint.json").write_text(
            json.dumps(blueprint_copy, indent=2, default=str)
        )

        self.rate_model.to_json(str(path / "rate_model.json"))

        tables_dir = path / "rate_tables"
        tables_dir.mkdir(exist_ok=True)
        for name, tbl in self._all_tables.items():
            tbl.write_parquet(str(tables_dir / f"{name}.parquet"))

        (path / "config.json").write_text(
            json.dumps({"predictors": self.predictors}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> EasyGLM:
        path = Path(path)
        model = joblib.load(str(path / "glm_model.joblib"))
        blueprint = json.loads((path / "blueprint.json").read_text())
        config = json.loads((path / "config.json").read_text())

        rate_model = RateModel.from_json(str(path / "rate_model.json"))

        all_tables: dict[str, pl.DataFrame] = {}
        tables_dir = path / "rate_tables"
        if tables_dir.exists():
            for parquet_file in tables_dir.glob("*.parquet"):
                name = parquet_file.stem
                all_tables[name] = pl.read_parquet(str(parquet_file))

        return cls(
            blueprint=blueprint,
            model=model,
            rate_model=rate_model,
            predictors=config["predictors"],
            all_tables=all_tables,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_type": self.rate_model.metadata.model_type,
            "target": self.rate_model.metadata.target,
            "weight_col": self.rate_model.metadata.weight_col,
            "train_test_col": self.rate_model.metadata.train_test_col,
            "predictors": self.predictors,
            "base_rate": self.rate_model.base_rate,
            "num_variables": len(self.rate_model.variables),
            "snapshots": len(self.rate_model.snapshots),
        }

    def launch_editor(self, data=None, test_data=None, port=8501, **kwargs):
        self.rate_model.launch_editor(
            data=data, test_data=test_data, port=port, **kwargs
        )

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"EasyGLM(model_type={s['model_type']!r}, target={s['target']!r}, "
            f"predictors={s['predictors']}, base_rate={s['base_rate']})"
        )
