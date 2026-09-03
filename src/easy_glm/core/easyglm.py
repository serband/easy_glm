from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from easy_glm.engine.rate_model import RateModel

from .design import DesignSpec, StepEncoder
from .fit import GLMFit, TwoStageFit, fit_glm
from .split import TRAIN_FLAG, validate_train_test_column
from .tables import rate_tables, to_rate_model


class EasyGLM:
    """End-to-end insurance GLM pipeline (recommended entry point).

    :meth:`fit` runs design spec -> penalised GLM -> exact rate tables ->
    :class:`~easy_glm.engine.RateModel` in one call. The building blocks are
    :class:`~easy_glm.DesignSpec`, :func:`~easy_glm.fit_glm`,
    :func:`~easy_glm.rate_tables` and :func:`~easy_glm.to_rate_model`.
    """

    def __init__(
        self,
        glm: GLMFit,
        rate_model: RateModel,
        tables: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self.glm = glm
        self.rate_model = rate_model
        self._tables = tables if tables is not None else rate_tables(glm)

    # ------------------------------------------------------------------ fit
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
        alpha: float | None = None,
        cv: int | None = None,
        l1_ratio: float | list[float] = 1.0,
        monotone: Mapping[str, str] | None = None,
        n_bins: int = 20,
        min_level_share: float = 0.0025,
        knots: dict[str, list[float]] | None = None,
        categorical: list[str] | None = None,
        null_indicator: bool = True,
        base_rate: float | None = None,
        base: str = "modal",
        exposure_col: str | None = None,
        use_cv: bool | None = None,
        cv_params: dict | None = None,
        **glum_kwargs: Any,
    ) -> EasyGLM:
        """Fit on training rows only (``train_test_col == 1``).

        Parameters
        ----------
        data : pl.DataFrame
            Full dataset (train + holdout) with a split column: **1 = train**,
            **0 = holdout**.
        target, model_type, predictors
            Response column, family (``"Poisson"``, ``"Gamma"``, ``"Gaussian"``,
            ``"Tweedie"``, ``"Binomial"``) and raw predictor columns.
        weight_col, divide_target_by_weight
            Exposure/premium weights; divide the target by them to model a
            rate (e.g. frequency = counts / exposure).
        alpha, cv, l1_ratio
            Penalty strength, or the number of CV folds used to choose it
            (default ``cv=5`` when neither is given). ``l1_ratio=1`` is lasso.
        monotone
            ``{"DrivAge": "decreasing", ...}`` sign constraints on numeric
            predictors.
        n_bins, min_level_share, knots, categorical, null_indicator
            Design options, see :meth:`DesignSpec.from_data`.
        base_rate
            Override the base rate. By default it is calibrated exactly so that
            ``rate_model.predict(...)`` reproduces the GLM.
        base
            ``"modal"`` (relativity 1.0 on the most exposed bin) or
            ``"reference"`` (lowest bin / reference level).
        exposure_col
            Column the :class:`RateModel` multiplies predictions by when scoring
            (defaults to ``weight_col`` if the target was divided by it).
        use_cv, cv_params
            Deprecated aliases: ``use_cv=False`` requires ``alpha``;
            ``cv_params`` keys ``n_alphas``, ``l1_ratio``, ``min_alpha_ratio``
            and ``cv`` are mapped, the rest go to glum.
        glum_kwargs
            Forwarded to the glum estimator (``max_iter``, ``P1`` ...).
        """
        validate_train_test_column(data, train_test_col)
        train = data.filter(pl.col(train_test_col) == TRAIN_FLAG)

        # -- legacy knobs -------------------------------------------------
        if cv_params:
            cv_params = dict(cv_params)
            l1_ratio = cv_params.pop("l1_ratio", l1_ratio)
            for key in ("n_alphas", "min_alpha_ratio"):
                if key in cv_params:
                    glum_kwargs[key] = cv_params.pop(key)
            cv = cv_params.pop("cv", cv)
            glum_kwargs.update(cv_params)
        if use_cv is False and alpha is None:
            raise ValueError(
                "use_cv=False requires alpha=... (the old behaviour silently "
                "returned an almost unregularised model)."
            )
        if cv is None and alpha is None:
            cv = 5

        spec = DesignSpec.from_data(
            train,
            predictors,
            n_bins=n_bins,
            min_level_share=min_level_share,
            knots=knots,
            categorical=categorical,
            null_indicator=null_indicator,
            weight_col=weight_col,
        )
        glm = fit_glm(
            train,
            spec,
            target,
            family=model_type,
            weight_col=weight_col,
            divide_target_by_weight=divide_target_by_weight,
            alpha=alpha,
            cv=cv,
            l1_ratio=l1_ratio,
            monotone=monotone,
            **glum_kwargs,
        )
        if exposure_col is None and divide_target_by_weight:
            exposure_col = weight_col
        rm = to_rate_model(
            glm,
            base=base,  # type: ignore[arg-type]
            base_rate_override=base_rate,
            exposure_col=exposure_col,
            train_test_col=train_test_col,
            model_type=model_type,
        )
        return cls(glm, rm, rate_tables(glm, base=base))  # type: ignore[arg-type]

    # ----------------------------------------------------------- accessors
    @property
    def spec(self) -> DesignSpec:
        return self.glm.spec

    @property
    def model(self):
        """The underlying glum estimator."""
        return self.glm.model

    @property
    def predictors(self) -> list[str]:
        return self.spec.variables

    @property
    def relativities(self) -> dict[str, pl.DataFrame]:
        return dict(self._tables)

    @property
    def base_rate(self) -> float:
        return self.rate_model.base_rate

    @property
    def blueprint(self) -> dict[str, list]:
        """Legacy view of the spec: knots for numeric, levels for categorical."""
        return {
            var: list(enc.knots) if isinstance(enc, StepEncoder) else list(enc.levels)
            for var, enc in self.spec.encoders.items()
        }

    def coef_table(self, *, drop_zero: bool = False) -> pl.DataFrame:
        return self.glm.coef_table(drop_zero=drop_zero)

    def predict(self, raw_data: pl.DataFrame) -> pl.Series:
        """GLM predictions on raw data (per unit weight if the target was
        divided by the weight)."""
        return pl.Series("prediction", self.glm.predict(raw_data))

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Write the spec, the fitted estimator(s), the RateModel and the tables.

        A two-stage fit (mains frozen, interaction cells on top) writes **both**
        glum estimators and is rebuilt as a :class:`~easy_glm.TwoStageFit` by
        :meth:`load`; stage 1's estimator alone could not score the composed
        spec."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.spec.to_json(path / "spec.json")
        joblib.dump(self.glm.model, str(path / "glm_model.joblib"))
        two_stage = isinstance(self.glm, TwoStageFit)
        if two_stage:
            joblib.dump(self.glm.stage2.model, str(path / "glm_model_stage2.joblib"))
        self.rate_model.to_json(str(path / "rate_model.json"))
        tables_dir = path / "rate_tables"
        tables_dir.mkdir(exist_ok=True)
        for name, tbl in self._tables.items():
            tbl.write_parquet(str(tables_dir / f"{name}.parquet"))
        config = {
            "version": 3,
            "stages": 2 if two_stage else 1,
            "family": self.glm.family,
            "link": self.glm.link,
            "target": self.glm.target,
            "weight_col": self.glm.weight_col,
            "offset_col": self.glm.offset_col,
            "divide_target_by_weight": self.glm.divide_target_by_weight,
            "monotone": self.glm.monotone,
            "modal_bins": self.glm.modal_bins,
            "n_train_rows": self.glm.n_train_rows,
            "predictors": self.predictors,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EasyGLM:
        path = Path(path)
        if not (path / "spec.json").exists():
            raise FileNotFoundError(
                f"{path} has no spec.json. Models saved by easy_glm < 0.3 "
                "(blueprint.json) cannot be loaded; refit with EasyGLM.fit."
            )
        config = json.loads((path / "config.json").read_text())
        spec = DesignSpec.from_json(path / "spec.json")
        common: dict[str, Any] = {
            "family": config["family"],
            "link": config["link"],
            "target": config["target"],
            "weight_col": config.get("weight_col"),
            "offset_col": config.get("offset_col"),
            "divide_target_by_weight": config.get("divide_target_by_weight", False),
            "monotone": config.get("monotone", {}),
            "modal_bins": config.get("modal_bins", {}),
            "n_train_rows": config.get("n_train_rows", 0),
        }
        glm: GLMFit
        if config.get("stages", 1) == 2:
            # the two stages were saved separately; rebuild the pair, whose
            # composed spec is the one on disk (mains then cells)
            glm = TwoStageFit(
                GLMFit(
                    spec=spec.main_effects_spec(),
                    model=joblib.load(str(path / "glm_model.joblib")),
                    **common,
                ),
                GLMFit(
                    spec=spec.interactions_spec(),
                    model=joblib.load(str(path / "glm_model_stage2.joblib")),
                    **{**common, "offset_col": None, "monotone": {}, "modal_bins": {}},
                ),
            )
        else:
            glm = GLMFit(
                spec=spec,
                model=joblib.load(str(path / "glm_model.joblib")),
                **common,
            )
        rm = RateModel.from_json(str(path / "rate_model.json"))
        tables: dict[str, pl.DataFrame] = {}
        tables_dir = path / "rate_tables"
        if tables_dir.exists():
            for f in sorted(tables_dir.glob("*.parquet")):
                tables[f.stem] = pl.read_parquet(str(f))
        return cls(glm, rm, tables or None)

    def to_excel(self, path: str | Path) -> Path:
        """Write the **fitted** rate tables to an ``.xlsx`` workbook: a ``Summary``
        sheet, an ``Index``, the ``Coefficients`` table and one sheet per variable
        (``from`` / ``to`` / ``label`` / ``coef`` / ``relativity`` / ``is_base``).
        Manual adjustments are *not* reflected here; use ``rate_model.to_excel``."""
        from .excel import write_rate_tables_xlsx

        summary = {
            "tables": "fitted (pre-adjustment) relativities — for the tables the "
            "scorer uses, including manual adjustments, use RateModel.to_excel",
            **self.summary(),
        }
        return write_rate_tables_xlsx(
            self.relativities,
            path,
            summary=summary,
            coef_table=self.coef_table(),
        )

    # ------------------------------------------------------------- reporting
    def summary(self) -> dict[str, Any]:
        return {
            "model_type": self.rate_model.metadata.model_type,
            "target": self.rate_model.metadata.target,
            "weight_col": self.rate_model.metadata.weight_col,
            "offset_col": self.glm.offset_col,
            "train_test_col": self.rate_model.metadata.train_test_col,
            "predictors": self.predictors,
            "base_rate": self.rate_model.base_rate,
            "alpha": self.glm.alpha,
            "num_features": len(self.glm.coef),
            "num_nonzero": int((self.glm.coef != 0).sum()),
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
            f"predictors={s['predictors']}, alpha={s['alpha']:.4g}, "
            f"base_rate={s['base_rate']:.6g})"
        )
