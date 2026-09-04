"""easy_glm — insurance pricing with LASSO-regularised GLMs.

API layers
----------
**Recommended — full pipeline**

    eglm = EasyGLM.fit(data, target=..., model_type="Poisson", predictors=[...],
                       weight_col="Exposure", divide_target_by_weight=True, cv=5)

Runs :class:`DesignSpec` -> :func:`fit_glm` -> :func:`rate_tables` ->
:class:`~easy_glm.engine.RateModel` in one call. ``eglm.predict`` gives GLM
predictions on raw data; ``eglm.rate_model`` is the portable lookup-table
scorer (``.easyglm`` export, relativity editor) and reproduces the GLM exactly.

**Building blocks**

1. :class:`DesignSpec` — how each predictor becomes features (step knots for
   numerics, one-hot with an ``Other`` bucket for categoricals). Build it from
   training data with :meth:`DesignSpec.from_data` or by hand; JSON round-trip.
2. :func:`fit_glm` — penalised glum fit on ``spec.build(train)``; returns a
   :class:`GLMFit` with ``predict`` / ``coef_table``.
3. :func:`rate_tables` / :func:`to_rate_model` — exact relativities and base
   rate read off the coefficients.

With interactions, :func:`fit_two_stage` fits the mains first and the cells on
top of them (:class:`TwoStageFit`), so adding an interaction never moves a
main-effect table or the base rate and every cell is a pure adjustment.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .app import launch as launch_workbench
from .engine.rate_model import RateModel

try:
    __version__ = _pkg_version("easy_glm")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "0.0.0"

from .core import (
    CategoricalEncoder,
    DesignSpec,
    EasyGLM,
    GLMFit,
    InteractionEncoder,
    LinearEncoder,
    StepEncoder,
    TwoStageFit,
    add_train_test_split,
    base_rate,
    fit_glm,
    fit_two_stage,
    load_external_dataframe,
    plot_all_ratetables,
    rate_tables,
    to_rate_model,
    validate_train_test_column,
    write_rate_tables_xlsx,
)

__all__ = [
    # High-level pipeline (start here)
    "EasyGLM",
    "RateModel",
    "load_external_dataframe",
    "add_train_test_split",
    "launch_workbench",
    # Building blocks
    "DesignSpec",
    "StepEncoder",
    "CategoricalEncoder",
    "InteractionEncoder",
    "LinearEncoder",
    "GLMFit",
    "TwoStageFit",
    "fit_glm",
    "fit_two_stage",
    "rate_tables",
    "base_rate",
    "to_rate_model",
    "write_rate_tables_xlsx",
    "plot_all_ratetables",
    "validate_train_test_column",
]
