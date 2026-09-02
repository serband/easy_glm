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

**Legacy (deprecated, removed in 0.4)**

``generate_blueprint``, ``prepare_data`` (needs the ``legacy`` extra for
DuckDB), ``fit_lasso_glm``, ``ratetable``, ``generate_all_ratetables``.
"""

from .core import (
    CategoricalEncoder,
    DesignSpec,
    EasyGLM,
    GLMFit,
    StepEncoder,
    base_rate,
    fit_glm,
    fit_lasso_glm,
    generate_all_ratetables,
    generate_blueprint,
    load_external_dataframe,
    plot_all_ratetables,
    predict_with_model,
    prepare_data,
    rate_tables,
    ratetable,
    to_rate_model,
    write_rate_tables_xlsx,
)

__all__ = [
    # High-level pipeline (start here)
    "EasyGLM",
    "load_external_dataframe",
    # Building blocks
    "DesignSpec",
    "StepEncoder",
    "CategoricalEncoder",
    "GLMFit",
    "fit_glm",
    "rate_tables",
    "base_rate",
    "to_rate_model",
    "write_rate_tables_xlsx",
    "plot_all_ratetables",
    # Legacy (deprecated)
    "generate_blueprint",
    "prepare_data",
    "fit_lasso_glm",
    "predict_with_model",
    "ratetable",
    "generate_all_ratetables",
]
