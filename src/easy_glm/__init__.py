"""easy_glm — insurance pricing with LASSO-regularised GLMs.

API layers
----------
**Recommended — full pipeline**

    eglm = EasyGLM.fit(data, ..., train_test_col="traintest")

Runs blueprint → prepare → :func:`fit_lasso_glm` → rate tables →
:class:`~easy_glm.engine.RateModel` in one call. Use :meth:`EasyGLM.predict`
for GLM predictions on raw data, or ``eglm.rate_model`` for portable
lookup-table scoring (``.easyglm`` export and the relativity editor).

**Advanced — step-by-step building blocks**

Use when you need control between stages (custom blueprint, reuse prepared
data for several fits, inspect intermediate tables):

1. :func:`generate_blueprint`
2. :func:`prepare_data`
3. :func:`fit_lasso_glm` — fits glum on **prepared** data only (step 3)
4. :func:`generate_all_ratetables` or :func:`ratetable`
5. :class:`~easy_glm.engine.RateModel.from_glm_model` or ``from_rate_tables``

:func:`fit_lasso_glm` is **not** an alternative to :meth:`EasyGLM.fit`; the
latter calls the former internally after blueprint and preparation.

**Scoring**

* :class:`~easy_glm.engine.RateModel` — production scoring from relativities
* :func:`predict_with_model` — predictions from a fitted glum model on
  prepared feature matrices
"""

from .core.all_ratetables import generate_all_ratetables
from .core.blueprint import generate_blueprint
from .core.data import load_external_dataframe
from .core.easyglm import EasyGLM
from .core.model import fit_lasso_glm, predict_with_model
from .core.plots import plot_all_ratetables
from .core.prepare import prepare_data
from .core.ratetable import ratetable
from .core.transforms import lump_fun, lump_rare_levels_pl, o_matrix

__all__ = [
    # High-level pipeline (start here)
    "EasyGLM",
    "load_external_dataframe",
    # Step-by-step building blocks (advanced)
    "generate_blueprint",
    "prepare_data",
    "fit_lasso_glm",
    "predict_with_model",
    "ratetable",
    "generate_all_ratetables",
    "plot_all_ratetables",
    # SQL transform helpers (low-level)
    "o_matrix",
    "lump_fun",
    "lump_rare_levels_pl",
]
