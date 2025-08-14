import sys
from typing import Any

# Lazy import implementation
class _LazyImport:
    def __init__(self, module_name: str, attr_name: str):
        self.module_name = module_name
        self.attr_name = attr_name
        self._module = None

    def _load(self):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name, __name__)
        return getattr(self._module, self.attr_name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)

# Define all functions to be lazily imported
prepare_data = _LazyImport(".core.prepare", "prepare_data")
fit_lasso_glm = _LazyImport(".core.model", "fit_lasso_glm")
predict_with_model = _LazyImport(".core.model", "predict_with_model")
generate_blueprint = _LazyImport(".core.blueprint", "generate_blueprint")
load_external_dataframe = _LazyImport(".core.data", "load_external_dataframe")
ratetable = _LazyImport(".core.ratetable", "ratetable")
generate_all_ratetables = _LazyImport(".core.all_ratetables", "generate_all_ratetables")
plot_all_ratetables = _LazyImport(".core.plots", "plot_all_ratetables")
lump_rare_levels_pl = _LazyImport(".core.transforms", "lump_rare_levels_pl")
lump_fun = _LazyImport(".core.transforms", "lump_fun")
o_matrix = _LazyImport(".core.transforms", "o_matrix")

__all__ = [
    "prepare_data",
    "fit_lasso_glm",
    "predict_with_model",
    "generate_blueprint",
    "lump_rare_levels_pl",
    "lump_fun",
    "o_matrix",
    "load_external_dataframe",
    "ratetable",
    "generate_all_ratetables",
    "plot_all_ratetables",
]
