import sys
from typing import Any

# Lazy import implementation
class _LazyImport:
    def __init__(self, module_name: str, attr_name: str = None):
        self.module_name = module_name
        self.attr_name = attr_name
        self._module = None
        self._attr = None

    def _load(self):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        if self.attr_name and self._attr is None:
            self._attr = getattr(self._module, self.attr_name)
        return self._attr if self.attr_name else self._module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

# Create lazy import functions for each core function
def prepare_data(*args, **kwargs):
    """Lazy import for prepare_data function."""
    from .core.prepare import prepare_data
    return prepare_data(*args, **kwargs)

def fit_lasso_glm(*args, **kwargs):
    """Lazy import for fit_lasso_glm function."""
    from .core.model import fit_lasso_glm
    return fit_lasso_glm(*args, **kwargs)

def predict_with_model(*args, **kwargs):
    """Lazy import for predict_with_model function."""
    from .core.model import predict_with_model
    return predict_with_model(*args, **kwargs)

def generate_blueprint(*args, **kwargs):
    """Lazy import for generate_blueprint function."""
    from .core.blueprint import generate_blueprint
    return generate_blueprint(*args, **kwargs)

def load_external_dataframe(*args, **kwargs):
    """Lazy import for load_external_dataframe function."""
    from .core.data import load_external_dataframe
    return load_external_dataframe(*args, **kwargs)

def ratetable(*args, **kwargs):
    """Lazy import for ratetable function."""
    from .core.ratetable import ratetable
    return ratetable(*args, **kwargs)

def generate_all_ratetables(*args, **kwargs):
    """Lazy import for generate_all_ratetables function."""
    from .core.all_ratetables import generate_all_ratetables
    return generate_all_ratetables(*args, **kwargs)

def plot_all_ratetables(*args, **kwargs):
    """Lazy import for plot_all_ratetables function."""
    from .core.plots import plot_all_ratetables
    return plot_all_ratetables(*args, **kwargs)

def lump_rare_levels_pl(*args, **kwargs):
    """Lazy import for lump_rare_levels_pl function."""
    from .core.transforms import lump_rare_levels_pl
    return lump_rare_levels_pl(*args, **kwargs)

def lump_fun(*args, **kwargs):
    """Lazy import for lump_fun function."""
    from .core.transforms import lump_fun
    return lump_fun(*args, **kwargs)

def o_matrix(*args, **kwargs):
    """Lazy import for o_matrix function."""
    from .core.transforms import o_matrix
    return o_matrix(*args, **kwargs)

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
