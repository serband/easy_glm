## AGENTS.md

Purpose: Provide build/test commands, architecture guidance, and code style guidelines for AI agents operating in this repo.

---

## Build, Lint, and Tests

- Single test: `pytest tests/test_engine.py -k test_clone --maxfail=1 -q`
- Full suite: `pytest -q` (121 tests)
- Lint: `ruff check .`
- Format: `black .`
- Run all quality steps: `black . && ruff check . && pytest -q`

## Quick Verification

```bash
PYTHONPATH=src .venv/bin/python -c "
from easy_glm.engine import RateModel
from easy_glm.engine.models import FromToRow, VariableConfig
rm = RateModel(base_rate=0.1, variables={})
print('OK')
"
```

---

## Installation

- `python setup_dev.py` — handles editable install + symlink fallback
- `PYTHONPATH=src` also works as a quick workaround for imports
- **IMPORTANT**: Tests use `PYTHONPATH=src` and hit the live source. Streamlit
  uses the *installed* package (site-packages). After editing source, sync with:
  ```bash
  rm -rf .venv/lib/python*/site-packages/easy_glm
  ln -sf "$(pwd)/src/easy_glm" .venv/lib/python*/site-packages/easy_glm
  ```
- Dataset caching: `load_external_dataframe()` caches to `~/.cache/easy_glm/` by default

---

## Architecture & Module Layout

```
src/easy_glm/
├── __init__.py             # Public API exports
├── core/
│   ├── blueprint.py        # generate_blueprint (quantile breaks, level lumping)
│   ├── prepare.py          # prepare_data (DuckDB SQL transforms)
│   │                       #   NOTE: _own_connection flag — only close connections
│   │                       #   we created. If user passes `con`, we never close it.
│   ├── model.py            # fit_lasso_glm, predict_with_model
│   │                       #   Uses pandas .astype("category") for text columns
│   │                       #   (replaced dask-ml Categorizer in v0.2)
│   ├── ratetable.py        # ratetable (per-variable relativity extraction)
│   ├── all_ratetables.py   # generate_all_ratetables
│   ├── transforms.py       # o_matrix, lump_fun, lump_rare_levels_pl (SQL generators)
│   ├── data.py             # load_external_dataframe (with Parquet caching)
│   ├── plots.py            # plot_all_ratetables (matplotlib/seaborn)
│   └── easyglm.py          # EasyGLM pipeline (fit/predict/save/load)
├── engine/
│   ├── rate_model.py       # RateModel — the core model representation
│   │                       #   Key methods:
│   │                       #   - predict(data)          → np.ndarray
│   │                       #   - clone()                → RateModel (deep copy)
│   │                       #   - update_relativity(...) → mutates + precomputes
│   │                       #   - compute_ae_for_variable(data, var) → dict
│   │                       #   - non_constant_variables → dict (property)
│   │                       #   - create_snapshot / switch_to → versioning
│   │                       #   - to_json / from_json    → serialization
│   ├── _scoring.py         # score_numeric (np.searchsorted fast path)
│   │                       #   score_categorical (dict lookup)
│   │                       #   Both have fallback paths when precompute is None
│   └── models.py           # Dataclasses: FromToRow, VariableConfig, Snapshot,
│                           #   Change, ModelMetadata, SessionState
├── ui/
│   ├── __init__.py         # launch_editor (non-blocking, uses subprocess.Popen)
│   ├── app.py              # Streamlit relativity editor
│   │                       #   Architecture:
│   │                       #   - baseline_rm → read-only original model
│   │                       #   - working_rm  → clone, all edits go here
│   │                       #   - saved_models → dict of named RateModel instances
│   │                       #   - Sidebar: variable overview, column mapping,
│   │                       #     A/E controls, save/reset, saved models list
│   │                       #   - Main panel: baseline vs working relativity
│   │                       #     charts, baseline vs working A/E charts,
│   │                       #     editable table
│   ├── charts.py           # Plotly charts (histogram, relativity, A/E)
│   └── metrics.py          # compute_actual_expected, FORMULAS
└── benchmarking/
    ├── benchmark.py        # run_benchmarks (easy_glm vs statsmodels vs CatBoost)
    ├── data_generators.py  # Synthetic data generators (Poisson/Gamma/Gaussian/Binomial)
    └── metrics.py          # Deviance, RMSE, MAE per family
```

### Editor Architecture (Baseline vs Working Copy)

The editor uses a **git-style fork model**:

| Entity | Mutability | Purpose |
|---|---|---|
| `baseline_rm` | Read-only | Original model loaded from disk |
| `working_rm` | Mutable | `baseline.clone()` — all edits go here |
| `saved_models[name]` | Read-only | `working.clone()` saved with a name |

**Key invariants:**
- The baseline is NEVER mutated by the editor.
- `RateModel.clone()` uses `_to_dict()` → `_from_dict()` for guaranteed independence.
- `update_relativity()` calls `_precompute_variables()` so the fast scoring path
  is rebuilt after every edit.
- A/E caches (`ae_baseline`, `ae_working`) are invalidated on edit and
  recomputed reactively (or manually if toggled).

### Data Flow: Edit → Scoring → A/E

```
User edits relativity in table
  → working.update_relativity(var, from_, to_, new_value)
    → config.table[row].relativity = new_value
    → _precompute_variables(self.variables)  # rebuild breakpoints/cat_map
    → Invalidate ae_working[var]
    → If auto_recompute: compute_ae_for_variable(working, data, var)
  → Rerun → charts reflect new relativities + A/E
```

---

## Code Style and Conventions

- Imports: standard library first, third-party second, local imports last; blank-line groups.
- Formatting: adhere to Black; line length 88; trailing commas where helpful.
- Types: use type hints everywhere; `from __future__ import annotations` where possible.
- Naming: descriptive names; no abbreviations; PascalCase classes; snake_case functions.
- Error handling: specific exceptions; no bare `except:`; meaningful messages.
- Tests: small, fast unit tests; `pytest`; follow test style in `tests/`.
- Documentation: docstrings for public API; comments sparingly but clear.

### `RateModel` Conventions

- Always call `_precompute_variables()` after mutating `VariableConfig.table` or
  after deserializing, so `breakpoints`/`relativities`/`cat_map`/`fallback` are populated.
- `clone()` serializes to dict and back — this is the only safe way to deep-copy
  without shared references across dataclass lists.
- `predict()` handles exposure multiplication internally via `_apply_exposure()`.
  Pass `exposure_col=None` to skip.

### `prepare_data` Conventions

- Tracks connection ownership via `_own_connection`. Only close if we created the connection.
- Uses `quote_identifier()` for all SQL identifiers to handle spaces and reserved words.
- Empty blueprints (`[]`) cause the column to be skipped entirely.

---

## Key Tests

| File | What it tests |
|---|---|
| `test_engine.py` | RateModel: from_rate_tables, predict (numeric/categorical/multi), update_relativity, snapshots, switch_to, clone, JSON roundtrip, exposure, column mapping, metadata |
| `test_scoring.py` | Isolated scoring: score_numeric (searchsorted), score_categorical (dict lookup), edge cases, fallbacks |
| `test_model_and_ratetable.py` | fit_lasso_glm, predict_with_model, ratetable, EasyGLM pipeline, serialization |
| `test_blueprint.py` | generate_blueprint basic smoke test |
| `test_nulls.py` | Null handling in blueprint and prepare_data |
| `test_ui.py` | Metrics: compute_actual_expected (train/test split, formulas, edge cases). Charts: histogram, relativity, A/E |
| `test_imports.py` | import easy_glm does not eagerly import matplotlib |
| `test_benchmarking.py` | Data generators, metrics, benchmark runner |
