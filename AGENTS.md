## AGENTS.md

Purpose: Provide build/test commands, architecture guidance, and code style guidelines for AI agents operating in this repo.

---

## Build, Lint, and Tests

- Single test: `pytest tests/test_engine.py -k test_clone --maxfail=1 -q`
- Full suite: `pytest -q` (includes Streamlit `AppTest` smoke tests for every workbench page)
- Persona e2e: `EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <python-with-playwright> -m pytest tests/e2e` (navigate via sidebar links only; grids are canvas widgets — edit a cell with `_helpers.edit_grid_cell`, which double-clicks the cell, types into the overlay editor and waits for the expected text; set `EASY_GLM_E2E_SHOTS=<dir>` to get screenshots when an edit does not register). Any message drawn right before `st.rerun()` must go through `ui.flash()` (shown at the top of the next run), or Streamlit ≥ 1.63 drops it.
- Workbench: `python -m easy_glm.app [project.json]` (or `easy-glm-workbench`); headless: `--headless`
- Command line: `easy-glm run|export|validate|workbench project.json` (module form for a
  source checkout: `PYTHONPATH=src python -m easy_glm.cli run project.json --out out/`)
- Lint: `ruff check .`
- Types: `mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports` (a CI step;
  `core` and `workflow` are kept clean, the app and engine are not checked yet)
- Format: `black .`
- Run all quality steps: `black . && ruff check . && mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports && pytest -q`

## Releasing

Every `v*` tag pushed to GitHub auto-publishes to PyPI via Trusted Publishing.

```bash
# 1. Bump version in pyproject.toml
#    version = "0.2.3"

# 2. Commit and push
git add pyproject.toml
git commit -m "v0.2.3: description of changes"
git push origin main

# 3. Tag and push the tag (must start with 'v' to trigger publish)
git tag v0.2.3
git push origin v0.2.3
```

The workflow lives at `.github/workflows/publish.yml`. PyPI authenticates
via OpenID Connect (no tokens) — the project's Trusted Publisher is
configured at https://pypi.org/manage/project/easy-glm/settings/publishing/.

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

- `uv venv && uv pip install -e ".[dev]"` (dev includes streamlit + plotly)
- `PYTHONPATH=src` also works as a quick workaround for imports
- **IMPORTANT**: Tests use `PYTHONPATH=src` and hit the live source. Streamlit
  uses the *installed* package (site-packages). After editing source, sync with:
  ```bash
  rm -rf .venv/lib/python*/site-packages/easy_glm
  ln -sf "$(pwd)/src/easy_glm" .venv/lib/python*/site-packages/easy_glm
  ```
- Dataset caching: `load_external_dataframe()` caches to `~/.cache/easy_glm/` by default

---

## Public API (layers)

1. **Recommended:** `EasyGLM.fit()` — full pipeline.
2. **Building blocks:** `DesignSpec.from_data` → `fit_glm` (returns `GLMFit`) →
   `rate_tables` / `to_rate_model`. `EasyGLM.fit` is exactly these three calls.
3. **Scoring:** `RateModel` (lookup tables, reproduces the GLM exactly);
   `GLMFit.predict` (glum on `spec.build(data)`).
4. **Hand-built tables:** `RateModel.from_rate_tables({var: DataFrame(from, to,
   relativity)}, base_rate, ...)` — the table format written by `rate_tables`,
   `rate_model_tables` and the Excel export. The 0.2/0.3 blueprint + DuckDB
   pipeline was removed in 0.4.

Key invariant of the 0.3 core: `to_rate_model(fit).predict(data, exposure_col=None)
== fit.predict(data)` to ~1e-15, including nulls and unseen levels
(`tests/test_design_fit_tables.py::TestRateTables`).

---

## Architecture & Module Layout

```
src/easy_glm/
├── __init__.py             # Public API exports
├── core/
│   ├── design.py           # DesignSpec, StepEncoder (1{x>=k} + null col), LinearEncoder (one slope
│   │                       #   column per band + clamp; no knots = "continuous"),
│   │                       #   InteractionEncoder (A×B cells), CategoricalEncoder (one-hot + Other),
│   │                       #   Feature metadata, quantile_knots, frequent_levels; JSON round-trip
│   ├── fit.py              # fit_glm -> GLMFit (glum wrapper: families/links,
│   │                       #   alpha or CV, monotone_bounds -> lower/upper_bounds);
│   │                       #   fit_two_stage -> TwoStageFit (mains frozen, cells on top)
│   ├── tables.py           # rate_tables, base_rate, to_rate_model (exact, from coefs)
│   ├── excel.py            # write_rate_tables_xlsx, rate_model_tables (EasyGLM/RateModel.to_excel)
│   ├── easyglm.py          # EasyGLM pipeline (fit/predict/save/load) on the above
│   ├── data.py             # load_external_dataframe (with Parquet caching)
│   ├── plots.py            # plot_all_ratetables (matplotlib/seaborn via the viz extra)
│   └── split.py            # TRAIN_FLAG / HOLDOUT_FLAG, validate_train_test_column
├── cli.py                  # `easy-glm` console script: run / export / validate / workbench
├── workflow/               # GUI-agnostic workflow engine (docs/WORKBENCH_PLAN.md)
│   ├── project.py          # Project spec (data/roles/recodes/derived/filters/split,
│   │                       #   design overrides, model configs, adjustments); JSON; validate
│   ├── prep.py             # load_source, apply_variables (incl. the derived
│   │                       #   log(current premium) offset), add_split_column, prepare
│   ├── explore.py          # univariate, leakage_report (single-factor GLM strength etc.)
│   ├── diagnostics.py      # deviance, lift, gini, double lift, ae_by_variable,
│   │                       #   residual_factor_search, alpha_path, model_metrics,
│   │                       #   relativity_diff / describe_diff (champion vs challenger)
│   ├── run.py              # build_design, run_model -> ModelRun, rebuild_rate_model,
│   │                       #   solve_base_rate (target loss ratio), exposure_for/link_for
│   ├── export.py           # to_script (self-contained Python; tested by execution)
│   ├── report.py           # to_report_html — ONE self-contained HTML file
│   └── _svg.py             # the report's charts as plain SVG (no JS, no library)
│                           #   every chart carries a <title> = its accessible name
├── app/                    # Streamlit workbench (thin views over workflow + state)
│   ├── main.py             # st.navigation entry; --project=path
│   ├── state.py            # session Project, hash-keyed caches (raw/prepared/runs/leakage), autosave
│   ├── charts.py, ui.py    # plotly charts (incl. heatmaps, linear curves), shared widgets
│   ├── grids.py            # pure grid-edit rules (row / cell adjustments, pair matrices)
│   └── pages_*.py          # one module per page: project, variables, explore, split,
│                           #   design, model, diagnostics, compare, tables, export
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

### Core (0.3) Conventions

- `DesignSpec` is the single source of truth for what a design column means;
  never derive meaning from feature-name strings. Column order = encoder order,
  see `DesignSpec.slices()`.
- Step columns are `1{x >= knot}`; nulls are all-zero step columns (lowest bin)
  plus an `is null` column. Bin `j` relativity = `exp(cumsum(step coefs)[:j])`.
- **Linear (piecewise-linear) columns are per-band amounts**
  `clip(x_clipped - k_j, 0, k_{j+1} - k_j)`, one per band of
  `[lo, k1, ..., km, hi]`, so coefficient `beta_j` **is** band `j`'s slope and the
  lasso zeroes *slopes* (flat sections), not bends — the actuary's answer to Q3.
  Never reintroduce hinge (`max(x-k, 0)`) columns: the table's `slope` column is
  read straight off the coefficients and `log_rel_at_from` accumulates from `lo`,
  so continuity is automatic. Nulls are all-zero band columns plus `is null`.
  A `LinearEncoder` with no interior knots is the `continuous` kind (one band).
- **Interactions are fitted in two stages, never jointly** (`core/fit.py::fit_two_stage`,
  the actuary's answer to Q5). Stage 1 is `spec.main_effects_spec()` — bit for bit the
  fit the model gets with no interaction — and stage 2 is `spec.interactions_spec()`
  with `fit_intercept=False` and `offset = eta1` (stage 1's linear predictor plus any
  offset column). The composed `TwoStageFit` *is* a `GLMFit` (full spec, stage 1's
  coefficients then stage 2's, stage 1's intercept), so nothing downstream special-cases
  it. Never take a main table or the base rate from anything but stage 1: "adding an
  interaction moves no main relativity" is a promise made to the actuary and is tested
  to 1e-13 (glum's own run-to-run noise) in `test_recovery.py`. Whether a fit *had* two
  stages is `isinstance(fit, TwoStageFit)`, never "the design has an interaction": an
  interaction whose cells are all below the exposure floor has an encoder and no
  columns, and `fit_two_stage` then returns a plain `GLMFit`. Anything that emits or
  branches on a stage 2 (the exported script, the Model page, `EasyGLM.save`) must ask
  the fit. Stage 2 carries no intercept, so any overall re-levelling it wants lands in
  the cells — say so wherever a cell is described as a "pure adjustment".
- **Band columns and interaction cells carry a `P1`** (`core/fit.py::penalty_weights`).
  glum penalises the *standardised* coefficient, so a column with little spread buys a
  large effect cheaply. For a band the effect is its **rise** (`beta_j x width_j`), so
  `P1_j = 0.5 / sd(column_j / width_j)` makes one unit of rise cost the same in every
  band; without standardisation the same equality needs `P1_j = width_j x n_bands /
  (hi - lo)`. Never rescale the columns themselves to achieve this — `beta_j` must stay
  band `j`'s slope so the table can read it straight off the coefficients. Note the
  standardised form **raises** a term's total penalty (`sd(u) <= 0.5`, so every
  `P1_j >= 1`): 1.6-4.1x on the check's multi-band fits and 3.6-6.3x on a one-band term.
  Only the unstandardised form is a pure redistribution. Any statement that this rule
  "only redistributes" is wrong. For cells the two forms are the *same* penalty
  (`0.5/sd` x `sd` = `0.5`), which is what lets stage 2 — where glum refuses to
  standardise, having no intercept — penalise a cell exactly as a joint fit did.
- Categorical reference level = `levels[0]` (most frequent, no column); `Other`
  column catches lumped, unseen and null values.
- `fit_glm` requires `alpha=` or `cv=`; never let glum's `alpha_search` pick
  (its `coef_` is the least-regularised end of the path).
- Monotone constraints are coefficient sign bounds on the step columns of a step
  term or the band-slope columns of a linear/continuous term (work with L1; glum's
  own `monotonic_constraints` does not). Bounding slopes makes the curve monotone
  without forcing convexity, which is why linear terms may be constrained.
  Categoricals and interactions cannot; a constraint binds the factor's own curve,
  never the interaction cells on top of it.
- The 1.00 point of a linear term must be an **edge of a table row** — that is how the
  rate table, the Excel `is_base` column and `from_rate_tables` carry `x_base`. A
  one-band (`continuous`) term therefore bases at the lower clamp unless the
  exposure-weighted median is past `CONTINUOUS_BASE_AT_HI` (0.6) of the range
  (`core/fit.py::_continuous_base_row`), not at a point inside the band. The threshold is
  off centre so a mid-range factor cannot flip between clamps on sampling noise.
- Numeric `VariableConfig` tables may end with a `FromToRow(None, None, rel)` null
  row; `_precompute_variables` stores it as `null_relativity` and `score_numeric`
  applies it to NaN. Without that row NaN still raises (legacy behaviour).

### Rate-change, penalties, families (piece E)

- A column with the role **`current_premium`** makes `prep.apply_variables` derive
  `log_<premium>` **after the row filters** (so `pl.col('Premium') > 0` does its job
  first) and refuse the whole frame, by name and count, when a premium has no
  logarithm. `Project.offset_column` pre-fills that column as the offset of every new
  model; `rename_column` and `apply_role_change` carry the derivation with the role.
  `to_script` writes the `pl.col(...).log().alias(...)` line out: the setup must be
  visible in the script, never implied by a role.
- `ModelMetadata.offset_is_premium` only changes **labels**
  (`RateModel.relativity_label` / `relativity_note`, used by Excel, the Rate tables page
  and the Export page). No number depends on it. It defaults to False, so `.easyglm`
  files written before it read correctly and the format version did not move.
- **Every encoder carries a `penalty_weight`** that multiplies whatever per-column rule
  `penalty_weights` already applied to it (band rises, interaction cells); 0 =
  unpenalised. It weights the **L1** penalty only — with `l1_ratio < 1` glum's ridge
  part still applies to every column.
- `resolve_family(family, tweedie_power)`: the power must be strictly in (1, 2), and
  passing it for another family is an error, not a silent no-op.
- **`log` and `logit` are both multiplicative links.** A logit fit's tables are *odds*
  relativities and its base rate is the base risk's odds; `RateModel._response` turns
  the product back into a probability. Such a model refuses an exposure column in three
  places — `to_rate_model`, `RateModel.predict` and `exposure_for`, which never gives
  it one — because a probability is not an amount.
- All of the above compose with the **two-stage** interaction fit: stage 1 (the
  frozen mains) carries the offset column, the family and every main's
  `penalty_weight`; stage 2 fits only the cells, so a main's weight never reaches
  it, and the cells' `P1` is `penalty_weight * 0.5` there (glum cannot
  standardise without an intercept). `tweedie_power` travels in `fit_two_stage`'s
  kwargs, so both stages use the same distribution, and the exported two-stage
  script writes it in both calls.
- `solve_base_rate(run, df, r)` sets the base rate so **total actual / total expected =
  r** on the rows given. It works on a `TwoStageFit` unchanged: the base rate is
  stage 1's and the cells are stage 2's, but the prediction is still proportional
  to the base rate. For a rate-change model that ratio is the loss ratio the book
  would be written at; for an ordinary model r = 1 rebalances it. It reads the run's
  *current* base rate, so an existing override cancels out and solving twice is
  idempotent. Logit models are refused (a probability is not proportional to the base
  rate).

### Command line (piece F)

- `easy_glm/cli.py` never raises through to the user: every failure is a `CliError`
  rendered as a message with exit code 1 (2 is argparse's usage error).
- Every artefact command **fits afresh** — persisted workbench runs are deliberately
  not reused, so a command line result never depends on someone's browser session.
- Two `.easyglm` files from the same project are *not* byte-identical: snapshots carry a
  timestamp, and glum's solver is not bit-reproducible (~1e-15 on a relativity, even
  between two fits in one process). Compare predictions, or JSON with timestamps
  stripped and floats to a tolerance — never bytes.
- `safe_filename` lives in `workflow.project` (the CLI must not import Streamlit);
  `app.ui` re-exports it.

### Workbench Conventions

- Pages never compute; they call `easy_glm.workflow` and read/write the `Project`
  through `app.state` (`S.project()`, then `S.touch()` after any mutation = autosave).
- Caches are keyed on spec hashes (`state.model_hash` excludes adjustments / base-rate
  override, which are applied post-fit via `rebuild_rate_model`).
- Two frames in state: `prepared_frame()` (full data — fits, diagnostics, tables,
  leakage, knots/levels) and `sample_frame()` / `raw_sample()` (exploration only:
  Explore page, Design/Variables previews). `data_hash`/`model_hash` exclude the
  sample settings; `sample_hash` keys the sample.
- **Bump `app.state.PERSIST_FORMAT` whenever the shape *or the meaning* of anything
  pickled changes** (`ModelRun`, `GLMFit`, `RateModel`, `DesignSpec`, what their
  coefficients mean). A change of meaning that leaves the shape alone is the dangerous
  one: the pickle unpickles cleanly, `_design_matches` compares the new feature names
  against themselves and passes, and the cached fit is silently re-read under the new
  rules. Worked example: piece **B2** turned a `LinearEncoder`'s coefficients from
  change-of-slope (hinge) numbers into per-band slopes without touching a single pickled
  field — `PERSIST_FORMAT` 2 → 3 is what makes those runs a cache miss. The installed
  version number is not a substitute: it does not move in a development checkout.
- A browser reload starts a new Streamlit session: the project (spec) survives via the
  autosaved JSON; fitted runs are restored from `<project>.easyglm-runs/` when
  `run_key` (spec hash + data file identity + library versions + `PERSIST_FORMAT`)
  matches, else refitted.
  `load_persisted_run` treats any failure as a cache miss, and deletes the file only
  when the pickle is corrupt or its design no longer matches *readable* data — data
  that cannot be read right now is a miss that keeps the fit;
  adjustments/base-rate override are re-applied from the project on load.
- Comparing two models: `relativity_diff` puts **numeric and piecewise-linear**
  factors on the union of both models' band edges (so a moved knot, or a factor
  banded in one model and straight in the other, is still compared like for like)
  and matches **categorical levels and interaction cells by label**. Identical
  values are never a change, zeros included. The base rate is both a row of the
  table and, through `base_rate_change`, the headline above it — a band's premium
  change is its relativity change times the level change.
- Champion vs challenger: the sidebar (`main.py`) owns one "compare with" model in
  `state.CHALLENGER_KEY` (`S.challenger()` / `S.set_challenger()`; not an app-state
  key, so another project never inherits it). Diagnostics, Rate tables, Compare and
  the Export page's report default to it and allow a page-level override by putting
  the sidebar value **in the widget key** (`f"diag_chal_{sidebar}"`), so moving the
  sidebar re-defaults the page widget while a page choice sticks until it does.
  `S.fitted_models()` / `S.latest_run()` are the model lists pages select from.
- Navigation between pages must be client-side (sidebar links) to keep session state;
  Playwright drivers should click sidebar links rather than `goto` page URLs.
- Errors are messages, never tracebacks: pages call `ui.guarded` / `ui.require_data`,
  `state.prepared_frame()` stores a failing step in `prep_error` and returns None,
  `state.save_project` / `state.touch` return or record errors instead of raising.
  `open_project_file` validates before replacing the open project.
- Multi-tab rule: `state.touch()` compares the file's stamp (`_file_stamp`: mtime_ns,
  size and a sha1 of the bytes — mtime alone is too coarse on NFS/SMB/FAT) with the one
  this session last read/wrote; a mismatch sets `conflict`, pauses autosave and shows
  `ui.conflict_notice()` (reload = `set_project` from disk, overwrite = forced save).
- Runs-folder rule (W4; plain-language page: `docs/checks/w4-runs-folder.md`): the
  folder is shared by every tab, so `persist_run` refuses to write while
  `runs_write_paused()` (the conflict notice is up) and nothing is deleted while
  `runs_delete_paused()` (that, or the project file changed on disk under this tab).
  `_prune_runs` keeps the key of the project *as saved on disk* (`_saved_project()`)
  and of this session's current spec; only files matching neither go, together with
  files of models absent from both projects and sidecars whose pickle is gone. A fit
  writes a `<tag>-<key>-<session>.fitting` marker (`_mark_fit_started`) that
  `interrupted_fits()` turns into a notice in `ui.status_bar()` (once per session, and
  again in later sessions while it stays true); a marker is *removed* only when its
  result is on disk, when this session wrote it, or when it is older than
  `MARKER_GRACE_SECONDS` — a younger one from another session may be a fit still
  running, and a running fit is indistinguishable from an interrupted one from
  another tab (said on the check page). Bump `PERSIST_FORMAT` when a pickled class
  changes shape.
- `Project.to_json` writes to a unique temporary file and renames it over the target,
  so no reader ever sees half a project file and two writers cannot interleave; a
  target the user made read-only is still refused (the rename would not be).
  A successful save (autosave included) drops the "Autosave failed" banner.
- Rename rule: column renames go through `Project.rename_column` (roles, types,
  recodes, design, split, row filters and derived expressions — `pl.col('old')`
  references only — and every model reference); role changes through
  `Project.apply_role_change`; `Project.missing_columns` / `validate(columns=...)` refuse
  a model that references a column the prepared data lacks — never re-point a selector
  (`index=None` + an error). Model names go through `validate_model_name`; downloads use
  `ui.safe_filename`.
- `state.set_project` bumps `project_token` and drops every session-state key that is
  not app state (`_APP_STATE_KEYS`) or `_`-prefixed, so widgets never carry the previous
  project's values; project-page text boxes use `state.widget_key(name)`.
- Break-it findings left open after W4 (cosmetic): 15, 23, 35 in
  `docs/reviews/w2-breakage.md`; 6 and 8 in `docs/reviews/w3-breakage-2.md`
  (offset plausibility needs an actuary's rule; keying a fit on the data file's
  contents instead of its mtime means reading the whole book on every page).
  Everything else in both reports is fixed.
- Widget rule: Streamlit refuses to set a widget's session-state key once the widget
  exists in that run, so a page that has to change a box's value (Create selecting the
  new model, `ui.number_in_range` putting a refused number back, the divide box after
  the weight is cleared) drops the key *before* the widget is created on the next run —
  a pending flag plus `st.rerun()`, never an assignment after the fact.
- Any message drawn immediately before `st.rerun()` (a repaired seed, a clamped
  training fraction, a knot outside the data) must go through `ui.flash`.
- Rate-table labels: the catch-all row prints `NULL_LABEL` ("Other / Unknown") unless
  the categorical encoder had to rename its bucket (a real level called `Other`), in
  which case `VariableConfig.other_label` carries the encoder's name through
  `level_label` to the tables, the Excel workbook and the Rate tables page.

### Golden numbers

- `tests/test_golden.py` fits a fixed model on `tests/fixtures/french_motor_50k.parquet`
  and compares against recorded numbers. Changing a golden number is a blocking
  review item and needs a written reason in the PR.

---

## Key Tests

| File | What it tests |
|---|---|
| `test_engine.py` | RateModel: from_rate_tables (0.3 table format, null/Other rows, validation), from_glm_model, predict (numeric/categorical/multi), update_relativity, snapshots (+ metrics), switch_to, clone, JSON roundtrip, exposure, column mapping, metadata |
| `test_golden.py` | Golden French-motor numbers on the checked-in 50k subsample (runs in CI) |
| `test_invariants.py` | RateModel == GLM, JSON and Excel round-trips over step / categorical (string and integer) / mixed / offset / interaction (joint **and** two-stage) / piecewise-linear designs with nulls and unseen levels |
| `test_linear.py` | Pieces B / B2: `LinearEncoder` (clamp, per-band slope columns, nulls, the one-band `continuous` kind), band slope = the coefficient itself, continuity, monotone as a sign bound on slopes, exactness at/beyond the clamp, band-edit rule, snapshots/JSON/Excel/`from_rate_tables`, interaction with a linear parent, project validation, script round trip, workbench pages |
| `test_interactions.py` | Piece A / A2: `InteractionEncoder` (cells, exposure, symmetry, JSON), the engine's cell table, Excel long + matrix sheets, the cell `P1` rule, and the two-stage fit (`TwoStageFit` composition, frozen mains, exactness, offsets, what it refuses) |
| `test_recovery.py` | Planted truth: the `Age x Region` cell recovered at a fixed and a CV alpha with thin cells left at 1.00, main tables unmoved by adding the interaction, and the flat/sloped/flat mileage curve of B2 |
| `test_c1_foundations.py` | 0.3 bug regressions, format versions and migrations, editor defaults |
| `test_scoring.py` | Isolated scoring: score_numeric (searchsorted), score_categorical (dict lookup), edge cases, fallbacks |
| `test_workflow.py` | Project JSON/validation, prep steps, univariate, leakage report on planted leaks, build_design overrides, run_model (metrics, exactness, adjustments, CV, the two stages and `Interaction.alpha`), diagnostics, exported script executed in a subprocess and compared |
| `test_e_f_extras_cli.py` | Pieces E/F: the `current_premium` role and its derived offset (error message, filter order), the "multiplier on current premium" labels through Excel and JSON, the **offset identity** of plan §R6/S1 (Poisson, `scale_predictors=False`, alpha x sum(P)/n, 1e-8) with Gamma recorded as *not* matching, per-variable penalty weights (`P1` aligned with `spec.features`, an unpenalised categorical keeping every level), Tweedie power, binomial (probabilities, odds labels, three refusals of exposure), `solve_base_rate` (rate change, rebalance, weights, an existing override, idempotence), and the `easy-glm` CLI through `subprocess` |
| `test_w4_runs_folder.py` | W4: the shared runs folder (two AppTest sessions per two-tab case) and every finding of `docs/reviews/w3-breakage-2.md` |
| `test_d3_d4_compare_report.py` | D3/D4: `relativity_diff` (identical runs, one known adjustment, a moved knot on the common grid, step-vs-linear, symmetry, the base rate, the tolerance boundary, two zeros), `to_report_html` (self-contained, **no `<script>` at all**, one section per predictor, an accessible name per chart, compare section only with a challenger — and an explanation when the challenger cannot be scored, size), `_svg` (ticks, degenerate charts, escaping), the Compare page / sidebar challenger / Export report button through AppTest. **D4's "opens in a browser with no console error"**: the static half (no script, no external `src`/`href`) is proved here on every run; the browser half is `test_it_opens_in_a_headless_browser_without_console_errors`, which *skips* where Playwright is absent (the default venv) and runs in the Playwright venv and in `tests/e2e/test_persona_data_scientist.py` — CI must run one of those two for the criterion to be covered |
| `test_w2_pages.py` | W2 pages: interaction section, linear editor, kind selector, A/E-by-pair, pair search, cell/band edits via `app.grids`, break-it (empty project, missing file, removed predictor) |
| `tests/e2e/` | Playwright persona runs (actuary rate review, data-scientist comparison incl. the Compare page and the downloaded HTML report); opt-in `EASY_GLM_E2E=1`, server from `EASY_GLM_SERVER_PYTHON` |
| `test_app.py` | AppTest: every workbench page renders (with and without a fit), main entry point, leakage scan action |
| `test_design_fit_tables.py` | 0.3 core: DesignSpec/encoders, fit_glm (alpha/cv/monotone/validation), exact RateModel reproduction incl. nulls + unseen levels, numeric null row scoring, EasyGLM save/load, A/E masks |
| `test_easyglm.py` | EasyGLM front door: fit/predict, equivalence with the building blocks, serialization |
| `test_ui.py` | Metrics: compute_actual_expected (train/test split, formulas, edge cases). Charts: histogram, relativity, A/E |
| `test_imports.py` | import easy_glm does not eagerly import matplotlib |
| `test_benchmarking.py` | Data generators, metrics, fit_glm families, benchmark runner (easy_glm vs statsmodels vs catboost) |
