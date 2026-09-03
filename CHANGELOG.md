# Changelog

## 0.4.0 (unreleased)

### Workbench state (piece W1)

- **Exploration sample vs full data.** The Project page setting is now an
  *exploration sample*: it only speeds up the Explore page and the preview
  charts on the Design and Variables pages. Fits, diagnostics, rate tables,
  the leakage report, and the knots and levels of every rating factor always
  use the full data. Behaviour change: a project saved by 0.3 with a sample
  set used to fit on the sample; it now fits on everything, and changing the
  sample never invalidates a fit.
- **Fitted models survive a reload.** Each fit is persisted next to the project
  file in `<project>.easyglm-runs/` (one file per model, latest only, plus a
  small JSON sidecar) and restored on the next page load when the model
  specification, the data file (path, size, last-modified time) and the
  library versions all match; anything else is discarded and refitted.
  Manual adjustments and the base-rate override are re-applied from the
  project file on load (an entry the model refuses is dropped with a message).
  A file whose key no longer matches is kept until a newer fit of the same
  model is saved, so looking at a different setting never erases the last fit;
  only unreadable files are removed. File names use a hash of the model name,
  so models whose names share a prefix (`freq`, `freq-2`) never touch each
  other's files. The folder holds pickles — trusted local content, like
  derived-column expressions. Unsaved projects persist nothing (the sidebar
  says so). Add `*.easyglm-runs/` to version-control ignores.
- `EasyGLM.summary()` reports the offset column; `easy_glm.__version__`.

### Piecewise-linear terms (piece B)

- Numeric factors can now be **piecewise-linear** instead of step functions
  (`DesignSpec.from_data(..., linear=["Mileage"])`, `LinearEncoder`,
  `VariableDesign(kind="linear")`): the relativity changes smoothly with the
  value along straight-line segments (on the log scale) between knots, and the
  lasso decides where the slope changes. The AGLM "L-dummies".
- The curve is **clamped to the training range** and flat beyond it (±infinity
  scores as the clamp value, like the GLM); missing values get their own row.
  The default clamp is the training minimum / maximum **rounded outward** to a
  round number (each end moves by less than 1 % of the range; the end bands keep
  their fitted slope up to it) — pass `clamp=` / set it on the Design page for
  an exact edge. Relativity 1.00 sits at the lower edge of the most exposed
  non-null band, recorded as `x_base` (JSON, Excel `is_base` column and Summary
  sheet); an edit of that band does not move `x_base`.
- Rate tables carry a `slope` per band and the value at both band ends; the
  `RateModel` scores them exactly (`"linear"` tables), Excel and the exported
  script round-trip them. `from_rate_tables` derives the slopes from the
  consecutive row values (so a table rounded to a few decimals reads back as a
  continuous curve), warns when a supplied `slope` column disagrees by more
  than 1 %, refuses a cliff at the lower clamp, zero-width bands, non-positive
  relativities and a table that lost its `slope` column but kept
  `relativity_to`, and warns when the null row is missing. JSON files are
  validated and ordered the same way.
- Every row of a linear table is a **node** of the curve: the `(None, lo)` row
  and the first band share the node at `lo` (editing either moves both, and
  `diff` lists both rows), the `(hi, None)` row is the node at `hi`. An edit
  re-derives the slope of the band(s) touching that node — one at either end,
  two in the interior — so the curve never jumps; the null row edits as a step.
  Non-positive relativities are refused; the workbench editor refuses them
  before saving, and an adjustment the model refuses is dropped with a message
  instead of locking the page.
- Monotone constraints are **not available** on linear terms in this release
  (`fit_glm` and `Project.validate` say so explicitly).
- Linear variables can be parents of interactions; diagnostics band them by the
  table's own edges (`Encoder.band_edges()`).

### Two-way interactions (piece A)

- Models can now include `A × B` interactions on top of the main effects
  (`DesignSpec.from_data(..., interactions=[("DrivAge", "VehPower")])`,
  `InteractionEncoder`, `ModelConfig.interactions`). Each cell of the two
  variables' rating rows gets its own multiplicative adjustment; cells with too
  little exposure (default 0.5% of the interaction's exposure) adjust by 1.0.
- Rate tables carry the adjustment as an extra table (`rate_tables`,
  `RateModel` type `"interaction"`, Excel long sheet plus a matrix sheet with the
  training exposure alongside). The scorer still reproduces the GLM exactly;
  the base rate is the prediction for the base risk *before* interaction
  adjustments, so a cell of 1.0 always means "no adjustment".
- Cells can be edited (`RateModel.update_relativity(..., from_b=, to_b=)`), are
  snapshot/diffed/JSON round-tripped, and recorded as `Adjustment(cell=True)` in
  projects and exported scripts.
- Interaction columns are penalised on the unstandardised scale (scaled so a
  50/50 cell matches a 50/50 main effect), so thin cells do not pick up noise.
- New diagnostic `workflow.ae_by_pair` (A/E by cell of two variables) to find
  interactions a model is missing.
- Every encoder exposes its rating rows and a shared `row_index` rule, used by
  the interaction cells, the tables and the scorer alike.
- Diagnostics (`univariate`, `ae_by_variable`, `ae_by_pair`) now label bands and
  the null / Other row exactly like the rate tables ("< 25.0", "[25.0, 30.0)",
  "Other / Unknown"), so they join onto tables by label.

#### Known limitations (to be addressed in the scale workstream)

- Interaction cells are built as a dense 0/1 block (one column per kept cell);
  at millions of rows with hundreds of cells this dominates memory. Workstream G
  will store the cell index instead.


Work in progress on the `release-0.4` branch. Plan: `docs/RELEASE_0.4_PLAN.md`;
independent reviews: `docs/reviews/`; plain-language checks: `docs/checks/`.

### Fixed (C1 — foundations)
- Excel exports (workbench download and the exported script) now contain the
  **adjusted** relativities; previously they silently held the fitted values.
- Categorical factors stored as whole numbers (e.g. vehicle power 4–12) scored
  every row as "Other"; levels are now compared as text everywhere.
- Models fitted with an **offset** (the rate-change setup) were ignored by the
  rate tables; the model file now records the offset column and applies it.
- Snapshot comparison (`RateModel.diff`) compares the two versions asked for.
- The stand-alone editor honours the requested port and derives its default
  actual-versus-expected formula from the model instead of guessing.

### Added (C1)
- `.easyglm` files carry `format_version: 2` (0.3 files open unchanged; newer
  files are refused with a clear message); unknown table types are an error.
- Project files carry `version: 2` with a migration hook and tolerant loading.
- Model metadata: `offset_col`, `offset_is_log`, `link`, `divide_target_by_weight`.
- `tests/test_invariants.py`: RateModel == GLM on every design, with nulls,
  unseen levels and offsets.

### Removed (C2 — legacy removal) — breaking
- `generate_blueprint`, `prepare_data`, `fit_lasso_glm`, `predict_with_model`,
  `ratetable` and `generate_all_ratetables` are gone, together with the DuckDB
  database engine they relied on and the `legacy` extra.
- `RateModel.from_rate_tables(tables, base_rate, ...)` takes the 0.3 table format
  (`from`, `to`, `relativity`; both-null row = null / Other) and no blueprint;
  `RateModel.from_glm_model(fit, ...)` takes a `GLMFit`; `create_rate_model`
  follows the same signature.
- `matplotlib` and `seaborn` moved to the `viz` extra; `scikit-learn` is no longer
  a direct requirement (glum still installs it); `rdata` is imported lazily. The
  base install declares 8 packages instead of 11.
- Exploratory scripts under `scripts/`, the broken `setup_dev` bootstrap scripts
  and the scoring prototype example deleted.

### Fixed (C2)
- `gini` pooled tied predictions inconsistently, so the reported Gini could move at
  the 1e-5 level between identical runs; ties are now pooled deterministically.

### Changed (C2)
- The benchmark runner fits easy_glm through `DesignSpec` + `fit_glm`.
- `RateModel.from_rate_tables` rejects duplicate levels, duplicate null/Other rows
  and gaps or overlaps between bands with messages that say what is wrong; bands
  may be listed in any order; integer-coded categorical tables are recognised.
- Snapshots can carry the metrics of their version (`create_snapshot(...,
  metrics=)`, `set_snapshot_metrics`); the workbench records train/holdout
  metrics on every fit.
- A golden French-motor test on a checked-in 50k-row subsample runs in CI.
