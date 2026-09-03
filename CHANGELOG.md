# Changelog

## 0.4.0 (unreleased)

### Champion vs challenger, and a report you can send (D3 / D4)
- **A "Compare with" box in the sidebar.** Pick a challenger once and the whole
  session follows it: the Diagnostics page overlays it, the Rate tables page
  draws its expected line on the A/E chart, the Compare page and the HTML report
  default to it. Each page still lets you override it for that page alone.
- **New Compare page** (after Diagnostics). Two fitted models side by side: rows,
  exposure, A/E, Gini, deviance explained and mean deviance on train and holdout,
  plus each model's alpha, non-zero terms, interactions, linear terms,
  adjustments and base rate. A/E by any variable with *both* models' expected
  lines, lift for each, the double lift, and **Make … champion** buttons. The
  metrics recorded with each saved version of the rate tables (`Snapshot.metrics`)
  are shown when there are any.
- **A table of which relativities actually differ.** `workflow.relativity_diff`
  lists every band whose relativity moved by more than a tolerance (default 1 %,
  editable on the page), on the log scale so `+0.10` reads as "the challenger
  charges about 10 % more for that band". Interactions are compared cell by cell,
  piecewise-linear terms by their band-start values, and the base rates against
  each other — the overall level change is also shown on its own line above the
  table (`workflow.base_rate_change`), because a band's premium change is its
  relativity change multiplied by it. **Numeric factors are compared on the
  union of both models' band edges**, so a moved knot reports exactly the range
  of ages that would be charged differently, and the same factor banded in one
  model and a straight line in the other is still compared like for like (the
  `kind` column then reads `numeric → linear`); levels and interaction cells are
  matched by name, and a factor only one model has is listed once. Two identical
  models — or two bands both floored to the same value, zero included — give an
  empty table. `workflow.describe_diff` puts the statuses into words for a page.
- **One self-contained HTML report.** `workflow.to_report_html(project, runs, df,
  champion=..., challenger=...)` and a **Download HTML report** button on the
  Export page: a single file — summary (data, split, metrics), one block per
  rating factor (relativities, actual vs expected on train and on holdout, the
  rate table), interaction heatmaps, lift and Gini, the comparison section when a
  challenger is chosen, every coefficient and the exported Python script in an
  appendix, with the generation time and library versions. Nothing is fetched
  from the internet when it is opened: the charts are written as plain SVG
  (`workflow/_svg.py`) with a `<title>` naming each one, which keeps the
  French-motor report at 350–400 kB instead of the 4.8 MB an inlined charting
  library would cost, and means the file contains no JavaScript and so cannot
  produce a browser error. A challenger the report cannot score on these rows
  is explained in the comparison section's place, never silently dropped.
- **Known limitation.** D4 asks for the report "from the Export page *and the
  CLI*". Only the Export page (and `workflow.to_report_html` for scripts) ships
  here — there is no `easy-glm` console script yet, so the CLI half of D4 lands
  with workstream F (`easy-glm run project.json`) and must not be forgotten when
  0.4.0 is cut.
- Tests: `tests/test_d3_d4_compare_report.py` (the diff on hand-made
  differences, the report's self-containment / one section per predictor /
  compare-section-only-with-a-challenger / size / headless render, and the pages
  through AppTest); the data-scientist persona e2e now drives the Compare page
  and opens the downloaded report in the browser; the plain-language replay is
  `docs/checks/d3-d4-compare-report.md`
  (`scripts/checks/d3_d4_compare_report.py --write`).

### Workbench hardening (W3) — the break-it review's blocking findings
- **No more silent loss of work.** *New empty project* asks for a second click and
  starts with no file (the old project file is never rewritten); the same project
  open in two tabs no longer overwrites itself — before every autosave the
  workbench checks whether the file changed on disk and, if so, pauses with a
  notice offering *Reload from disk* or *Overwrite*; renaming a column in the
  roles grid renames it in every role, type, recode, design, split and model
  reference instead of the Target box jumping to the first column; a random split
  can no longer be named after an existing data column.
- **Tracebacks became messages.** A data file whose columns differ from the
  project's, a rename onto an existing name, a cleared rename cell, a derived
  column that cannot run (now executed before it is added), the Split page with
  no usable column or a text indicator column, five kinds of broken project file
  (upload or path), saving to an impossible path or a read-only file, and a model
  called `a/b` (names are validated; legacy names get file-safe downloads) all
  end with a sentence on the page. A failing pipeline step is reported on every
  page, and a failing autosave at the top of every page.
- **Model references a missing column**: the model is never re-pointed. The
  selector is left blank, the Model page names the missing column, Fit is
  disabled and any persisted fit is ignored until you choose.
- Smaller fixes from the same review: target/weight/offset offer numeric columns
  only; uploads are stored next to the project rather than in a temp folder; a
  cleared recode cell is no mapping; the minimum-level-share message says what
  it means; a clamp range outside the training range is refused; relativities
  cannot be set to 0; a real level called *Other* gets a distinct lumped label;
  non-finite knots are refused; the Project page never shows another project's
  name or path; huge base-rate overrides warn and absurd percentages print as
  "—"; target = weight and alpha = 0 are validation problems; the divide box is
  unticked without a weight; the split slider clamps out-of-range values and an
  empty split name is refused; deleting a model removes its persisted fit;
  monotone on a categorical is refused in the grid.
- Tests: `tests/test_w3_hardening.py` (one per finding), an opt-in Playwright
  break-it run (`tests/e2e/test_breakit.py`) and the plain-language replay in
  `docs/checks/w3-hardening.md` (`scripts/checks/w3_hardening.py --write`).

### Workbench pages for interactions and piecewise-linear terms (W2)
- **Design page**: an *Interactions* section per model — add a pair (minimum cell
  exposure, penalty weight) with a preview of the training exposure per cell and
  how many cells would get their own adjustment; remove interactions; clear
  messages for the same variable twice, a duplicate pair or a non-predictor.
  The variable detail gained a *Kind* selector and a full piecewise-linear
  editor (knot strategy, custom knots, clamp lo/hi with the rounding rule shown,
  preview with the clamp points marked); bad inputs are messages, not errors.
- **Model page** lists the model's interactions next to its predictors and warns
  when a parent is no longer a predictor.
- **Diagnostics**: a new *A/E by pair* tab (heatmap of actual / expected per
  cell with actual, expected and exposure on hover, for any two variables), and
  *Search pairs* — every pair of the model's predictors ranked by the Pearson
  excess of its cells after re-fitting the margins (a z-score), the way to find
  interactions worth adding. `workflow.residual_pair_search` is the API.
- **Rate tables**: interaction tables show the adjustment heatmap, the A/E by
  cell and an editable cell grid (edits saved as cell adjustments, applied
  without a refit; 0 or below refused); linear tables show the continuous
  curve with the clamp and base points and a node editor with the slope and the
  value at both band ends. The base-rate caption explains that cells are
  adjustments (1.00 = none).
- **Robustness**: every page survives an empty project, a missing data file and
  a model whose predictors were removed with a message instead of a traceback;
  the data steps failing (a bad recode / derived column / filter) are reported
  on the page.
- **Notices survive reruns**: messages shown just before a rerun (a dropped monotone constraint, an added interaction, a refused adjustment, a filter added, a project opened) are queued with `ui.flash()` and drawn at the top of the next run, so they are visible on every supported Streamlit version.
- **Rate-table editors** show and accept relativities to 4 decimal places; cells no policy ever fell in are blank and cannot be edited; the pair search reports a dispersion-scaled Pearson z-score and draws the shown pair on the same 8-band grid it searched.
- **Persona e2e runs** (`tests/e2e`, Playwright, opt-in with `EASY_GLM_E2E=1`):
  an actuary's rate review and a data scientist's model comparison drive the
  real app end to end (about 30 s each), including running the exported script
  and comparing it with the downloaded scorer.


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
