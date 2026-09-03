# Changelog

## 0.4.0 (unreleased)

### Relativity tooling in the rate-table editor (D5)
- **A *Tools* panel above the editor**: smooth a curve (moving average over a
  window of bands, or an isotonic fit that will not let it turn back), cap and
  floor it, or round it to decimals or to a step such as 0.05. Every tool shows
  what it would do — the curve before and after, the bands that would change and
  the level check — before anything is applied, and writes the result as
  **ordinary manual adjustments**, so the project file stays the truth and the
  tables are rebuilt from the fit without refitting.
- **Smoothing preserves the exposure-weighted mean of the *log* relativities**
  (plan §R6), to 1e-12: the base rate is not refitted when a table is edited, so
  a smoothing that moved that mean would move every premium the factor touches.
  The moving average is re-centred to achieve it; the weighted isotonic fit
  preserves it by construction. A cap, a floor and a rounding move the level on
  purpose, and the panel says by how much.
- **Rate-table rows now carry their training exposure** (`FromToRow.exposure`,
  `BandRow.exposure`, from the new `GLMFit.row_exposure` — the count
  `_modal_bins` already took its argmax of). It is what the tools weight a band
  by, it is a column in the editor, in `rate_tables` and in the Excel export, and
  it tells "no data" apart from "no effect" when a relativity reads 1.00. Older
  files load with 0 and the tools then weigh every band the same, and say so.
- **The null / Other row is never touched by any tool**, and a **categorical**
  factor is not smoothed until the user confirms that its levels read in order
  (they are listed most-exposed first, which is not an order of the risk). A
  **piecewise-linear** table is smoothed at its *nodes* and the slopes are
  re-derived, so the curve stays continuous.
- **Undo / Redo** on the Rate tables page: 50 steps per model per session, one
  step per edit, tool, reset or restored snapshot; a step is the whole set of
  adjustments, so undo restores the previous tables exactly.
- **Snapshots**: *Snapshot as…* names the tables as they stand and keeps them in
  the **project file** (a named list of adjustments, `ModelConfig.snapshots`), so
  they survive a reload and a refit; snapshots can be restored, deleted and
  **compared** — the same table the Compare page shows for two models
  (`workflow.rate_model_diff` / `snapshot_diff`, and `RateModel.diff` for two
  versions of one model).
- Engine: `easy_glm.engine.tooling` (pure functions on one `VariableConfig`,
  returning the relativities a tool would set); `workflow.rate_model_for`
  compiles "this fit plus this list of adjustments" into tables, which is what a
  snapshot, a restore and a diff all use. `app.state.PERSIST_FORMAT` 4 → 5
  (rate-table rows changed shape), so runs cached by an earlier 0.4 build are
  refitted.
- Tests: `tests/test_d5_tooling.py` (engine unit tests per tool, the exposure
  plumbing from fit to JSON, exactness after a tool, and the page's apply / undo
  / redo / snapshot / diff through AppTest); plain-language page in
  `docs/checks/d5-tooling.md` (`scripts/checks/d5_tooling.py --write`).

### The persisted-run folder is shared state (W4) — the second breaker session
- **No tab may throw away another tab's fit.** Fits live in
  `<project>.easyglm-runs/` next to the project file, and every browser tab with
  that project open writes into the same folder. Three rules now govern it: a
  tab showing the conflict notice may fit but writes and deletes nothing (the
  page says the result is kept in this tab only, and *Delete* removes the model
  from that tab's project alone); nothing is deleted while a tab is out of step
  with the project file on disk; and the "latest run per model" tidy-up never
  removes the fit that matches the project *as saved on disk*, nor the one
  matching this tab's own spec. Before this, a tab that was one edit behind
  deleted the fit belonging to the saved project, silently.
- **A fit that never finished says so.** A marker is written next to where the
  result will be saved and removed once it is there; a session that finds a
  marker with no result reports "a fit of X was interrupted … fit it again"
  instead of showing a model that looks as if it was never fitted.
- **Create now switches to the model it created** — the picker, the whole
  configuration panel and Fit follow, so a note or a predictor change meant for
  the new model can no longer land on the champion.
- **A constant predictor no longer blocks the fit**: a column that is constant
  or all-null on the training rows is left out of the design and named on the
  page (`workflow.UnusableColumnError`, `build_design(..., dropped=[])`); every
  other design problem is still an error, because the user can act on it.
- **Number boxes never name a number the fit did not use.** A value outside the
  allowed range (a pasted `1e9` alpha, a typed seed of -5) is refused with a
  message and the box is put back to the value in the project, instead of the
  browser keeping the typed text on screen.
- Smaller fixes from the same review: the "Could not persist the fit" banner
  clears once saving works again; an orphaned sidecar whose pickle was deleted
  by hand is removed; the status chips no longer say "✓ Fitted" next to "refit
  to update"; the *Divide target by weight* box cannot stay ticked while it is
  disabled; a knot above the largest training value is accepted but flagged;
  Windows device names (`CON`, `NUL`, `PRN`, `AUX`, `COM1`…) are refused as model
  names; picking a modelling column as the train/holdout indicator asks first;
  a training fraction outside 0.50–0.95 in the project file is still repaired but
  the page now says what it changed; saving a project creates the file but never
  the folders around it (a typo is a message, not a new folder tree); the Model page names the interaction
  *parent* that left the predictor list; "1 rows" reads "1 row" and the
  *Prepared* chip goes out for a frame with no rows.
- From the W4 review: the "fit in progress" markers follow the same rules as the
  fits themselves (a paused tab removes none, and no tab removes another
  session's marker until it is five minutes old, since it may be a fit still
  running); the "removed from this tab only" notice is queued before the save,
  so it survives the rerun that the conflict raises; the check page says that a
  fit running in another tab may be reported as interrupted; deleting is also
  paused while the project file is missing; the project file is written to a
  temporary file and renamed, so no reader sees half of it; and the knot
  warning reads as English.
- Tests: `tests/test_w4_runs_folder.py` (one per finding, two AppTest sessions
  for the two-tab cases) and the plain-language replay in
  `docs/checks/w4-runs-folder.md` (`scripts/checks/w4_runs_folder.py --write`).

### Piecewise-linear terms: flat unless the data insists (B2)
- **The basis now penalises slopes, not bends** (the actuary's answer to Q3).
  A `LinearEncoder` builds one column per band — `clip(x - k_j, 0, band width)`,
  the amount of the value inside that band — instead of the AGLM hinges
  `max(x - k, 0)`. Each fitted number is therefore the **slope inside one band**
  and the lasso drives it to exactly zero, so a stretch the data does not argue
  about comes back perfectly flat rather than gently sloped. The curve is still
  continuous by construction and still flat outside the clamp range; the rate
  table is unchanged — `(from, to, relativity at from, slope)` — except that the
  `slope` column is now the coefficient itself rather than a cumulative sum, so
  Excel, the editor, the exported script, `.easyglm` files and
  `RateModel.from_rate_tables` all keep working unchanged. On the French motor
  set the `BonusMalus` curve improves from Gini 0.3072 / 4.79 % deviance
  explained (the step design) to 0.3103 / 4.94 % and two of its nine bands come
  back exactly flat (`docs/checks/b-linear.md`). The hinge basis earlier in this
  release reached 0.3091 / 4.88 %.
- **Every band pays the same penalty for the same rise.** glum penalises the
  standardised coefficient, so a wide band that few policies reach used to buy a
  large rise in relativity for a small penalty: on the French motor set the top
  bonus-malus band's rise cost about 4 % of what the first band's cost, which
  left the thinnest, least trustworthy part of a curve the *least* penalised part
  of it. Band columns now carry a `P1` weight (`core/fit.py::penalty_weights`,
  the rule interaction cells already had) that equalises the cost per unit of
  rise. The columns themselves are untouched, so a band's coefficient is still
  exactly its slope. On the French motor set this brings the bonus-malus
  relativity at 230 down from 89× to 30× with the holdout essentially unmoved
  (Gini 0.3106 → 0.3103).
  **This makes a given `alpha` stronger on these terms, not merely different**:
  no band's weight goes below 1, so a term's total penalty rises — 1.6× (Density,
  20 bands) to 4.1× (BonusMalus, 9 bands) on the check's own fits, and 3.6× to
  6.3× for a single-band `continuous` term, which has no bands to level against
  each other. A weak trend is therefore now shrunk away where it used to survive
  (the check document's continuous `Density` column is flat for exactly this
  reason); re-check the penalty, or let cross-validation choose it, after
  switching a factor to linear or continuous.
- **The 1.00 point of a `continuous` term is stated rather than incidental**: a
  single band has only two points a rate table can carry 1.00 at, so it sits at
  the lower clamp unless the exposure-weighted median is past 60 % of the way up
  the range. The threshold is off centre on purpose — at a halfway split a factor
  whose median sits near the middle would flip between the two clamps on
  sampling noise, moving every relativity and the base rate between refits for no
  reason. With one slope the choice only rescales the base rate; the ratios
  between relativities do not move.
- **Fits cached in a `*.easyglm-runs` folder by an earlier 0.4 development build
  are ignored and refitted** (`PERSIST_FORMAT` 2 → 3). The basis change altered
  what a `LinearEncoder`'s coefficients *mean* without changing anything about
  the pickle's shape, so such a run would have loaded cleanly and been re-read as
  if its numbers were band slopes. `.easyglm` scorers and project files are
  unaffected and need no migration.
- **Monotone constraints work on piecewise-linear terms again.** A direction is a
  sign bound on every band slope (`increasing` → slope ≥ 0), which keeps the
  curve rising or falling throughout without forcing it convex — the reason the
  constraint was refused in the first place no longer applies. The Design page,
  the Model page and `Project.validate` accept it, and switching a variable to
  *linear* no longer silently drops the constraint.
- **New variable kind `continuous`**: one straight line on the raw (clamped)
  value, no knots. It is the linear encoder with a single band, so it shares the
  rate-table type, the editor, the Excel sheet and the exported script. The
  Design page now offers *auto · step · linear · continuous · categorical* with
  a line of help each. Numeric variables still default to **step** (Q9).
- Planted-truth tests (`tests/test_recovery.py`) now plant a **flat** / sloped /
  **flat** mileage curve and assert that the rate table's slope is exactly zero
  for all nine bands inside the flat stretches and non-zero for exactly the six
  bands of the sloped one, which is recovered within 10 %, plus a monotone case
  where a *decreasing* constraint on a rising curve gives a flat term and never a
  positive slope. The old "bends are sparse" assertion is gone with this basis:
  there are no change-of-slope coefficients left to count.

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
- **Notices survive reruns**: messages shown just before a rerun (an added interaction, a refused adjustment, a filter added, a project opened) are queued with `ui.flash()` and drawn at the top of the next run, so they are visible on every supported Streamlit version.
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
  value along straight-line segments (on the log scale) between knots. (The
  basis first shipped as the AGLM "L-dummies" and was replaced in the same
  release by the per-band slope columns described under *B2* above.)
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
