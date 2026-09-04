# Changelog

## 0.4.3 (2026-09-04)

### Cleaner workbench startup

- The workbench no longer shows Streamlit's first-run email prompt or attempts
  to submit an email address. This avoids a misleading SSL traceback on
  restricted Windows networks.
- The workbench can now be opened directly from Python with
  `easy_glm.launch_workbench()`. Pass an in-memory Polars or pandas dataframe
  with `easy_glm.launch_workbench(data=df)` to open it in the workbench.
- Stopping the terminal launcher with Ctrl+C now closes it without a Python
  traceback.

## 0.4.2 (2026-09-03)

### A complete browser-workbench path

- The normal `pip install easy-glm` installation now includes the browser
  workbench, charts and rate-table plotting dependencies. There is no separate
  UI or visualisation extra to remember.
- `easy-glm-workbench` opens the workbench in the browser (normally at
  `http://localhost:8501`). The README now says to keep the terminal open,
  explains how to reopen a saved project and points first-time users to the
  in-app French motor sample, which downloads once and is cached locally.
- The Project & data page offers that sample as a guided starting point and
  the workbench now explains the project state, setup progress, comparison
  defaults, fitted versus working tables, and common modelling controls in
  pricing language.
- The public examples and README were streamlined around the frequency-model
  workflow; the release gate continues to execute their runnable examples.

## 0.4.1 (2026-09-03)

### A shorter path from first model to review

- Reworked the README for a practising actuary: a first frequency model, a
  controlled design-and-validation workflow, review of a saved scorer, then
  the workbench and specialist recipes. The first-use path now explains the
  required policy-period data, the frequency calculation, base rate,
  relativities, A/E and the `Other / Unknown` fallback in plain language.
- Reorganised examples into a learning sequence. `basic_usage.py` fits and
  saves a scorer; `exploring_fit.py` and `scoring_editor.py` consume that
  saved scorer without refitting; `easy_glm_demo.py` creates a workbench
  project. Rate change, lapse and scale remain clearly labelled specialist
  recipes.
- Added a focused documentation test that verifies the saved scorer passes
  from the first example to the review and scoring examples.

## 0.4.0 (2026-09-03)

### The README is a release gate now (R11)
- **Every code block on the README runs, checked by a test.** In 0.3 the
  README examples did not all run against the released package, which cost
  trust; `tests/test_readme.py` now extracts every fenced ```python block
  from `README.md`, in order, and executes them in one namespace (so the page
  reads as a tutorial — later blocks reuse `df`, `spec`, `fit`, `rm`,
  `project` from earlier ones), plus every `examples/*.py` via `subprocess`.
  A block that genuinely needs a browser or the workbench server is fenced
  ```python skip-test``` and capped at a handful, so a block cannot be
  quietly exempted by mislabelling it. The whole check runs in about 20
  seconds on the checked-in 50k-row fixture — see
  [`docs/checks/readme-gate.md`](docs/checks/readme-gate.md)
  (`scripts/checks/readme_gate.py --write`), regenerated from a real run.
- **The README was rewritten end to end against the final 0.4 API**, in the
  order a pricing actuary would work: install, load the checked-in French
  motor sample, design a model (step by default, a categorical, a linear
  term, a monotone constraint), fit at a fixed alpha (with cross-validation
  named as the more careful, slower alternative), read the rate tables, add
  an interaction (mains frozen, cells as pure adjustments), smooth and cap a
  factor and rebalance, the exact-scoring invariant and the `.easyglm`
  round-trip, the Excel export, a book of a few hundred thousand rows on the
  compact design path, a `Project` file and its exported Python script, lift
  / Gini / A/E / double-lift diagnostics, a rate-change model with the
  target-loss-ratio solver, a lapse (binomial) model, the browser workbench,
  the `easy-glm` command line, and the project-file/script round trip. Every
  feature the 0.4 CHANGELOG lists above is demonstrated running, not just
  described.
- **`examples/` brought up to the final API**: the six existing scripts now
  read the checked-in fixture instead of downloading the full dataset, and
  three are new — `rate_change.py`, `lapse_model.py` and `large_book.py`
  (`--rows`, default 300,000, above the compact-path threshold). Two
  pre-existing examples were quietly computing a meaningless "per-variable
  A/E" by summing `ae_by_variable`'s bins back into the overall A/E (the
  bins partition the same rows, so the sum always reproduces it); both now
  report the spread of A/E across bins instead, which is what the diagnostic
  is for.
- `pyproject.toml` bumped to `0.4.0`.

### Books of millions of rows fit in memory (G)
- **A 5-million-row book with a 227-column rating structure now fits and trains
  in 2.6 GB and 21 seconds** on a 24 GB laptop; 1M rows takes 0.8 GB and four
  seconds. In 0.3 the design matrix was written out in full — one float64
  column per band per row — which is 9 GB of design alone at 5M rows, and the
  machine could not do it (`docs/checks/g-scale.md`,
  `scripts/bench_scale.py`).
- **How.** A banded factor's columns are completely described by one number per
  row — which band the row is in — so that is what is stored: a new
  `core/stepmatrix.py` holds a `StepMatrix` tabmat block of `int32` bin indices
  and computes everything the solver asks for (matrix-vector products, the
  gradient, the Hessian, cross-products with the other blocks) straight from
  them with a cumulative-sum trick. Categorical factors become tabmat
  `CategoricalMatrix` blocks and an interaction becomes **one** categorical
  block over its kept-cell code, with "no kept cell" as the dropped category —
  four bytes a row however many cells it has, instead of the dense
  `rows x cells` block piece A built. `DesignSpec.build(data, sparse=)` returns
  the resulting `SplitMatrix`; **memory now grows with the number of factors,
  not the number of bands**.
- **The design bytes are arithmetic you can check**: `n x (4 per banded factor
  + 4 per categorical + 4 per interaction + 8 per missing-value indicator + 8
  per piecewise-linear band)`, asserted against the real matrix on every
  benchmark run (`DesignSpec.expected_design_bytes`). Piecewise-linear bands
  stay dense on purpose — their columns are real-valued, so a band index does
  not determine them — and the formula says what that costs.
- **No number moved.** float64 everywhere (`P1`, bounds, weights and the offset
  are cast); the compact and dense fits give the same non-zero set and
  predictions agreeing to 1e-10 — measured 3e-14 on the French motor fixture,
  with the same cross-validated alpha — across fixed-alpha, cross-validated,
  monotone, two-stage, piecewise-linear and continuous designs and on a 300k
  synthetic book (`tests/test_scale.py`). The compact form switches itself on
  at **200,000 rows**; the 50k golden fit is below that and still fitted on the
  dense matrix, which the test asserts explicitly.
- **Scoring never builds a design matrix.** `GLMFit.predict` and
  `linear_predictor` now add up one rate-table lookup per factor in 500,000-row
  chunks — the same arithmetic `RateModel` does, so the exactness invariant
  holds by construction rather than by coincidence — instead of calling glum on
  a freshly built matrix. Diagnostics on a big book therefore never materialise
  a second copy of the design.
- **New: `fit_glm(..., aggregate=True)`** fits one row per *distinct design
  row* with the summed weight and the weighted mean target. Exact for every
  family easy_glm offers (identical coefficients to 1e-12, tested for Poisson,
  Gamma and Tweedie with weights and with an offset in the grouping key), and
  **off by default**: it pays only on a coarse design, collapsing the 50k
  fixture by just 5 %. Refused together with `cv=`, because folds have to be
  assigned to rows.
- **New: `fit_glm(..., progress=callable)`** reports the stage and the elapsed
  time about once a second from a background thread, and the Model page shows
  it under the Fit button ("Stage 1, main effects — Fitting 1,000,000 rows x
  197 columns — 12s"). It is elapsed time rather than a percentage because glum
  exposes no per-alpha or per-fold hook; a callback that raises can never fail
  a fit.
- **glum is pinned to `3.4.*`.** Its input validation only passes through
  tabmat's own block types, so `install_glum_shim()` wraps one private function
  with a single branch that lets a `StepMatrix` through unchanged (documented
  in `core/stepmatrix.py`; the upstream fix is a one-line `isinstance` check).
  A test fits a two-block matrix through that path so the day glum changes it
  is the day the build goes red, not the day a 5M-row design is silently
  densified.
- Nothing pickled changed shape or meaning, so this piece did not move
  `PERSIST_FORMAT` (D5 and E/F did; the release ships at 7): a
  persisted `GLMFit` still holds the same glum model and the same coefficients,
  and reading it back under the new scoring gives the same predictions to
  1e-15.
- **The two-stage fit works the same on the compact design.** An `offset=`
  array still reaches stage 2 (it is folded into stage 1's linear predictor by
  hand, and the compact path did not change what that returns), an `EasyGLM`
  bundle saved from a compactly fitted `TwoStageFit` reloads and scores
  identically, stage 1's `P1` still does not reach the cells, and the exported
  script — which carries no `sparse=` and therefore takes the row-count default
  — reproduces a run that was fitted through the `SplitMatrix` to 1e-10.

### Interactions are fitted on top of frozen main effects (A2)
- **Adding an interaction no longer moves a single main-effect relativity** (the
  actuary's answer to Q5). A model with interactions is now fitted in **two
  stages**: stage 1 is the main-effect model — the same fit, number for number,
  that the model gets with no interaction at all — and stage 2 fits the
  interaction cells on top of it, with stage 1's linear predictor (plus any
  offset column) as its offset and no intercept of its own. The rate tables and
  the base rate come from stage 1 alone, so they are identical with and without
  the interaction, and every cell is a **pure adjustment**: 1.00 means "no
  adjustment" and nothing else. On the French motor set the joint fit used to
  re-price the youngest `DrivAge` band by −21.1 % and the base rate by 1.5 %
  when `DrivAge × BonusMalus` was added; both are now 0.00 %
  (`docs/checks/a-interactions.md`).
- **What it costs, stated plainly.** The mains can no longer take back part of
  what a cell is carrying, so the cells are larger and a little lift is given
  up: on the same data and the same alpha, holdout Gini 0.3083 and 4.84 %
  deviance explained against the joint fit's 0.3105 / 4.93 % — both still above
  the 0.3072 / 4.79 % of the model with no interaction. That trade is the
  point: an interaction you add, change or remove can never re-price a factor
  you have already signed off.
- **New: `TwoStageFit` and `fit_two_stage`** (both exported from `easy_glm`). A
  `TwoStageFit` *is* a `GLMFit` — the same spec (mains then cells), stage 1's
  coefficients followed by stage 2's, stage 1's intercept — so `rate_tables`,
  `base_rate`, `to_rate_model`, `coef_table`, the Excel export, `.easyglm`
  files and the diagnostics need no special case, and
  `RateModel.predict == exp(eta1 + eta2)` to 1e-10 on nulls in both parents,
  unseen levels, offsets and piecewise-linear parents. `fit_glm` gained
  `offset=<array>` and `fit_intercept=` for the second stage.
- **The exported script writes both stages** — `fit_glm` on
  `spec.main_effects_spec()`, `eta1 = stage1.linear_predictor(train)`, `fit_glm`
  on `spec.interactions_spec()` with `offset=eta1, fit_intercept=False`, then
  `TwoStageFit(stage1, stage2)` — so what runs outside the workbench is what ran
  inside it, cell adjustments included. Whether there really were two stages is
  read off the **fit**, not off the design: an interaction whose every cell is
  below the exposure floor has no columns to fit, so no stage-2 block is written
  (it would have been a fit on a zero-column design, which does not run). A
  script exported *before* fitting cannot know that yet — only the data can say
  which cells clear the floor — so it calls `fit_two_stage`, which decides at
  run time, and each interaction is now written out with its own cell floor and
  penalty weight instead of the shared defaults.
- **`EasyGLM.save` / `load` handle a two-stage fit** (bundle version 3): both
  glum estimators are written and the pair is rebuilt on load. Before, the
  bundle held stage 1's estimator against the composed mains+cells spec and the
  first prediction after loading raised.
- **The cell penalty rule is unchanged on the product path, and corrected off
  it.** `P1 = penalty_weight × 0.5 / sd` under glum's standardisation and
  `penalty_weight × 0.5` without it are the *same* penalty per unit of
  adjustment, which is what lets stage 2 — where glum refuses to standardise,
  because there is no intercept — penalise a cell exactly as the joint fit did.
  Getting there meant changing `penalty_weights`' **unstandardised** branch from
  `penalty_weight` to `penalty_weight × 0.5`: the two branches used to disagree
  by a factor of two, and 0.5 is the value R3 specified. Every path the
  workbench, the exported script and `EasyGLM` take standardises, so no fitted
  model moves; a caller who passed `scale_predictors=False` to `fit_glm` on a
  design with cells by hand will find those cells penalised half as hard as
  before and should halve their `alpha` to reproduce an earlier fit.
- **The second stage has its own alpha.** It defaults to the mains' alpha (a
  cell then costs what a main effect that half the exposure shares costs), and
  cross-validates on its own path when the mains do. `Interaction.alpha`
  overrides it; per-interaction differences still belong in `penalty_weight`,
  because the second stage is one fit.
- The Model page says which model was fitted in two stages, with the alpha of
  each and how many cells were rated, and shows a regularisation path per stage;
  when an interaction is present but **no** cell cleared its exposure floor it
  says that instead, rather than falling silent about a matrix of 1.000s. The
  Design page offers each interaction's *cells alpha*, and the Rate tables page
  says the base rate comes from the main-effect fit alone. `run.summary()` and a
  saved snapshot's metrics carry both alphas.
- **A cell is a pure adjustment to the mains, but not purely an interaction.**
  Stage 2 has no intercept — that is what pins the base rate — so any overall
  re-levelling it wants goes into the cells. On the French motor check that is
  0.38 % of each adjusted cell, and holdout A/E moves 1.0191 → 1.0223 with the
  interaction added. Both are now stated in `docs/checks/a-interactions.md` and
  on the Model page; a base-rate override moves the level back in one number
  without touching a relativity.
- **Fits cached in a `*.easyglm-runs` folder by an earlier 0.4 development build
  are ignored and refitted** (`PERSIST_FORMAT` 4 → 7, see D5 and E/F below): such a run
  holds a joint fit whose main tables include part of the interaction — the same
  shape, a different meaning.

### Relativity tooling in the rate-table editor (D5)
- **A *Tools* panel above the editor**: smooth a curve (moving average over a
  window of bands, or an isotonic fit that will not let it turn back), cap and
  floor it, or round it to decimals or to a step such as 0.05. Every tool shows
  what it would do — the curve before and after, the bands that would change and
  the level check — before anything is applied, and writes the result as
  **ordinary manual adjustments**, so the project file stays the truth and the
  tables are rebuilt from the fit without refitting.
- **Smoothing preserves the exposure-weighted mean of the *log* relativities**
  (plan §R6), to 1e-12 — the *shape* rule: the moving average is re-centred to
  achieve it, the weighted isotonic fit preserves it by construction.
- **The panel reports what a tool does to the money, separately.** Preserving a
  mean of logs is not preserving the premium: a premium is a product of
  relativities and a book is the sum of those products, so every tool — a
  smoothing included — moves total expected claims (a 3-band moving average on
  DrivAge takes 0.57 % off the French motor book; a cap at 3.00 on BonusMalus
  takes 4.86 %). The Tools panel therefore shows **the change in total expected
  claims on the training rows**, measured by scoring both sets of tables, and it
  says "no change" only when that change is zero to 1e-9.
- **Rebalance base rate**: one click sets the base-rate override so total
  expected claims on the training rows are exactly what the fitted model
  expected — the off-balance correction of a rate review — without touching a
  relativity. The page shows the current off-balance whenever a model has been
  edited.
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
  step per edit, tool, reset, rebalance or restored snapshot. A step is the
  model's whole post-fit state — the adjustments **and the base-rate override**
  — so undo restores the previous tables exactly, level included (a snapshot
  carries a base rate, and restoring one used to leave that base rate in force
  after an undo).
- **Snapshots**: *Snapshot as…* names the tables as they stand and keeps them in
  the **project file** (a named list of adjustments plus the base rate,
  `ModelConfig.snapshots`), so they survive a reload and a refit; snapshots can
  be restored, deleted and **compared** — the same table the Compare page shows
  for two models (`workflow.rate_model_diff` / `snapshot_diff`, and
  `RateModel.diff` for two versions of one model). A snapshot that no longer
  fits the model (it adjusts a factor the model has lost) is **refused by name
  and changes nothing**, rather than half-applying and tracebacking the page;
  removing an interaction now strips its cell adjustments from every snapshot as
  well as from the working set (`ModelConfig.drop_adjustments_for`); and
  deleting a snapshot asks twice, because it is the one action undo does not
  cover.
- An adjustment naming a variable the model does not have is now an
  `AdjustmentError` rather than a bare `KeyError`, so the workbench drops it and
  says so instead of showing a traceback — the same treatment a stale *band*
  already got (this also fixes the pre-0.4 crash after deleting a predictor that
  carried an adjustment).
- Engine: `easy_glm.engine.tooling` (pure functions on one `VariableConfig`,
  returning the relativities a tool would set, plus `preview_model` for pricing
  one before applying it); `workflow.expected_claims`,
  `workflow.rebalance_override`, `workflow.missing_variables`; and
  `workflow.rate_model_for`, which compiles "this fit plus this list of
  adjustments" into tables — what a snapshot, a restore, a rebalance and a diff
  all use. `app.state.PERSIST_FORMAT` moves on (rate-table rows changed shape here, and
  a model with interactions became a two-stage fit in A2), so runs cached by an
  earlier 0.4 build are refitted.
- Tests: `tests/test_d5_tooling.py` (engine unit tests per tool, the exposure
  plumbing from fit to JSON, exactness after a tool, and the page's apply / undo
  / redo / snapshot / diff through AppTest); plain-language page in
  `docs/checks/d5-tooling.md` (`scripts/checks/d5_tooling.py --write`).

### Rate reviews, modelling extras and a command line (E / F)
- **Fit the change from the premium you charge today.** Give a column the role
  **current premium** on the Variables page and easy_glm derives `log_<premium>`
  and pre-fills it as the offset of every new model. The rate tables are then
  **multipliers on the current premium** — the base rate carries the level, each
  relativity is a differential change — and the Rate tables page, the Export
  page and the Excel `Summary` sheet all say so, so a multiplier cannot be
  misread as a rate. The derivation is written into the exported Python script
  as a line of polars rather than left implicit in a role. Rows whose premium
  has no logarithm (zero, negative, missing) are refused by name and count, with
  the row filter to add; the filter runs first, so `pl.col('Premium') > 0` is the
  fix and not a trap. Renaming the premium column carries every model's offset
  with it; taking the role away clears them with a notice.
- **A target loss ratio, solved.** `workflow.solve_base_rate(run, df, ratio)` and
  a box on the Model page set the base rate so that total actual ÷ total expected
  equals the number you type. For a rate-change model that ratio *is* the loss
  ratio the book would be written at; for an ordinary model 1.00 rebalances it
  (overall A/E exactly 1). Closed form — one pass over the rows, no search — and
  it reads the model's *current* base rate, so an existing override cancels out
  and solving twice gives the same answer. The relativities never move. Binomial
  models are refused: a probability is not proportional to the base rate.
- **A penalty weight per factor.** `VariableDesign.penalty_weight` (a column on
  the Design page, `DesignSpec.from_data(penalty_weight={...})`) multiplies the
  per-column rules `core/fit.py::penalty_weights` already applies: 2 shrinks a
  factor twice as hard as the rest of the design and **0 leaves it unpenalised**,
  so every level of a territory table you have committed to survives the lasso.
  On the check's book a heavy penalty leaves 8 of 20 regions; at weight 0 all 20
  stay. It weights the L1 penalty only.
- **Tweedie power on the Model page.** `ModelConfig.tweedie_power` /
  `fit_glm(tweedie_power=...)`, strictly between 1 and 2, default 1.5, saved with
  the model and written into the exported script. Passing it for another family
  is an error rather than a silent no-op.
- **Binomial models have rate tables now.** `log` and `logit` are both
  multiplicative links: a lapse or conversion model compiles to the same tables
  read as **odds relativities** (labelled that way on every page and in Excel),
  its base rate is the base risk's odds, and the scorer converts back, returning
  a probability that matches the GLM to 1e-16. Because a probability is not an
  amount, such a model **refuses** to be multiplied by exposure — in
  `to_rate_model`, in `RateModel.predict`, and in the workbench, which never
  hands it an exposure column. `rate_tables` on a binomial fit used to raise;
  a genuinely non-multiplicative link (identity) still does.
- **`easy-glm` on the command line.** `easy-glm run project.json [--model NAME]
  [--out DIR]` fits and writes all four artefacts — the `.easyglm` scorer, the
  Excel rate tables, the runnable Python script and the self-contained HTML
  report — then prints rows, alpha, base rate and train/holdout A/E, Gini and
  deviance explained. `easy-glm export --script | --report | --excel` writes any
  subset, `easy-glm validate` checks a project *and its data* without fitting,
  and `easy-glm workbench` opens the browser tool. Every artefact command fits
  afresh, which is what makes the exported script self-contained. Nothing ever
  prints a traceback: problems are messages with exit code 1, so a scheduled job
  can tell success from failure. This closes the CLI half of D4.
- **`mypy` on `core` and `workflow`** is a CI step (`--ignore-missing-imports`).
  It found 30 problems; all are fixed rather than silenced except two
  `type: ignore` on polars scalars. Three mattered: a model with no target
  reached `diagnostics.unit_values` / `totals` and failed inside polars instead
  of saying so; `single_factor_strength` and `run_model` passed a possibly-`None`
  target to `fit_glm`; and `EasyGLM.blueprint` asked every encoder for `.levels`,
  which a piecewise-linear term does not have.
- `app.state.PERSIST_FORMAT` is **7** at the end of 0.4: encoders, `ModelMetadata`
  and `ModelConfig` each gained a field here, rate-table rows gained exposure in
  D5, and A2 made interaction models two-stage fits; each piece bumped the number
  on its own branch and the merged tree moved on past all of them. Any run
  pickled by an earlier 0.4 build is a cache miss, never a half-built object;
  the comment in `app/state.py` lists every reason.
- **A rate change and an interaction together.** A model that offsets on the
  current premium *and* has an interaction is fitted in two stages like any
  other: the base rate and main tables come from stage 1, the cells from stage 2,
  the RateModel applies the premium offset on top, and `RateModel.predict`
  reproduces the GLM to 1e-10. The main tables are identical with and without
  the interaction, offset and all; `solve_base_rate` works unchanged (the
  prediction is still proportional to the base rate); and the two-stage exported
  script carries the premium derivation, the offset column, the
  `offset_is_premium` label and the Tweedie power in **both** stages.
- Tests: `tests/test_e_f_extras_cli.py` (83) — including the offset identity of
  plan §R6/S1 measured at 5.6e-12 (Poisson, `scale_predictors=False`, alpha
  × Σ premium ÷ n) with the Gamma case recorded as *not* matching, and the CLI
  driven end to end through `subprocess`. Plain-language replay:
  `docs/checks/e-f-extras-cli.md` (`scripts/checks/e_f_extras_cli.py --write`).

### Known limitations in 0.4.0 (the 0.4.1 / 0.5 backlog)
Items the independent reviews accepted as not worth a further round before the
release. None changes a number a model produces.
- **Scale**: piecewise-linear bands are stored dense (8 bytes per row per band);
  the `(band, overlap)` pair form is the fix if it ever binds. There is no
  per-penalty progress callback (glum 3.4 has none), only elapsed time.
  `aggregate=True` is refused with cross-validation. Data is not loaded lazily;
  a 5M-row book needs the frame in memory. The glum shim's install check is not
  lock-protected (a benign double-wrap if two fits start at once).
- **Interactions**: the cells-alpha box shows `0.00000` for alphas below 5e-6
  (the stored value is kept); `Interaction.alpha` has no upper bound in
  `validate`; a `RateModel` built from a fit given an offset *array* (not a
  column) does not carry that offset — use `offset_col`.
- **Rate reviews**: the Excel `Summary` wording "current multiplier on current
  premium" stumbles; `solve_base_rate` on a target with nulls reports a
  non-positive total rather than naming the nulls; there is no `easy-glm score`
  command (scoring new business is a two-line Python example in the README).
- **Tooling**: smoothing preserves the mean log relativity (plan R6) and shows,
  but does not automatically correct, the change in total expected claims — open
  question Q17 for the owner.
- **Docs and packaging**: `aggregate=`, `progress=`, `penalty_weight` and
  `tweedie_power` are documented in the check pages rather than on the README;
  `elastic-net` and `tabmat` are used without a gloss; the CI matrix is 3.10–3.13
  (3.14 untested); there is no documentation site beyond the README and
  `docs/`.
- The runs folder and multi-tab conflict detection are per machine; nothing
  coordinates two people editing one project on a shared drive.

### Hand-edited project files (W5) — the third breaker session
- The third break-it session targeted the surfaces added late in 0.4: Compare,
  the HTML report, the rate-table tools and snapshots, the rate-change flow,
  penalty weights, Tweedie and binomial models, the cells alpha, the CLI and the
  compact-matrix path. Those all held. What broke was **a project file edited by
  hand**: a non-numeric or out-of-range `alpha`, `tweedie_power`, `l1_ratio`,
  `cv`, `n_alphas`, `clamp`, `split.fraction` or an interaction's `alpha` /
  `penalty_weight` / `min_cell_exposure` crashed `Project.validate()`, the CLI
  (raw traceback), and the Model and Design pages — and once the pages rendered,
  they **silently autosaved a fallback number over the value in the file**. A
  legal number outside a widget's range (`alpha: 1e9`) made the Model page rerun
  forever.
- Fixed: `Project.validate()` reports every such field as a problem string
  instead of raising (`l1_ratio` must be in [0, 1], `cv` ≥ 2 or absent,
  `n_alphas` ≥ 2); the CLI prints them and exits 1; the pages **repair and say
  so** — a notice names the field, the value found and the value used, before
  anything is saved (`ui.repair_number`, the pattern the Split page already used
  for its seed); `ui.number_in_range` repairs the stored value before building
  the widget, so it cannot loop. Nothing a model computes changed
  (`core`/`engine` untouched; golden green).
- Tests: `tests/test_w5_breakage.py` (36); record in
  `docs/reviews/w5-breakage-3.md`, independent review in
  `docs/reviews/w5-breakage-3-review.md`.

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
