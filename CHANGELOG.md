# Changelog

## 0.473 — 11 October 2026

- Added single-model and staged-model routes. Each stage has its own variables
  and fits on the complete score from earlier stages.
- Grouped interaction tables with their GLM so they can be added and reviewed
  after fitting the main effects.
- Allowed CatBoost interactions after a log-link Gaussian GLM.
- Added keyed lookup models, including scores for keys absent from training
  data.
- Added diagnostics for the cumulative score through a selected stage or the
  full sequence.
- Added scoring of new datasets, with CSV and Parquet downloads.
- Added exports for fitted model sequences: Python refitting scripts, saved
  scorers and Excel component workbooks.
- Added optional AGLM numeric curves with automatic or chosen grids and flat or
  extended tails. Curves carry through tables, amendments, saved models and
  Excel/Python exports.
- Added interaction-size previews and coarser table bands without changing a
  GLM curve.
- Fixed the Continue button when the next model already has variables set.
- Made one-way selectors searchable and alphabetical.
- Fixed rare binary bands and rounding for very small ranges.

## 0.472 — 23 September 2026

- Added the interactive pricing workflow and worked example: GLM fitting, A/E,
  missing-factor checks, interactions, rate amendments and Excel export.
- Included CatBoost and Optuna in the standard installation.
- Added family-appropriate metrics and clarified that Gini measures ranking,
  not prediction accuracy.
- Estimated variable importance on 30% of training rows by default, with an
  adjustable percentage. Small or sparse samples use all training rows;
  model fitting is unchanged.
- Fixed Gaussian screening with negative targets.
- Fixed Windows access-denied errors from fit-progress updates.
- Added categorical and mixed-variable report associations and clearer skipped
  checks.
- Allowed unassigned variables in interactions without making them GLM main
  effects. Changing a variable to unassigned keeps its bins and interaction
  tables.
- Added a tuning time limit for each CatBoost interaction. GLM fitting and the
  final refit have no time limit.

## 0.471 — 18 September 2026

- Fixed access-denied errors in feature selection by removing temporary files.

## 0.470 — 18 September 2026

- Added training-only GLM screening against shuffled variables and random
  noise, with ranked and sortable results.
- Fitted CatBoost pairs sequentially on main effects and earlier interaction
  tables, with cross-validation tuning and fixed main effects.
- Kept chosen interaction bins in corrections, later interactions, diagnostics
  and exports.
- Added three-decimal interaction heatmaps and live bin-distribution previews.
- Exported screening settings, candidates and reviewed choices to Python.
- Excluded existing pairs from interaction searches and clarified review
  navigation, fit status and variable counts.

CatBoost interactions needed separate training dependencies when introduced;
they have been in the standard installation since 0.472.

## 0.464 — 18 September 2026

- Added default, per-variable and explicit numeric bins, with training counts
  and missing values previewed before application.
- Stored binning in the Role JSON editor and kept it synchronised with controls.
- Preserved pending Variables previews during background model updates.

## 0.463 — 11 September 2026

- Added smaller challenger models from selected importance results and time
  stability checks with optional factor breakdowns.
- Reused compatible main-effect and cross-validation fits when interactions
  changed.
- Clarified train/holdout assignment, including Role JSON and two-value splits.
- Reset prior models and fits when loading examples, and improved Compare and
  wide-screen layout.

## 0.462 — 11 September 2026

- Added training permutation importance, coefficient paths and selected-lambda
  markers to reports, with separate main-effect and interaction-cell paths.
- Added training-data summaries, distributions and numeric correlations to
  reports.
- Used a reproducible sample of up to 10,000 rows for correlations.
- Shortened the README and linked detailed tutorials separately.

## 0.461 — 10 September 2026

- Added configurable checks for possible leakage, related predictors and mostly
  missing columns, using up to 10,000 training rows without fitting each
  predictor.
- Made a check flag prompt review rather than prove leakage.
- Added review-before-apply removals and kept role JSON, factor selections and
  saved adjustments in sync.
- Cleared stale screening results after settings changes and prevented scanning
  or selecting flags from changing the project automatically.

## 0.460 — 10 September 2026

- Made the Svelte workbench the default for `easy-glm-workbench`,
  `easy-glm workbench` and `easy_glm.launch_workbench()`.
- Kept the legacy Streamlit interface available with `--legacy-streamlit` or
  `legacy_streamlit=True`.
- Added file, project and French motor/Swedish motorcycle example loading.
- Kept loaded examples editable and unfitted. Failed loads preserved the
  current work.
- Added training-data one-way effects, A/E and exposure review, regularisation
  paths, comparisons and permutation importance without refitting.
- Added second-stage interactions with fixed main effects, adjustable cell
  exposure and penalty.
- Used final main effects for final coefficients in cross-validated interaction
  fits. Saved fits needed an explicit refit to use the correction.
- Added previewed table adjustments, smoothing, caps, rounding, manual edits,
  Undo, snapshots and clean new fits. Moving averages used the current point
  and the preceding N−1 points.
- Added Excel, JSON scorer, project JSON and HTML exports. Reproduction scripts
  required saved source data and reproduced interaction fits and weighted A/E.
  Unsaved previews were excluded.
- Improved chart colours, labels and displayed precision.

Some advanced preparation, leakage and factor-design controls remained in the
legacy interface. Project JSON kept settings, applied adjustments and named
snapshots. Reopening did not restore fitted runs or session Undo history.

## 0.452 — 8 September 2026

- Listed every role in generated JSON, including empty and ignored roles, and
  kept JSON and table reset behaviour consistent.
- Avoided calculating hidden role-table distinct counts during wide JSON edits.

## 0.451 — 8 September 2026

- Kept role tables, JSON, factor selections and factor-design tables in sync.
- Retained an unapplied JSON draft when another update replaced it.
- Made residual suggestions assign the needed predictor role and retain
  suggested interactions.
- Required a target, predictor and valid split before later pages opened, and
  returned direct links to locked pages to Variables.
- Added compact proposed-edit feedback and clearer validation guidance.
- Improved startup under IDE Python paths and kept running sessions stable
  during updates.
- Required a restart after upgrading.

## 0.4.5 — 7 September 2026

- Added copy/paste JSON for roles, renames and type overrides, with proposed
  changes and validation before application.
- Put normal data loading ahead of optional examples and reduced sidebar
  duplication.
- Limited exports to completed, current fits.
- Moved splitting before Explore so exploration used training data and holdout
  remained separate.
- Made the first fitted model the champion and allowed diagnostics suggestions
  to add factors or interactions for review and refitting.

## 0.4.4 — 5 September 2026

- Added French frequency and Swedish Tweedie starter datasets with editable
  roles, a 70/30 split and an unfitted starter model.
- Added a warned restart path for changing samples and expanded walkthroughs.
- Combined factor design, interactions, fitting and results on the Model page.
- Kept residual-factor searches on training rows and excluded Ignore and ID
  columns, including leakage fields.
- Shuffled cross-validation folds reproducibly and used out-of-fold main-model
  predictions for interaction validation.
- Made missing-factor rankings account for statistical noise, used the correct
  null benchmark for offset models, and prevented a missing-value row becoming
  the base risk.
- Kept fitted models through page changes and reloads, compared challengers on
  compatible bands, and used a null model for double lift without an incumbent.
- Kept original fitted predictions beside adjusted predictions and actual
  experience during rate-table reviews.
- Clarified project files, custom knots, interactions, base-rate choices,
  model creation and train/holdout views.
- Refitted stale cached development fits when their statistical meaning changed.

Ideas intentionally left for later releases are in
[`docs/FUTURE_RELEASES.md`](docs/FUTURE_RELEASES.md).

## 0.4.3 — 4 September 2026

- Added `easy_glm.launch_workbench()` for Python, including in-memory pandas or
  Polars data.
- Removed the misleading Streamlit first-run email/network prompt and made
  Ctrl+C exit cleanly.

## 0.4.2 — 3 September 2026

- Included the workbench and charts in normal installation, added
  `easy-glm-workbench` and the first French motor example, and clarified
  project state, setup progress, fitted versus working tables and comparisons.

## 0.4.1 — 3 September 2026

- Reworked the README around a claim-frequency model, organised examples into
  a learning sequence and added a check that documented examples run.

## 0.4.0 — 3 September 2026

- Added the workbench workflow for data roles and preparation, train/holdout
  splitting, factor design, fitting, diagnostics, comparison, table review and
  export.
- Added step, categorical, continuous and piecewise-linear factors, monotone
  constraints and two-way interactions with fixed main effects.
- Added Poisson, Gamma, Tweedie, Gaussian and binomial models with portable
  rate-table scoring.
- Added smoothing, caps, rounding, manual adjustments, undo/redo, snapshots
  and base-rate rebalancing.
- Added Python, command-line, Excel, scorer and self-contained HTML exports,
  compact fitting/scoring for multi-million-row books, and safer saved-project
  and cached-run handling.

MIT licensed. See [LICENSE](LICENSE).
