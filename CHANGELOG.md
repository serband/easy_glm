# Changelog

This is the user-facing record of useful new features and fixes.

## 0.460 — 10 September 2026

- **The new workbench is now the default.** The standard command and Python
  launcher open the Svelte interface. Open CSV, Parquet, Excel or a saved project,
  or pass a pandas/Polars dataframe from Python. The previous interface remains
  available with `--legacy-streamlit`.
- **A clearer modelling workflow.** Set up variables in the table or JSON, define
  main effects and interactions, fit in the background, and compare fitted models.
- **Faster diagnostics.** Cached actual-versus-expected views, permutation
  importance, residual-factor searches, lift and double lift sit alongside
  regularisation paths with retained coefficients on a secondary axis.
- **Adjustments keep the original fit visible.** Choose a tool from the dropdown,
  review its effect and Apply to save. Moving averages use the last N points;
  isotonic smoothing, caps/floors and row edits share Undo and named snapshots.
  A fresh fit starts without inherited adjustments.
- **Complete exports.** Download Excel rate tables, a JSON `.easyglm` scorer,
  project JSON, Python reproduction scripts and HTML reports. Exports include
  applied adjustments and exclude unsaved previews.
- **More reliable scoring checks.** Reproduction scripts use the final frozen
  main effects when fitting interactions and calculate weighted A/E correctly.
  Weighted-binomial null-model fitting no longer uses redundant intercept terms.

## 0.452 — 8 September 2026

### Fixes and improvements

- **Every role is visible in JSON.** Generated settings list all supported roles,
  including unused ones. Empty single-column assignments use `null`; empty role
  groups use `[]`. Ignored columns remain explicit when switching between the
  table and JSON or resetting the editor.
- **Reset uses the current setup.** The reset button replaces the JSON draft
  before the editor renders, without an extra explicit rerun. Changes still
  require the Apply button.
- **Faster JSON editing on wide datasets.** JSON edits and resets no longer
  calculate distinct-value counts for the hidden roles table. Table statistics
  are calculated when the table is shown or needed for automatic role assignment.

## 0.451 — 8 September 2026

### Fixes and improvements

- **Variable edits stay in sync.** The roles table and JSON editor reflect the
  latest applied setup. Model predictor selections and factor-design tables
  refresh after changes elsewhere instead of restoring old values. An unapplied
  JSON draft is retained for reference if another update replaces it.
- **Residual suggestions update the project correctly.** Adding an unassigned
  factor also gives it a predictor role, so the roles table, JSON and model
  agree. Suggested interactions remain part of the selected model and need no
  extra column role.
- **Setup comes before modelling.** Explore and later pages unlock only after
  a target, at least one predictor and a valid train/holdout split are assigned.
  Direct links to locked pages return to Variables. Other columns may remain
  unassigned, ignored or IDs.
- **A quieter Variables page.** JSON schema help is available from the toggle's
  question mark. Proposed edits show a compact column count instead of a large
  preview table; validation and the explicit Apply button remain. Missing model
  settings have actionable guidance where they are needed.
- **More reliable startup.** The launcher avoids Python paths injected by IDEs,
  and startup handles older split-readiness interfaces safely.
- **Stable running sessions.** The workbench no longer reloads package code
  while sessions are active, avoiding import failures during updates. Restart
  the workbench after upgrading EasyGLM.

## 0.4.5 — 7 September 2026

### New features

- **Bulk variable setup with JSON.** On the Variables page, switch from the
  table to a copy/paste JSON editor to assign roles, rename columns and set
  type overrides across large datasets. EasyGLM shows the proposed changes
  first and refuses invalid roles, unknown columns and conflicting names
  without changing the project.
- **Less duplication in the sidebar.** The sidebar now identifies the current
  project and its save state without repeating the save and open controls from
  Project & data.
- **Your data comes first.** Project & data now leads with the normal file
  path, type and upload controls. The French and Swedish starter datasets are
  smaller, optional examples at the bottom of the data-source panel.
- **Exports represent completed work.** The Export page now stays closed until
  a model has been fitted and lists only models whose current specification has
  a valid fit.
- **Split before you explore.** Train/holdout setup now sits on the Variables
  page. Later workflow pages stay locked until both subsets exist, and Explore
  uses training rows only so the holdout remains an honest validation set.
- **Champion status is explicit.** The first model actually fitted becomes the
  champion automatically. Its Model-page control now says **Champion**, while
  an unfitted model cannot be promoted accidentally.
- **Diagnostic recommendations are actionable.** One or more missing factors,
  or a suggested interaction, can be added to the selected model directly from
  Diagnostics. EasyGLM then opens that model for review and refitting.

## 0.4.4 — 5 September 2026

### New features

- **A second built-in example.** The Project & data page now offers sample datasets to build a Poisson claim-frequency model and another to build a Tweedie burn-cost model. Each option loads sensible variable roles, a 70/30
  train/test split and an editable starter model; nothing is fitted automatically.
- **A clearer way to restart the workbench workflow.** After trying a sample, select **Start over
  and choose another sample** to return to both choices. EasyGLM warns before
  discarding an unsaved setup.
- **Better workbench documentation.** The README now has a short visual tour.
  A separate examples index and workbench walkthrough cover roles, splitting,
  model design, interactions, fitting, diagnostics, model comparison,
  rate-table adjustments and export without making the front page unwieldy.
- **Model design and fitting in one place.** Factor design, optional
  interactions, fit settings, fitting and results now form one Model-page
  workflow.

### Fixes and improvements

- **Residual-factor searches no longer use the holdout data.** They always use
  training rows and exclude columns marked **Ignore** or **ID**, including
  leakage fields such as known claim count in a burn-cost model.
- **Cross-validation is safer and reproducible.** Folds are shuffled with the
  project seed instead of depending on input row order. Interaction validation
  now uses out-of-fold predictions from the main model.
- **Diagnostics are more reliable.** Missing-factor rankings account for
  statistical noise, offset models use the correct null benchmark, and a
  missing-value row can no longer accidentally become the base risk.
- **Model comparisons behave consistently.** Fitted models remain available
  after page changes or reloads, challenger plots use comparable bands, and
  double lift falls back to a null model when there is no incumbent.
- **Rate-table reviews distinguish fitted and adjusted predictions.** After a
  smoothing or manual change, the original model prediction remains visible
  beside the adjusted prediction and actual experience.
- **The interface explains more of its own workflow.** Project files, custom
  knots, interactions, base-risk choices, model creation and training versus
  holdout views now have clearer labels and guidance.

### What may look different

- A cross-validated model may select a different penalty than an earlier
  version because validation folds are now shuffled correctly.
- Previously cached development fits are ignored where their statistical
  meaning has changed. EasyGLM refits them rather than silently reusing them.

Ideas intentionally left for later releases are in
[`docs/FUTURE_RELEASES.md`](docs/FUTURE_RELEASES.md).

## 0.4.3 — 4 September 2026

- Added `easy_glm.launch_workbench()` so the workbench can be opened from
  Python, including with an in-memory pandas or Polars dataframe.
- Removed Streamlit's first-run email prompt, which caused a misleading network
  error on some Windows machines even though the workbench had started.
- Stopping the launcher with Ctrl+C now exits cleanly.

## 0.4.2 — 3 September 2026

- Included the browser workbench and its charting dependencies in the normal
  `pip install easy_glm` installation.
- Added the `easy-glm-workbench` launch command and the first guided French
  motor sample.
- Improved explanations of project state, setup progress, fitted versus
  working tables and comparison defaults.

## 0.4.1 — 3 September 2026

- Reworked the README around a practical first claim-frequency model.
- Reorganised the examples into a clearer learning sequence.
- Added an automated check that the documented examples actually run.

## 0.4.0 — 3 September 2026

- Introduced the complete modelling workflow used by the current workbench:
  data roles and preparation, train/holdout splitting, factor design, model
  fitting, diagnostics, champion/challenger comparison, rate-table review and
  export.
- Added step, categorical, continuous and piecewise-linear factors, monotone
  constraints and two-way interactions whose main effects stay fixed.
- Added Poisson, Gamma, Tweedie, Gaussian and binomial models with portable
  rate-table scoring.
- Added smoothing, caps, rounding, manual table adjustments, undo/redo,
  snapshots and base-rate rebalancing.
- Added Python, command-line, Excel, scorer and self-contained HTML report
  exports.
- Added compact fitting and scoring for books containing millions of rows.
- Hardened saved projects and cached runs against stale data, conflicting
  browser sessions and hand-edited project files.

MIT licensed. See [LICENSE](LICENSE).
