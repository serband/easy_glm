# Changelog

## 0.4.0 (unreleased)

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
  `ratetable`, `generate_all_ratetables` and the SQL transform helpers are gone,
  together with the DuckDB dependency and the `legacy` extra.
- `RateModel.from_rate_tables(tables, base_rate, ...)` takes the 0.3 table format
  (`from`, `to`, `relativity`; both-null row = null / Other) and no blueprint;
  `RateModel.from_glm_model(fit, ...)` takes a `GLMFit`; `create_rate_model`
  follows the same signature.
- `matplotlib` and `seaborn` moved to the `viz` extra; `scikit-learn` dropped;
  `rdata` is imported lazily. The base install is lighter.
- Exploratory scripts under `scripts/` and the scoring prototype example deleted.

### Fixed (C2)
- `gini` pooled tied predictions inconsistently, so the reported Gini could move at
  the 1e-5 level between identical runs; ties are now pooled deterministically.

### Changed (C2)
- The benchmark runner fits easy_glm through `DesignSpec` + `fit_glm`.
- Snapshots can carry the metrics of their version (`create_snapshot(...,
  metrics=)`, `set_snapshot_metrics`); the workbench records train/holdout
  metrics on every fit.
- A golden French-motor test on a checked-in 50k-row subsample runs in CI.
