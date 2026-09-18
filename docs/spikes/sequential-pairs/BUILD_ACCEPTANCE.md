# Sequential pair build acceptance

Review build on `codex/sequential-catboost-pairs`, 18 September 2026.
No release, tag or push; README unchanged.

The main GLM stays frozen. Each ordered CatBoost pair learns from two raw columns
against earlier deployed tables, then becomes a table on the configured bins.
The next pair uses that table. Scoring, diagnostics, Python scoring export and
Excel use these same tables. Initial support is Poisson/log and Tweedie/log.

## Checks completed

- Full Python regression suite: 1,434 passed, one skipped, one slow test deselected.
- Final focused pair, desktop, variable-sync and format-compatibility suite:
  108 passed after the final save-state changes.
- Ruff, Black and core/workflow mypy passed. Svelte check reported no errors or
  warnings; the production frontend build passed.
- Browser tests: four editor/conversion tests, one real fitted-pair workflow,
  one legacy main-effects reuse test and one reduced-challenger test passed.
  The fitted workflow covers manual edit preview/apply, downstream invalidation,
  reorder, refit impact and a 390-pixel layout without page overflow.
- Independent Astra technical and actuarial reviews passed. Adversarial checks
  covered malformed configurations, role removal, cell edits, stage identity,
  cancellation and stale-result publication.
- Fresh-process table scoring worked with CatBoost imports blocked. Training
  Python export replayed an upstream manual edit before fitting the next pair.
  Independent Excel reconstruction reproduced table scoring.

Numerical, leakage, resource and cancellation evidence is recorded in
`RESULTS.md` and `ASTRA_ACCEPTANCE.md`. Runtime and memory figures describe the
measured workloads; the memory estimate is not a hard operating-system limit,
and the deadline is checked between native fits. The large slow-scale suite
was not run for this review build.
