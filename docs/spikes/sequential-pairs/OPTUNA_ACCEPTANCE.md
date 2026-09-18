# Independent Astra review: automatic Optuna tuning

Reviewed 18 September 2026 in `/private/tmp/easyglm-feature-selection`, using
Optuna 4.9.0. Read-only production review; corrections were made by Sol.

## Executed independent checks

`/private/tmp/pair-review-optuna.py` fitted an **automatic → fixed → automatic**
three-stage chain on the small numerical fixture. The automatic stages used four
outer-stage trials and three inner-prefix trials, deliberately exceeding TPE's
three/two startup counts so adaptive proposals were exercised.

The driver checked:

- Neutral remained eligible for every stage.
- Every completed trial contained five fold records, with its objective exactly
  reproducible as summed loss contributions divided by summed fitting weights.
- Both later stages retained five independent prefix-study traces.
- Poisoning outcomes and raw input A on outer validation fold zero left that
  fold's earlier-prefix trial parameters and loss records unchanged in both later
  stages. This check intentionally does not require current-stage global adaptive
  proposals to remain unchanged: those legitimately use the CV selection scores.
- Poisoning holdout outcomes and raw input A reused all three fitted stages with
  identical saved study traces and tables.

`/private/tmp/pair-review-optuna-seed.py` changed the downstream stage seed while
retaining caches. It checked that five new prefix-study cache entries were made,
the unchanged upstream table was reused, the downstream stage was refitted, and
its warm-cache prefix-study records and selected parameters exactly matched a
fresh-cache run with the new seed. Both drivers passed.

## Findings corrected

The initial prefix-study cache key omitted the current stage's seed although the
sampler used that seed. It now records the derived sampler seed, preventing stale
study reuse after seed changes. Adaptive prefix ties initially used trial order
after counting active stages; they now prefer lower aggregate depth and tree
count before trial order. Durable Optuna tests initially used only startup
trials; the builder added a four/three-trial case that exercises adaptive TPE.

## Code review conclusions

Outer-fold contexts are constructed once per stage and reused across proposals.
Earlier-prefix tuning sees only that outer fold's training partition, evaluates
complete fixed parameter tuples across five inner folds, and never consumes
full-training winners or histories. Mixed fixed/automatic chains resolve actual
parameter tuples, rather than reusing invalid candidate-list indices.

Neutral is scored independently; model selection uses weighted observed-response
deviance from deployed tables. Teacher loss remains diagnostic. Sampling is seeded
and serial, startup counts are explicit, and pruning is disabled. Trial records
are ordinary serializable data. Search budgets/settings, source/algorithm identity
and Optuna/dependency versions participate in reuse identity. Frozen scoring does
not call the search machinery.

No remaining statistical/cache blocker was identified within this review's
scope. The root agent owns final full-suite, export/browser and measured runtime
acceptance. This record makes **no runtime claim based on the older fixed-grid
benchmarks** and is not release approval.
# Integrated checks

After the review fixes, the focused numerical, desktop, export, scoring,
variable-sync and persistence suite passed: **116 tests**. Ruff, Black (199
files) and mypy (32 core/workflow files) passed. The frontend check and build,
five editor/legacy browser cases and the real fitted-stage browser case passed.

The live review at port 8830 switched both existing stages to automatic tuning,
saved them and fitted the default eight/four-trial search on 5,648 training rows
in **9.332 seconds**. The first pair selected a correction; the second selected
no change. The visible stage summaries show automatic tuning and no numbered
candidate editors. The current project configuration was saved separately at
`/private/tmp/easyglm-pair-review/optuna-preview-project.json`.

The reproducible 100,000-row benchmark (80,000 training rows, two stages) measured
54.286 seconds and 428.8 MiB peak RSS; see `RESULTS.md` for its scope and command.
These measurements are workload-specific, not timing guarantees.
