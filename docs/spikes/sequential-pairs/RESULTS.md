# Phase 0 sequential pair evidence

Executed on the local macOS CPU with CatBoost 1.2.10, NumPy 2.4.5 and SciPy
1.17.1. These are bounded experiments, not claims about every actuarial book.

## Numerical contract

- Independent scalar minimisation and gradient checks match the Poisson and
  Tweedie loss-optimal cell multipliers for unequal weights and widely varying
  upstream means. The distiller accumulates in float64 log space, rejects bad
  baselines/teacher means, ignores zero fitting weights, accepts zero outcomes
  during teacher fitting, and leaves empty/unsupported cells neutral.
- CatBoost `RawFormulaVal` on a `Pool` with a baseline includes that baseline;
  prediction on raw feature columns returns only the correction. The teacher
  adapter consumes the latter. `allow_const_label=True` allows all-zero and
  all-equal labels; neutral remains a CV choice, not a forced shortcut.
- For Tweedie, the training-only scale is `max(1, max_positive_y/1000)` on each
  fitting partition. Target and baseline are divided together, and original
  response units are restored by `baseline_original * exp(raw_correction)`.
  Scale is recorded per fold and final stage. It changes CatBoost's optimisation
  and is not claimed to preserve identical fitted trees.
- A uniform 20×20 grid with ten positive-weight rows per cell has every cell
  supported at the new-pair default minimum share 0.001. Legacy GLM interaction
  support remains 0.005.

## Executed CPU prototypes

`PYTHONPATH=src .venv/bin/python docs/spikes/sequential-pairs/prototype.py`
used 2,800 rows, a deployed main numeric table, a nonzero external offset and
three overlapping raw pairs. Each teacher received the preceding frozen TABLE
baseline. Validation Poisson deviance was 1.237041 → 1.168739 → 1.156802 →
1.154159. Three depth-3, 70-tree teachers took 0.252 seconds together; peak
process RSS was 262.8 MiB. This simple split is a mechanics spike, not model
selection evidence.

`OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python
docs/spikes/sequential-pairs/nested_benchmark.py` used 5,000 rows (4,000
training), three stages, five outer and five inner folds, two shallow default
teachers plus neutral per stage, and a fixed main-GLM alpha. It executed 348
CatBoost fits and 30 distinct fold-local main-GLM fits in 14.163 seconds, with
371.2 MiB peak process RSS. Selected deployed-table training-CV deviance by
stage was 1.304126 → 1.244975 → 1.201028. These are model-selection scores,
not untouched holdout performance. The full-training prefix was reused;
fold-local prefixes were separately fitted and cached by partition and settings.

The same script with `--rows 100000 --stages 2` used 80,000 training rows,
20,000 untouched holdout rows, the same default two-teacher search and CPU one
thread. It executed 77 CatBoost fits and 30 fold-local main-GLM fits in 22.202
seconds, with 411.9 MiB peak process RSS. Selected deployed-table training-CV
deviance was 1.272470 → 1.213326. This is a larger two-stage resource check;
it does not establish a memory guarantee for million-row books or eight stages.

With `--rows 100000 --stages 2 --optuna`, the automatic 8-trial main search and
4-trial whole-prefix search executed 162 CatBoost fits and 30 fold-local main
GLM fits in 54.286 seconds, with 428.8 MiB peak process RSS. The selected
deployed-table training-CV deviance was 1.269660 → 1.218588. The search used
Optuna 4.9.0 TPE with three startup trials for the current stage, two for a
prior-prefix study, one CPU thread, and no pruning or outer-validation early
stopping. This measurement included a few seconds of another short CPU check;
the numbers are observed prototype results rather than an isolated speed claim.

## Initial v1 limits

- At most eight stages and 10,000 cells per pair grid including missing/Other.
  Fixed mode allows three CatBoost candidates and eight deterministic complete
  prefix configurations. Automatic mode allows at most 16 current-stage and
  eight complete-prefix trials; the new workbench defaults are eight and four.
  Both modes always evaluate the neutral choice separately. CPU thread count
  is one.
- The fixed-mode defaults remain depth 2 / 60 trees / learning rate 0.08 /
  L2 3 and depth 3 / 120 trees / learning rate 0.06 / L2 5. Automatic mode
  searches depth 2–5, 40–160 trees in steps of 20, learning rate 0.03–0.15
  and L2 0.1–20 on log scales. A whole-prefix study is rerun on each outer
  training partition and never receives full-training selected parameters.
- Preflight, before the main fit, estimates teacher-fit count, at most 30
  distinct fold-main fits for a multi-stage run, the configured main penalty
  CV and alpha grid, cells and table bytes. It refuses more than 5,000
  possible teacher fits or a conservative estimated process peak above 3 GiB.
  The estimate starts at 512 MiB plus 512 bytes per training row, four times
  the main design's dense/compact byte estimate and 128 bytes per pair cell.
  It is a rejection gate,
  not a measured peak guarantee; no rows or bins are silently discarded.
- A 900-second cooperative deadline starts before the main GLM fit and is
  checked between fits. Native CatBoost/GLM calls need desktop process
  termination for prompt cancellation. Cancellation latency still needs a
  measured desktop process test.

Poisson/log and Tweedie/log with `1 < power < 2` are the enabled pair families.
Binomial/logit, Gaussian/identity, Gamma and other links remain on the legacy
path until their own loss and deployed-table contracts are implemented.
CatBoost and Optuna are now included in the standard installation; saved table
scoring does not import them. Manual pair-cell edits replay in CV by exact categorical
identity and explicit fixed numeric cuts. A changed numeric cut refuses replay
with an actionable error.
