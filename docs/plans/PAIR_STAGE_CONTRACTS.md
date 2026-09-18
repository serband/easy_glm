# Sequential pair implementation contracts

Status: implementation contract; resource defaults are conservative preflight ceilings,
not measured claims that all workloads meet the runtime budget. README is untouched.

## Ownership

- Sol numerics/fitter lead: `workflow/pair_distillation.py`, `workflow/pair_stages.py`,
  `workflow/run.py`, numerical/fitting tests and Phase 0 spike.
- Sol engine: `engine/models.py`, `engine/rate_model.py`, `workflow/project.py`,
  `core/excel.py`, pair engine/project tests.
- Sol desktop: `desktop/`, `app/state.py`, focused desktop pair tests.
- Sol frontend: `frontend/` and focused UI tests.
- Sol outputs (when assigned): `workflow/export.py`, `workflow/report.py`,
  `workflow/_svg.py` and focused output tests.
- Astra architecture: this document, independent validation and review. Root
  handles cross-module integration review. Builders coordinate APIs before edits.

## Project and scorer

`PairCandidateConfig(depth, iterations, learning_rate, l2_leaf_reg)` is a fixed
teacher parameter set. `PairStageConfig(stage_id, a, b, candidates,
min_weight_share=0.001, seed=42, cv_folds=5)` defines one ordered stage. Neutral
is always an implicit candidate. IDs are stable and independent of order/name.
`ModelConfig.pair_stages` defaults to an empty list. `pair_method` preserves
`sequential_catboost` for a saved model even when its stage list is empty; old
models default to `legacy_glm`, while a nonempty pair list infers the new method. Reject legacy interactions
mixed with pair stages, duplicate/reversed pairs, unsupported family/link, invalid
roles and malformed search settings. Parents need predictor roles but need not be
selected main effects. Start with Poisson/log and Tweedie/log, 1 < power < 2.

`PairTableConfig(stage_id, parents, axes, cells)` lives in engine/models.py.
`parents` and `axes` are ordered pairs. Each axis is an independent numeric or
categorical `VariableConfig` with neutral row relativities, explicit tails and
missing/Other mapping. Never insert pair-only axes into the main variables dict.
Numeric pair knots resolve Variables settings independently of main term kind:
a continuous main must not erase its pair axis's interior cuts. Use the existing
`engine._scoring.row_index` kernel against each independent axis.

`PairCellRow` carries canonical rectangle coordinates, relativity, positive-weight
row count, fitting-weight sum, weight share and fallback reason. Avoid calling
arbitrary fitting weights exposure. Cell IDs for numerical routines are row-major,
`axis_a_index * axis_b_size + axis_b_index`. All cells including neutral support
fallbacks are represented or deterministically reconstructible.

`RateModel(..., pair_tables=None)` owns an ordered pair-table list. Existing
variables/legacy interactions remain unchanged. Pair multipliers apply before
external offset, response conversion and optional exposure. Add a stable
`linear_predictor(data, column_map=None, include_offset=True)` that accumulates
log relativities and applies the external offset exactly once, without taking the
log of exposure-multiplied predictions. Scoring imports no CatBoost.

Project and scoring JSON advance to version 3 with explicit pair serialization;
legacy version 1/2 predictions remain unchanged. Confirm an actual released loader
rejects new pair models. Bump persisted-run and desktop training-cache formats.
Clone/snapshot/reconstruction paths must carry pair tables. Excel needs explicit
ordered pair metadata and both axes for independent reconstruction.

## Fitting boundary

`ModelRun.fit` and `spec` remain the main GLM. Add `pair_stages` containing ordered
`PairStageArtifact` records: stage/table identity, selected candidate or neutral,
input-prefix fingerprint, fold/training provenance, every fold loss contribution
and validation fitting weight, prefix/table/teacher losses, approximation loss,
timings and reuse. Native teachers are optional private diagnostics, never scoring.

`fit_pair_stages(project, train, model_config, main_rate_model, *, cache=None,
progress=None)` returns `(new_rate_model, stage_artifacts)`. Resolve exact signature
with run.py owner before implementation. Invoke after building/applying upstream
main edits and before metrics. The fitter receives training data only. It never
mutates an already published/cache-owned scorer.

CatBoost learns from exactly two raw prepared columns. Its native baseline is the
complete frozen prefix on the link scale. CatBoost prediction on a Pool WITH a
baseline includes that baseline; raw-feature prediction returns correction only.
The response adapter explicitly consumes correction only. If training uses
scaled targets, divide both target and baseline by the same training-only scale;
inverse prediction is original baseline times exp(correction), without another
scale factor. Distil teacher means, never observed outcomes.

## Bounded nested validation

Separate `fit_prefix_fixed(training_rows, complete_configuration)` from
`select_prefix(outer_training_rows, bounded_configurations)`. The fixed fitter
builds every main design, pair axis, teacher and deployed table only on its passed
rows and NEVER recursively calls prefix selection. Prefix selection evaluates a
small deterministic list of COMPLETE prefix configurations over inner folds, then
refits its winner on outer-training rows. Stage-k outer candidates share that
fold-local fitted prefix. All held-back rows score through frozen tables.

Candidate specifications and any data-derived main-alpha grids must exclude outer
validation rows. Do not seed folds with full-training selected alphas, parameters,
axes or models. The main penalty request remains honored; do not silently replace
its n_alphas or CV settings with a smaller search. Internal main-penalty tuning on
a passed training partition is bounded additional work and must be accounted for
explicitly in preflight. If a proposed implementation tunes mains inside the fixed
prefix fitter, label the complete configuration as a fixed *fitting procedure*,
and ensure its own selection sees only those passed training rows; do not claim
its selected alpha was fixed before the inner fold. This is a permissible bounded
nonrecursive implementation, subject to statistical reviewer approval.

Use five outer and five inner row folds for v1. No outer-validation early stopping.
Aggregate observed-response loss contributions divided by total validation fitting
weight. Neutral is eligible at every stage; numerical ties prefer neutral, then
lower complexity. Never advertise training-CV selection loss as untouched final
performance. Holdout data is assessment only.

## Resource preflight

Proposed initial hard ceilings to confirm against Phase 0 measurements:

- Eight pair stages, 10,000 Cartesian cells per stage including missing/tail rows.
- At most three teacher candidates per stage, plus implicit neutral.
- At most eight complete-prefix configurations in an inner search; deterministic
  data-independent enumeration/subsampling, with an all-neutral configuration.
- CPU only, one training thread by default; two maximum for this implementation.
- Tree depth at most six and iterations at most 200 per teacher candidate.
- A cooperative 900-second total pair-fit deadline, checked between fits and
  recorded as a failed fit without publishing partial results; desktop process
  termination provides cancellation during a native fit, target at most 5 seconds.
- Preflight enumerates expected teacher/main fit counts, main n_alphas/CV cost,
  cells and estimated table bytes before any fit. A conservative 5,000 teacher-fit
  cap rejects an oversized search with an actionable explanation. This cap is not
  a claim of acceptable runtime; Phase 0 must calibrate it downward if necessary.

No silent row subsampling, bin merging, candidate dropping or partial publication.
The initial default search is neutral plus two fixed teachers: depth 2, 60 trees,
learning rate 0.08, L2 3; and depth 3, 120 trees, learning rate 0.06, L2 5. The
lead reported a three-stage 2,800-row/70-tree CPU-two prototype at 0.25 seconds
and 263 MB including imports; this is NOT a nested-CV benchmark. Resource budgets
must be shown as limits and measured evidence separately.

Cache main fits independently of complete-prefix configuration: identical exact
inner-training rows/main settings must not repeat the same user-requested main
penalty CV for every prefix candidate. Use deterministic outer/inner partitions
that are invariant across stage index so reusable work is genuinely identical.
With five outer/five inner folds, eight prefix configurations and three teachers,
the uncached teacher upper bound for pair stage k is 205*(k-1)+16. Eight stages
would mean 5,868 fits, exceeding the proposed 5,000-fit cap; reject that workload
or substantiate a lower actual count through safe prefix reuse. Do not hide the
main n_alphas times CV solver work from the estimate.

## Reuse, edits and publication

Fingerprint actual training content/order, preparation and roles, resolved split,
main settings, axes, exact upstream deployed tables/manual edits, candidate space,
fold identity and dependency/algorithm/cache versions. Never share a fitted object
between full-training and fold caches. Appending reuses the full-training prefix;
a change invalidates only the dependent suffix. A stage's manual cell edit leaves
that stage fitted but invalidates its successors. Main/base changes invalidate all
pairs. Canonical fixed-cut cell edits may replay in folds; incompatible edges fail
with guidance, never approximate silently.

`rate_model_for` must reattach stored pair tables before edits. `jobs.py` must stop
blanket-clearing upstream edits during suffix refits. Existing model_hash excludes
adjustments/base override, so lineage requires a separate deployed-prefix check.
Cancellation/failure/stale revision cannot publish a partial chain or replace the
last complete scorer.

## Desktop and outputs

Use server-derived ordered stage cards with stable stage_id, sequence, parents,
baseline labels, status, table dimensions/support, candidate/neutral result,
loss/approximation information and reuse. Coordinate the exact JSON packet between
desktop and frontend owners. States: Not fitted, Up to date, Needs refitting,
Fitting, No improvement, Failed. Keyboard accessible up/down controls preserve IDs.
Legacy models retain their existing interaction UI; pair-only inputs are eligible.

Every model diagnostic and permutation importance must call the complete scorer
and include pair-only parents. Main coefficients/path views remain main-only.
Training Python export must replay full workflow; exact scoring export stores tables
and requires no CatBoost. Reports and Excel must not omit pair effects.
