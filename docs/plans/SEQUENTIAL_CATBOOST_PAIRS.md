# Sequential CatBoost pair corrections — build and test plan

Status: proposed implementation plan, not implemented or released.
Baseline: review branch `codex/glm-feature-selection`, commit `81075ac`.

## What the user will get

Keep the main-effects GLM. Fit each requested pair separately with CatBoost on
its two raw columns. Convert that fitted correction to a two-way rate table on
the user's bins. Freeze that table before fitting the next pair. The next pair
uses the GLM **and every earlier deployed table** as its offset.

The Model page will show that sequence as numbered stages, including what each
stage builds on, what is already fitted, and what needs refitting. A later-stage
change must not unnecessarily refit an unchanged upstream stage.

CatBoost is a training tool. The deployed model is the GLM plus the tables.
Diagnostics, saved scorers, Excel and Python must agree on that deployed model.
Additional one-way effects inside a pair correction are explicitly allowed.

## Decisions already agreed

- Stage 1 is the existing main-effects GLM, with its existing external offset.
- Each subsequent stage has exactly two distinct raw source variables as features.
  Configured renames/recodes/types/filters still apply; numeric inputs are not
  replaced by model-bin codes before CatBoost training.
- Each baseline contains the entire upstream deployed prefix, on the link scale.
- A teacher is converted to a table and frozen before the next teacher is fitted.
- Numeric table boundaries come from Variables; categorical axes use resolved
  levels. Unchanged main effects and earlier tables remain unchanged.
- No purification, removal of residual one-way effects, or automatic centring.
- Fitting, automatic boundaries, parameter selection and table construction use
  training data only. Holdout results are for assessment, never automatic tuning.
- The table approximation can lose performance. Show and test that loss.
- Order matters. Reordering changes the fitting problem and invalidates a suffix.
- Preserve the README. This plan does not authorise a release, tag or push.

## Architecture

### 1. New stage configuration; legacy models remain legacy models

Add an ordered `pair_stages` field to `ModelConfig`, with stable stage IDs, raw
variable references, CatBoost search settings and table support settings. Keep
old `Interaction`, `TwoStageFit` and legacy GLM-cell models working unchanged.
Do not reinterpret an old project's interaction list as CatBoost stages.

For the first implementation, reject mixing legacy GLM-cell interactions and
new pair stages in one model. Offer conversion as a new model: preserve the
source model, copy its mains, and create ordered pair stages requiring new fits.
New eligible models use the staged editor. Existing models clearly show their
legacy method until converted. Update the repository's old two-stage-only
instructions when the new implementation is introduced, not in this planning PR.

Stage IDs are independent of display names and position. A reversed duplicate
pair is a duplicate, not a second independent effect. Reject duplicate pairs in
v1. A pair variable must be an eligible predictor, but need not have its own
selected GLM main effect. Protected target/weight/offset/split/time/id roles are
not silently promoted. Store pair axes independently of main-effect tables so
pair-only inputs do not become artificial GLM main effects or inflate factor counts.

Use versioned JSON and migration tests. Bump `PERSIST_FORMAT` and desktop
training-cache formats when fitted artefacts change. A stale cache is a cache
miss; a saved legacy scoring model remains loadable with unchanged predictions.

### 2. Keep the teacher separate from the scoring representation

Add a workflow module for pair fitting/distillation, not CatBoost code inside
`core.fit_glm`. Extend `ModelRun` with ordered stage artefacts; keep its GLM fit
for GLM coefficients and paths. Do not manufacture GLM coefficients or a fake
`TwoStageFit` for a tree model.

A stage artefact contains:

- Stage ID, pair, chosen parameters, objective/link/Tweedie power and seed.
- Resolved bin/category axes, cell values, support and neutral-fallback reasons.
- Input-prefix fingerprint, fold identity when relevant and training provenance.
- Table CV loss, prefix-only CV loss, teacher loss and approximation loss.
- Reuse state, timings and optional training-only teacher diagnostic artefact.

The scoring artefact contains tables only. An optional native CatBoost model may
be retained privately for teacher diagnostics, but is neither required nor called
by deployed scoring. Changing an earlier raw variable must affect all applicable
upstream and downstream table lookups during whole-model scoring.

Add an explicit persisted `PairTableConfig(stage_id, parents, axes, cells)` and
ordered pair-table collection to `RateModel`. Reuse numeric/categorical row-lookup
kernels, not the legacy requirement that every pair axis must be a main table.
Resolve parents and order from fields; never parse `A×B` names or rely on dict
insertion order. Keep legacy interaction-table loading unchanged. New scoring
artefacts need a format/version mechanism that makes incompatible loaders fail
instead of silently returning main-effects-only predictions. A new version field
alone is insufficient if older loaders ignore it: verify against released-loader
fixtures and use a distinct required representation or format if necessary.

Pair numeric axes resolve the shared Variables bin settings on the relevant
training partition, independently of whether the main term is step, linear or
continuous. Main-GLM curves remain unchanged. Step terms with the same settings
share equivalent boundaries; a continuous main slope must not collapse a pair
axis to one cell. Persist the stage's actual edges, tails and missing mappings;
show its table dimensions explicitly. Categorical axes similarly have explicit
resolved levels/Other. Test this contract in Phase 0 before extending the engine.
Add a stable link-scale scoring path to avoid applying external offsets twice
or taking the log of already exposure-multiplied totals.
All app diagnostics and feature importance must use the complete table scorer.
Coefficient and lambda-path views apply to the GLM main-effects stage only.

### 3. Table conversion is defined by the model loss

For a log-link stage, let `b_i > 0` be the upstream mean in the modelling response
units, `t_i > 0` the teacher's complete mean including that baseline, and `w_i`
the exact fitting weight. A cell deploys `b_i * r`.

A single adapter must convert CatBoost's raw correction plus supplied baseline
and any target scaling into the complete response-scale teacher mean `t`.
Distillation never receives raw formula values in place of means. Reject a stage
with non-finite/non-positive baseline or teacher means; ignore zero-weight rows
and require a positive finite cell denominator. Observed response zero remains
valid. Never silently clip an invalid baseline into a successful fit.

Minimising Tweedie loss against the teacher's soft predictions gives:

```
r = sum(w_i * t_i * b_i ** (1-p)) / sum(w_i * b_i ** (2-p))
```

For Poisson (`p=1`) this reduces to `sum(w*t) / sum(w*b)`. For Tweedie it is
not generally an arithmetic mean of multipliers or a total-claims ratio. Verify
this derivation with an independent scalar optimiser and gradient checks.
Do not replace teacher predictions with observed outcomes in this step: that
would refit the cell means and discard the CatBoost smoothing we are trying to
retain. Evaluate the resulting table against observed outcomes on validation.

Preserve the existing target/weight/offset convention exactly. Do not silently
convert a totals model into a rate model or substitute a different exposure
weight for a general Tweedie power. Accumulate in float64 and use stable
log-domain sums when powers/extreme baselines require them. CatBoost's internal
quantisation is separate from the existing GLM float64 requirement.

Recommended v1 cell policy: retain the current minimum-cell-support control;
insufficient-support and empty cells receive neutral correction `1`. Record why.
Use sum of positive fitting weights divided by total positive training fitting
weight as the v1 support share (row count if unweighted); record row count,
fitting-weight total and any distinct exposure total under separate names.
This avoids calling an arbitrary sample weight "exposure". The exact default
threshold is locked at Phase 0 and included in the CV candidate specification.
Well-supported missing/Other buckets may be learned if present in the configured
axes; truly unseen/unrepresented combinations receive the documented fallback.
Numeric tails follow the existing parent-band/clamp rules. Do not turn all
out-of-range values into a new undocumented bin. Do not extrapolate ICE into
empty cells to manufacture support. No silent winsorisation or renormalisation.

Quantify the overall prediction change of a stage, even though the GLM base rate
and earlier tables remain unchanged. A pair correction may legitimately change
the book's expected total; normalising that away changes the fitted problem.

### 4. Family and dependency gate

The first complete vertical slice covers Poisson/log and Tweedie/log with
`1 < p < 2`. Reuse the same power throughout the chain. Native CatBoost baselines
receive raw link-scale predictions, not means and not signed outcome residuals.
Large-target scaling, if required, must transform targets and baselines together
and be learnt from fold-training rows. Test the inverse transformation.

Create an explicit capability matrix before enabling any other family/link.
Binomial/logit requires odds tables and a one-dimensional loss-optimal cell
solve. Gaussian/identity needs additive scoring rather than forcing positive
multipliers. Gamma or nonstandard links must not silently use a nearby Tweedie
power or RMSE. Retain the legacy path for unsupported combinations and explain
support before a fit is queued. Broader family support is a separate gate, not an
unannounced approximation inside this change.

CatBoost currently lives in the `benchmark` extra. Move training support to an
appropriate production dependency/extra with a tested version range; choose the
installation policy at the first checkpoint. Prefer CPU initially for deterministic
resource budgets and platform coverage. Table-only scoring must work without
CatBoost installed; training replay must report its missing dependency clearly.

### 5. Cross-validation of the deployed pipeline

The validation unit is the whole prefix, not a CatBoost fitted against one shared
full-data or shared-OOF baseline vector.

For each stage candidate and validation fold:

1. Construct the main design and fit the GLM on that fold's training rows.
2. Build each earlier stage on those same fold-training rows, converting and
   freezing its table before the next stage.
3. Fit the candidate teacher and distil its table on fold-training rows only.
4. Score held-back rows through those tables. Compare with the same prefix with
   a neutral new stage. Select using weighted observed-response deviance.

Earlier-prefix tuning must also exclude the held-back rows. If its parameters
need selection, perform that selection inside the fold-training partition; never
reuse parameters selected using the outer fold's outcomes and describe the score
as untouched validation. Nested tuning is potentially expensive: the feasibility
spike must measure it, and cache fold-local prefixes. Training-CV model selection
scores are labelled as such, not advertised as unbiased final performance.

Pin down the nested search as follows: for an outer validation fold and stage k,
select earlier-prefix settings solely inside the outer training partition. The
inner search evaluates a bounded set of complete prefix parameter configurations;
for each candidate configuration, parameters are fixed before its inner fold is
fitted, and all prefix models/grids/tables are built on that inner training fold.
This avoids recursively creating an unbounded new tuning search at every stage.
Refit the selected prefix on outer-training rows, then test each fixed stage-k
candidate on the outer validation rows. Average those outer losses to choose
stage k. Main-GLM penalty selection belongs inside this boundary too. Never seed
fold caches with the full-training fitted model's selected parameters. Explicit
user-fixed parameters are configuration, not a fitted-cache shortcut.

Use deterministic five-fold row CV initially, preserving the current project
train/holdout split. No new automatic time/group splitter is implied by this
feature; expose the limitation where relevant. Inner splits derive reproducibly
from the parent training indices and seed. A stage's neutral candidate is always
eligible; ties within a documented numerical loss tolerance choose neutral first,
then lower complexity. Record selected settings, all fold losses and effective
fold weights so aggregation can be independently reproduced (sum weighted loss
contributions divided by sum validation fitting weights).

Start with a bounded, seeded search over shallow-tree complexity/iterations and
regularisation, plus the neutral correction. Select using the deployed table's
CV loss, not the teacher's training or validation score. Tie-break toward simpler
models/the neutral stage. CatBoost early stopping may use only an inner training
validation partition; outer fold labels must not also select tree count.
The search budget/defaults are set from the spike, not invented in the UI first.

After choosing settings, fit the accepted stage on all training rows against the
already frozen full-training prefix and convert it to the final table. Appending
a stage does not refit that full-training prefix. Fold-local copies remain
separate from it. Stage order is user-defined, not searched using holdout results.

### 6. Reuse, cancellation and manual edits

Cache prefixes using actual training values/row order, role/preparation/weight/
offset/split settings, main fit and resolved axes, stage settings and the exact
upstream deployed tables. Include algorithm, dependency and cache-format versions.
Include raw content and row-order fingerprints, resolved split membership,
categorical dtypes/level order, filtering/derived expressions, missing-value
policy, CPU/thread policy and exact manual table contents. Keep full-training
artefacts separate from fold-specific and tuning artefacts.

Editing stage k invalidates k onward. Removing/reordering invalidates from the
first changed position. Changing main settings invalidates the chain. Adding a
last stage reuses everything upstream. Never mutate a cached prefix in place.
Cancelled/failed/stale jobs cannot publish partial fitted models or overwrite the
last valid model. Retain completed reusable prefixes privately and advertise only
stages that have been atomically published for the current settings.

Manual adjustments need a dedicated integration gate. Recommendation: when an
upstream *deployed* table or base rate changes, mark dependent stages as needing
refit; their next fit consumes the adjusted upstream scorer. Preserve the prior
complete model for comparison. A last-stage edit has no downstream invalidation.
CV must replay fixed manual adjustment rules on fold-local prefixes, with an
explicit error for adjustments that cannot be mapped to those axes. Never score
CV against full-training adjusted tables. Stage-aware refits preserve accepted
upstream edits instead of the current blanket clearing of model adjustments.
Use stage-aware adjustment identity: stage ID plus canonical raw-axis cell keys.
Editing stage k's deployed cells invalidates k+1 onward, not k itself. Explicitly
refitting stage k replaces its manual edits, with a preview listing the edits
that will be cleared. Fitting only the suffix retains all upstream edits.
Base-rate/main-table changes invalidate every pair stage. Renaming a variable
updates parent references, not stage IDs. Snapshots retain stage configuration,
order and adjustment identity/provenance so restoring values cannot falsely
restore a downstream fit against a different prefix.

The v1 mapping rule is exact: categorical cell edits match by level identity;
numeric rectangle edits must align with the fold's canonical edges. If fixed
manual cuts make this possible, replay the edits; otherwise refuse that tuning
request with a clear explanation and a route to reset those edits or specify
fixed cuts. Do not approximate overlaps silently or reuse the full-data table.
This restriction must be reviewed in Phase 0 before promising parity for all
existing adjustment tools. Test undo/redo and snapshots by exact prefix
fingerprint, not stage label alone.

### 7. Model-page flow

The main view is a vertical sequence:

```
1  Main effects                          Up to date
   GLM settings and factor design
                  ↓ frozen main-effects prediction
2  Driver age × Region                   Up to date
   Baseline: Main effects
                  ↓ frozen main effects + Stage 2 table
3  Vehicle age × Region                  Needs refitting
   Baseline: Main effects + Driver age × Region

[Add pair correction]              [Fit remaining stages]
```

Each card shows its pair, baseline, status, table dimensions/support and relevant
settings. Use up/down reorder controls with keyboard support and stable stage IDs.
Show the affected suffix immediately when editing/reordering. Collapse completed
cards without hiding their stage number, baseline or status. Report stage and fold
progress; distinguish full-training prefix reuse from CV work still needed.

Use `Not fitted`, `Up to date`, `Needs refitting`, `Fitting`, `No improvement` and
`Failed` as user-facing states. A selected neutral stage stays visible. A table
preview is explicitly the deployed effect; raw teacher views are labelled as
training diagnostics. ICE varies only the selected pair-stage inputs and keeps
the baseline fixed; it does not construct the deployment table or imply purity.

### 8. Exports and diagnostics are part of the feature

- `.easyglm` and Excel contain every deployed stage table, axes and ordered
  provenance; independent table reconstruction must reproduce predictions.
- Python workflow export retains preparation, the optional variable search,
  stage sequence, settings, CV, teacher fits and table conversion. Running it
  retrains the workflow. It must not quietly export only the final main GLM.
- Exact scoring export uses the frozen tables and needs no CatBoost. Keep these
  two purposes explicit; do not claim a fresh stochastic training run recreates
  saved table bytes exactly.
- Reports show incremental deployed-stage performance, support and table-vs-teacher
  approximation loss. Main-effect coefficients/lambda paths stay labelled as such.
- Whole-model permutation importance, A/E, lift, time stability, residual search
  and Compare must score the deployed tables, including interaction-only inputs.
- Adjustment tools, cloning, smaller challengers, variable removal/rename and
  snapshot restore must remove/remap stages and invalidate the right suffix.
  A smaller challenger drops pairs whose parents were removed, visibly.

## Implementation workstreams and gates

| Phase | Owner | Main scope | Must pass before continuing |
|---|---|---|---|
| 0. Feasibility spike | Primary + actuarial reviewer | Two raw pairs, Poisson/Tweedie, baseline, loss-based tables, strict CV cost | Independent formulas; three-stage baseline trace; bounded runtime estimate |
| 1. Contracts | Architecture builder | Project stage schema, immutable artefacts, migration, canonical grids, capabilities | Legacy roundtrip; stable IDs; invalid config errors |
| 2. Fitting | Statistical builder | Teacher adapter, distillation, fold-local prefix training, neutral candidate | Numerical, leakage, ordering and full-prefix invariance tests |
| 3. Scoring and reuse | Engine builder | RateModel composition, cache fingerprints, worker artefacts, manual edits | Fresh-process parity; suffix invalidation; cancellation/publication tests |
| 4. App workflow | Desktop/frontend builder | Strict API, staged progress, cards, ordering, fit remaining | API revision safety and realistic browser journeys |
| 5. Outputs | Export/report builder | Python workflow/scorer, Excel, diagnostics, report, CLI | Export execution and independent workbook reconstruction |
| 6. Independent acceptance | Code reviewer + actuary + test critic | Review final implementation and full validation evidence | All blockers resolved; review preview before release |

Contracts and the vertical statistical slice precede broad UI work. Once the
contract is stable, UI and output work can proceed in parallel with separate file
ownership. `workflow/run.py`, project types and server/job lifecycle are integration
hotspots with one owner at a time. Builders do not approve their own work.

Likely files: `workflow/project.py`, new `workflow/pair_stages.py`,
`workflow/run.py`, `engine/models.py`, `engine/rate_model.py`,
`workflow/diagnostics.py`, `workflow/export.py`, report modules, `core/excel.py`,
`desktop/modeling.py`, `desktop/jobs.py`, `desktop/fit_worker.py`,
`desktop/server.py`, diagnostic/review/importance caches, and
`frontend/src/ModelWorkbench.svelte` plus a dedicated stage editor.

## Acceptance tests

1. **Numerical contract:** direct loss optimisation agrees with the Poisson and
   Tweedie cell formulas, including unequal weights, strongly varying baselines,
   zero observed claims, nonzero external offsets and scaling. Predictions stay
   finite; never silently clip an invalid fit into an apparently valid result.
2. **Sequential baseline spy:** for three overlapping pairs, inspect every fit's
   features and offset. Exactly two raw features; offset equals main GLM plus
   all earlier TABLES, once. Deliberately make teachers and tables differ so an
   accidental teacher-prefix path cannot pass.
3. **Leakage sentinels:** poison holdout outcomes/values and prove training fits,
   bins, hyperparameters and tables unchanged. Poison a validation fold and
   inspect its fold-training artefacts unchanged (its measured loss may change).
   Check fold-local parameter selection and early stopping, not just row counts.
4. **Prefix invariance/reuse:** append/edit/remove/reorder stages, change bins,
   source values, target, weights, split and external offset. Assert the exact
   expected fits run, unaffected prefix artefacts stay identical, and stale ones
   are not reused. Include cache reload across process/dependency versions.
5. **Cell semantics:** numeric edges, linear/continuous parent axes, categorical
   levels, nulls, unseen levels, tails, empty/sparse cells, pair-only variables,
   duplicate pairs, renamed variables and stage-name collisions.
6. **Deployment parity:** on genuinely new rows, compare direct sequential table
   calculations, RateModel, reloaded `.easyglm`, exported Python scorer and a
   scorer independently rebuilt from Excel. CatBoost import is blocked during
   scoring. Expected tolerance 1e-12 relative / 1e-12 absolute on moderate fixtures;
   scaling stress tolerances must be justified separately. Include source names
   containing the multiplication symbol, non-ASCII names and Excel sheet-name
   collisions. A new-format model must fail clearly in an incompatible loader.
7. **Training export:** execute `.py` end to end with bin overrides, variable
   search and two sequential pairs. Check training-only behaviour, deployed
   table outputs and capability errors. Same-seed teacher reproducibility gets
   a stated numerical tolerance, not a byte-for-byte promise.
8. **Manual adjustment lineage:** prefix table/base edits, last-stage edits,
   refits, undo/redo, clone and snapshot restore; no lost upstream adjustments,
   no silently current downstream stage against a changed prefix.
9. **UI journey:** fit mains; add two pairs; inspect baseline labels; fit; edit
   only last pair; reorder; cancel mid-fold; reload; build smaller challenger;
   export. Test narrow and wide screens, keyboard reorder and no unwanted refits.
10. **Actuarial evidence:** synthetic known signal, null pairs, overlapping pairs,
    marginal-only corrections, unequal exposure and zero-heavy Tweedie. Record
    deployed CV improvement, teacher approximation loss, calibration/support and
    order sensitivity. Compare legacy GLM interactions, mains-only and new tables
    on a fixed benchmark; do not promise CatBoost will win every case.
11. **Resource budget:** benchmark a realistic large training sample with several
    stages; record peak memory, candidate/fold count, wall time, cache reuse and
    cancellation latency. Agree concrete budgets after Phase 0 and enforce them.
12. **Regression:** old saved projects, old scorers, legacy two-stage fitting,
    current main-effect reuse, feature selection and bin exports remain valid.

Run focused tests per phase, then `ruff check .`, Black check, mypy on core and
workflow, `pytest -q`, frontend check/build and relevant Playwright journeys.
Run slow/scale gates in a suitably provisioned process; do not hide resource
failures by calling a reduced sample the full acceptance test.

Phase 0 cannot pass until numeric limits for runtime, memory, maximum
cells/stages/candidates and cancellation latency are written down. Preflight
pair-grid size (including missing/tail levels); no unbounded dense matrix/Excel
expansion from a high-cardinality pair. Fail with an actionable size estimate
rather than silently changing the user's bins.

## Review checkpoints and remaining choices

Before the main build, review Phase 0 evidence and lock: the search/resource
budget, CatBoost packaging, the initial family capability matrix, minimum-support
policy and the precise manual-adjustment/CV mapping. The recommendations above
are defaults to review, not additional instructions already agreed by the user.

Plan reviews completed: architecture planner, actuarial critic, builder
assessment, independent technical critic and independent build/test critic.
Their findings are incorporated above: explicit pair-table identity/axes,
response-unit adapter, correct Tweedie conversion, a bounded nested-CV protocol,
manual-edit lineage, cache isolation and enforceable resource gates. The actuary
found no statistical blocker to Phase 0. Phase-0 product/feasibility gates remain
explicit; this is not a claim that the implementation exists or has passed tests.

After implementation, an agent that did not build the code reviews the actual
diff; the actuary reviews executed numerical/calibration evidence; a separate
test critic inspects coverage and independently exercises exported artefacts.
Fix and re-review blockers before showing the complete preview. Planning review
is not implementation sign-off.
