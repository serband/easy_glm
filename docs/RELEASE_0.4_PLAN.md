# easy_glm 0.4 — release plan

*Status: agreed 2026-09-02, amended the same day after the independent plan
review (`docs/reviews/00-plan-review.md`; all five blocking items accepted — see
§Revisions). Supersedes the "Phase 3" list in `docs/WORKBENCH_PLAN.md`. Work
happens on a `release-0.4` branch with one PR per piece; each PR ships behind
green CI.*

## Theme

0.3 made rate tables exact and put the whole workflow in the browser. 0.4 makes
the modelling **expressive enough for a real book** — interactions and
piecewise-linear terms, two things real rating models routinely need that 0.3
cannot express — removes the legacy path, and turns the workbench from
"functional" into something people prefer over a spreadsheet.

## Workstreams

### A. Two-way interactions (core → engine → workbench → export)

The GLM term `A × B` on top of the mains `A + B`. With step / one-hot mains the
exact decomposition is

    relativity(a, b) = rel_A(a) · rel_B(b) · rel_AB(a, b)

so the export stays Emblem-shaped: the two main-effect tables plus one
two-dimensional adjustment table.

* **core** — `InteractionEncoder(a, b)`: columns are the outer product of the
  two parent encoders' *bin indicators* (not their step columns), so each
  coefficient is one cell's adjustment and lasso zeroes empty / weak cells.
  Cell exposure threshold lumps thin cells into "no adjustment". Optional
  penalty weight so interactions are penalised harder than mains.
* **engine** — `VariableConfig(type="interaction")` with a 2-D table
  (`from_a, to_a, from_b, to_b, relativity`) and a fast path (two
  `searchsorted`s / dict lookups → index into a matrix). `RateModel.predict`
  multiplies mains × interactions. JSON round-trip, snapshots, `update_relativity`
  on a cell.
* **workbench** — Design page: "Add interaction" (pick two predictors, cell
  exposure threshold); Diagnostics: A/E heatmap by the pair (in or out of the
  model — the standard way to *find* interactions); Rate tables: heatmap +
  editable grid; Excel: one matrix sheet per interaction.
* **export** — `InteractionEncoder(...)` written out; adjustments per cell.
* **Exactness test** unchanged: `RateModel.predict == fit.predict`.

### B. Piecewise-linear (L-dummy) terms

`LinearEncoder(var, knots)`: continuous piecewise-linear curves for mileage,
sum insured, vehicle value. **Basis (superseded by R10 / piece B2):** the
original hinge columns `max(x − k, 0)` penalised bends; after the actuary's Q3
answer the columns are per-band overlaps `clip(x − k_j, 0, k_{j+1} − k_j)` so
the lasso penalises *slopes* (flat unless the data insists) and monotone
constraints are sign bounds on slopes.

* **engine** — a numeric table type where each band carries a *slope*:
  `relativity(x) = exp(a_j + b_j · x)` within band `j` (log-linear in `x`). Stored
  as `from, to, relativity_at_from, slope`; scoring is one `searchsorted` and an
  `exp`. Excel shows relativity at both band ends.
* **workbench** — Design: kind = `linear`; the rate-table editor edits band-end
  relativities and re-derives the slope (keeps continuity by default).
* Mixed designs (step for age, linear for mileage) are just different encoders.

### C. Remove the legacy path (promised for 0.4)

Delete `generate_blueprint`, `prepare_data`, `fit_lasso_glm`, `predict_with_model`,
`ratetable`, `generate_all_ratetables`, `transforms.py`; drop the `legacy`
extra and DuckDB entirely. Rebuild the pieces that still lean on them:
`RateModel.from_glm_model` (take a `GLMFit`), the benchmark runner (on
`fit_glm`), the affected tests and scripts. `RateModel.from_rate_tables` stays
for hand-built tables but takes the 0.3 table format.

### D. Workbench: from functional to preferred

1. **Runs survive a reload.** Persist fitted runs next to the project file
   (`.easyglm-runs/<hash>.pkl`), keyed by the spec hash, so a browser refresh or
   reopening the project restores fits instantly.
2. **Exploration on a sample, fits on everything.** Two frames in state: the
   sample drives Explore / Design previews; fits, diagnostics and tables use the
   full data (principle 4 of the workbench plan; today the sample applies to all).
3. **Champion vs challenger everywhere.** A persistent "compare with" selector in
   the sidebar; A/E overlays on Diagnostics and Rate tables; a Compare page with
   metrics side by side, double lift, and a table of *which relativities differ*.
4. **HTML report.** One self-contained file: summary, metrics, per-variable
   relativity + A/E charts, coefficient table, the exported script in an
   appendix. From the Export page and the CLI.
5. **Relativity tooling.** Smooth (isotonic / moving average in log space),
   cap/floor, round-to-band, undo; a diff view between snapshots. Drag-to-edit
   is a *spike* only (Plotly editable shapes) — promoted if it works, else 0.5.
6. **Look and feel.** A proper theme (`.streamlit/config.toml`, typography,
   consistent chart palette), wide-layout density pass, keyboard-first
   variable switching, and a "Start from the French motor example" button so a
   new user sees a fitted model within a minute.
7. **Editor bugs found in 0.3:** `launch_editor` passes `--server.port` after
   `--` so the port is ignored; the A/E formula default for count targets.

### E. Modelling extras (small, high-value)

* Offset from an existing premium / current rate (`log(current)`), so the model
  fits a *change* — the standard rate-review setup.
* Per-variable penalty weights (`P1`): e.g. leave categorical mains unpenalised,
  penalise interactions more.
* Tweedie power and binomial (logit) in the workbench; binomial tables as odds
  relativities with a clear label.
* Target-loss-ratio base rate: enter a target and the base rate is solved.

### G. Scale: up to ~5M rows in memory

The 0.3 design matrix is dense float64 (678k rows × 197 columns ≈ 1 GB). At 5M
rows that is ~8 GB, so:

* **Design matrix as a tabmat `SplitMatrix`**: categorical one-hot blocks become
  `CategoricalMatrix` (indices, no zeros stored), step blocks stay dense but in
  **float32** (glum accepts float32; verify coefficient agreement to 1e-6 against
  float64 on the French motor set). Target: 5M rows × 200 columns in < 3 GB.
* **Chunked scoring** in `GLMFit.predict` / `RateModel.predict` (row blocks of
  ~500k) so diagnostics never materialise a second full matrix.
* **Lazy loading**: `load_source` scans parquet/csv lazily and materialises once;
  the exploration sample is drawn without loading everything twice.
* **Progress**: long fits report progress in the workbench (glum `verbose` hook →
  `st.progress`), and CV runs folds with the warm-start path.
* **Benchmarks**: a repeatable script (`scripts/bench_scale.py`) that fits 1M and
  5M synthetic rows and records time / peak RSS; numbers go in the README.

### F. CLI and packaging

* `easy-glm run project.json` — headless fit + tables + script + report;
  `easy-glm export project.json --script`.
* `mypy` in CI on `core`/`workflow`; docs site (mkdocs) with the Playwright
  screenshots; Python 3.14 in the matrix; version `0.4.0`.

## How each workstream is tested

Four layers, all automated except the last:

| Layer | What it proves | Where |
|---|---|---|
| **Unit** | each function does what its docstring says on tiny hand-checkable frames | `tests/test_*.py` |
| **Invariants** | the properties that make the product trustworthy, checked on every fit: `RateModel.predict == fit.predict` (1e-12); the exported script reproduces the model; JSON/Excel round-trips; predictions do not depend on row order or chunking | `tests/test_invariants.py` (new, parametrised over designs: step, categorical, linear, interaction, mixed) |
| **Planted truth** | synthetic data with a known effect must be recovered: a planted interaction cell, a planted slope, a planted leak | `tests/test_recovery.py` (new) |
| **Golden French motor** | metrics on the cached French motor set stay within tolerance of recorded values (holdout A/E, Gini, deviance explained, relativity shape by age) so refactors cannot silently change results | `tests/test_golden.py` (new; skipped when the cache is absent) |
| **Scale** | time and peak memory on 1M / 5M synthetic rows | `scripts/bench_scale.py`; a `-m slow` test asserts the 5M budget, nightly not per-PR |
| **App** | every workbench page renders with and without a fit; key actions work | `tests/test_app.py` (AppTest) + Playwright drive in CI, screenshots as artefacts |
| **Actuarial check** | the human review: plain-language summary, the numbers and pictures, and the domain questions | `docs/reviews/<workstream>.md`, delivered with each PR |

### Per workstream

**C — legacy removal**
- Tests: the full suite passes with the legacy modules deleted; `pip install easy_glm` in a clean venv has no DuckDB; the benchmark runner produces easy_glm rows on all four families; `RateModel.from_glm_model(fit)` equals `to_rate_model(fit)`.
- Actuarial check: none needed beyond "nothing changed": golden French motor numbers identical before and after.

**G — scale**
- Tests: float32/SplitMatrix coefficients agree with the 0.3 dense float64 fit to 1e-6 (French motor); chunked `predict` equals unchunked; lazy loader yields the same frame and sample as the eager one; `-m slow` test: 5M × ~200 columns fits in < 3 GB peak RSS and < 5 min.
- Actuarial check: a table of fit time and memory at 100k / 1M / 5M rows.

**A — interactions**
- Unit: `InteractionEncoder` column count = bins_A × bins_B (minus lumped cells); a cell below the exposure threshold gets no column; JSON round-trip.
- Invariants: with all interaction coefficients forced to zero, predictions equal the mains-only model; exactness with interactions present; Excel matrix sheet reads back to the same relativities; exported script round-trip.
- Planted truth: a synthetic book with a strong `Age × Cover` cell is recovered (cell relativity within 10% of truth, no spurious cells above 1.02 elsewhere); the A/E heatmap on a model *without* the interaction shows the cell (max |log A/E| in that cell > 0.2).
- App: Design page can add/remove an interaction; Rate tables shows the heatmap and edits a cell; Diagnostics heatmap renders.
- Actuarial check: French motor `DrivAge × VehPower` main tables + adjustment matrix, before/after A/E heatmaps, holdout Gini with and without the interaction. Question for you: is "mains + adjustment matrix" how you want to read it in Excel, and what cell exposure threshold is sensible?

**B — piecewise-linear terms**
- Unit: hinge columns; slope table construction; continuity at band edges (relativity at end of band j equals relativity at start of band j+1 within 1e-12 unless a step is also present).
- Invariants: exactness of the slope-table scoring vs `fit.predict`; Excel and script round-trips.
- Planted truth: a synthetic linear-in-log effect with two slope changes is recovered (slopes within 10%, knots kept only near the true changes).
- App: kind = linear selectable; editor edits band-end relativities and re-derives slopes; chart shows a continuous curve.
- Actuarial check: French motor `Density` as linear vs step — the two curves overlaid with A/E, and the deviance comparison.

**D — workbench**
- D1 run persistence: fit, reload the page (Playwright), fit is restored with identical predictions; changing the spec invalidates it.
- D2 sample vs full: with a 50k sample set, Explore uses ≤ 50k rows while the fitted `train_rows` equals the full training count.
- D3 compare: Compare page renders for two fitted models; the relativity diff lists exactly the bands that differ; overlays appear on Diagnostics and Rate tables.
- D4 report: the HTML file is self-contained (no external requests), opens in a headless browser without console errors, and contains one section per predictor.
- D5 tooling: smoothing preserves the exposure-weighted mean relativity; cap/floor and round are idempotent; undo restores the previous snapshot exactly.
- D6 theme: Playwright screenshots of every page reviewed by the reviewer and by you.
- Actuarial check: you use the workbench on a real book for one session; the list of frictions becomes the polish backlog.

**E — modelling extras**
- Tests: with an offset of `log(current premium)`, the fitted relativities equal those of a model on `claims / current premium`-style targets within 1e-8 (algebraic identity); `P1` weights change the number of non-zero terms in the expected direction; target-loss-ratio solver reproduces the target on the training set to 1e-10.
- Actuarial check: a rate-change setup (offset = current premium) fitted in the workbench, with the resulting relativities as a rate-change table.

**F — CLI / packaging**
- Tests: `easy-glm run project.json` in a subprocess produces the script, tables, `.easyglm` and report; `mypy` clean on `core`/`workflow`; the 3.14 CI leg passes.

## GUI quality: use it like a professional, then try to break it

Automated page tests prove nothing crashes on the happy path. Two more kinds of
testing run on every workbench change, both scripted with Playwright against a
real server so they are repeatable, plus an unscripted session by a "breaker"
agent whose only brief is to misuse the tool.

### Persona runs (scripted, kept in `tests/e2e/`)

**Actuary — rate review.** Open a project with untidy column names and a
current-premium column. Set roles, recode a categorical band, add derived
columns built from a conditional expression, filter to positive premium, random
split, fit frequency with `log(current premium)` as offset, add a two-way
interaction, look
at A/E by every rating factor on holdout, cap one relativity, export Excel and
the script, reopen the project file and confirm the fit and adjustments are
still there. Assertions: every step succeeds, exported script reproduces
predictions, Excel has one sheet per factor plus the interaction matrix.

**Data scientist — model comparison.** French motor: fit `freq_v1` (CV lasso),
clone to `freq_v2` with an interaction and a linear Density term, Compare page
shows both, double lift, residual factor search on v1 finds the interaction
pair, promote v2 to champion, HTML report contains both models. Assertions:
metrics tables agree with `workflow.model_metrics`; report opens headless with
no console errors.

### Break-it catalogue (scripted where possible, extended by the breaker agent)

Data & files: empty file; one-row file; a CSV with mixed types in a column;
column names with spaces, dots, unicode and leading digits; two columns
differing only by case; a 3,000-level categorical; an all-null column; a
constant column; negative and zero exposure; NaN and ±inf in the target;
a 2 GB path that does not exist; uploading the project JSON as data and data
as the project.

Variables: target = weight = split (same column); rename a column onto an
existing name; rename to an empty string; recode every level to the same value;
recode to an empty string; a derived column that references itself; a derived
expression that raises (division by a string); a filter that drops every row;
a filter that keeps one row; deleting a column used by a model.

Split & design: 100% / 0% training fraction; split column with three values;
integer knots on a float column with a 1e9 range; a custom knot list with
duplicates, text, one knot above the max; `n_bins = 200` on a binary column;
min level share = 0.5; monotone on a categorical.

Model: zero predictors; predictors only the id column; alpha = 0; alpha = 1e9;
CV with 2 folds on 30 rows; Tweedie on negative targets; binomial on counts;
divide by weight with no weight; rename the target after fitting; delete the
champion model; create a model named `""` or with a slash; fit twice quickly
(double click).

Rate tables & export: set a relativity to 0, to −1, to `1e12`, to text; edit
the null row; reset while editing; download every artefact for an unfitted
model; export with a level named `Other`; a variable named `from`.

Session: refresh mid-fit; open the same project in two tabs and edit both;
autosave to a read-only path; close the terminal that launched the server and
reopen; back/forward browser buttons; very narrow viewport.

**Rule for every finding:** the tool must never show a raw traceback or lose
the project. Acceptable outcomes are a clear message, a disabled control, or a
graceful fallback. Each finding gets a test that reproduces it before the fix.

### The breaker agent

Runs after the reviewer signs off on a workbench piece, against the live app,
with the catalogue as a starting point and instructions to invent more. It
writes `docs/reviews/<piece>-breakage.md`: what it did, what happened,
severity (data loss / crash / misleading output / cosmetic). Data loss and
crashes block the merge.

## Working protocol (builder / reviewer / actuary)

Roles
- **Orchestrator** (this session): owns the plan, cuts the work into PR-sized
  pieces, runs the loop below, merges.
- **Builder**: implements one piece on `release-0.4`, with tests, and writes the
  actuarial check.
- **Reviewer**: an independent agent that has *not* seen the builder's
  reasoning. It gets the plan section, the acceptance criteria, the diff and
  the test output, and returns findings ranked *blocking / should fix / nit*,
  each with a concrete failure scenario. It also reviews the plan itself
  before code is written.
- **Actuary** (you): reads `docs/reviews/<piece>.md` — never code — and answers
  the domain questions in it.

Loop per piece
1. Builder implements + tests → CI green locally.
2. Reviewer reviews → `docs/reviews/<piece>.md` (findings + verdict).
3. Builder addresses every *blocking* and *should fix* item (or argues back in
   the review file); reviewer re-checks. Max three rounds; unresolved
   disagreements are escalated to the orchestrator with both positions.
4. Orchestrator runs the full suite + Playwright, opens the PR, merges on green.
5. Actuarial check delivered to you with the numbers, pictures and questions.

Rules
- No merge without reviewer sign-off, green CI and a written actuarial check;
  workbench pieces additionally need the breaker's report with no open data-loss
  or crash findings.
- The reviewer may not edit code; the builder may not edit the review verdict.
- Every finding names a failure scenario; "I'd do it differently" is a nit.
- Questions to the actuary are domain questions only, batched, and each one
  says what happens by default if unanswered.

## Sequencing (each step ships behind green CI)

1. **C1 — foundations**: `tests/test_invariants.py` written first so it fails on
   today's bugs; fix them (Excel ignores adjustments, integer categoricals score
   as Other, offsets ignored by the RateModel, `diff(v1)`, editor port, A/E
   formula from metadata); `.easyglm` format version 2 with strict type dispatch
   and the metadata the later pieces need (`offset_col`, `offset_is_log`, `link`,
   `target_is_rate`); project format version 2 with a migration hook.
2. **C2 — legacy removal**: delete the blueprint/DuckDB path, scripts, rebuild the
   benchmark and `from_glm_model`, golden French-motor test on a checked-in
   deterministic subsample.
3. **A** interactions, core + engine (encoder contract in §Revisions).
4. **B** piecewise-linear, core + engine (encoder contract in §Revisions).
5. **D2** sample vs full (before D1 so persisted keys are stable), then **D1**
   run persistence.
6. **G** scale — starts with the two-day spike (bin-index `StepMatrix` block /
   aggregated rows); targets are set from the spike.
7. **A/B workbench + export**, then **D3** compare, **D7** editor fixes.
8. **E1–E2** offset from current premium, penalty weights.
9. **D4–D6** report, relativity tooling, theme; drag-to-edit spike; **E3–E4**,
   **F** CLI, docs. After step 7 a coherent 0.4.0 can be cut; the rest may slip
   to 0.4.x.

## Acceptance for 0.4.0

* A model with `A × B` and a linear term round-trips: fit → `RateModel` →
  Excel → script → refit, with `RateModel.predict == fit.predict` to 1e-12.
* A realistic model (interactions, derived columns from conditional
  expressions, a categorical recode, a piecewise-linear term) can be built
  entirely in the workbench and exported as a script that reproduces it.
* No DuckDB anywhere; `pip install easy_glm` is lighter than 0.3.
* Reloading the browser does not lose a fit.
* 5M synthetic rows × ~200 design columns fit in < 3 GB peak memory; float32
  coefficients agree with float64 to 1e-6 on the French motor set.
* Compare page and HTML report exist; relativity tooling (smooth, cap, round,
  undo) works in the editor; the drag-to-edit spike has a written verdict.

## Decisions (2026-09-02)

1. **Interactions** export as the two main-effect tables plus one A×B
   adjustment matrix (exact; Emblem-shaped; mains stay independently editable).
2. **Piecewise-linear terms** are exact log-linear within each band
   (`from, to, relativity_at_from, slope`), not a step-grid approximation.
3. **Scale target is ~5M rows in memory** → workstream G (SplitMatrix / float32
   design, chunked scoring, lazy loading, progress, scale benchmark).
4. **Workbench scope**: champion vs challenger + Compare page, HTML report,
   relativity tooling, and a time-boxed drag-to-edit spike — all in 0.4.

## Revisions after the plan review (2026-09-02)

The independent review (`docs/reviews/00-plan-review.md`) is accepted in full for
its blocking items; where this section conflicts with text above, this section
wins.

### R1. Format versions and strict dispatch (was B1) — piece C1
* `.easyglm` gains `format_version: 2`. Readers reject newer versions with a clear
  message and load version-less files as version 1. The same bump adds
  `ModelMetadata.offset_col`, `offset_is_log`, `link`, `target_is_rate`.
* `RateModel.predict` dispatches through an explicit `{type: scorer}` map and
  raises `ValueError` on unknown types.
* `PROJECT_VERSION = 2` with a migration hook in `Project.from_dict` that branches
  on the incoming version before overwriting it; unknown keys are ignored with a
  warning rather than crashing.

### R2. Piecewise-linear contract (was B2) — piece B
`LinearEncoder(var, knots, clamp=(lo, hi), null_indicator=True)`:
* `x` is clipped to `[lo, hi]` (defaults: training min/max) **before** the hinges,
  so the curve is exactly flat outside the range; hinge columns at every knot
  including `lo`, so the first real band has a fitted slope.
* Nulls: all hinge columns zero plus the `is null` column (relativity at `lo`
  times the null factor).
* Table rows `(from, to, log_rel_at_from, slope)` per band, plus the two flat end
  rows and the null row; scoring = clip, `searchsorted`, one `exp`.
* Base: relativity 1.00 at a stated `x_base` (default: the lower knot of the
  most-exposed band), recorded in the table.
* Monotone on linear terms is **refused with a clear error** in 0.4 (the sign
  bound would force convexity); revisit with a cumulative constraint later.

### R3. Interaction cells tied to rate-table rows (was B3) — piece A
* The cell index of a row is the **rate-table row index** of each parent (bins +
  null row for numerics; levels + Other row for categoricals), computed by one
  shared helper (factored out of `_modal_bins`) used by the encoder, the table
  builder and the scorer.
* Cells below the exposure threshold get no column → relativity 1.00; Excel and
  the editor show training exposure per cell so 1.00 reads as "no data" vs "no
  adjustment".
* Invariant reworded: setting every cell to 1.00 in the RateModel equals the GLM
  with the interaction slice of `coef` zeroed.
* Spec: `ModelConfig.interactions: list[Interaction(a, b, min_cell_exposure,
  penalty_weight)]`; `validate` and `model_hash` include them. Cell adjustments
  use `(from_a, to_a, from_b, to_b)`; touches `Adjustment`, `Change`,
  `update_relativity`, `_mask_for_row`, the editor.
* Interaction blocks are fitted with `scale_predictors=False` semantics (per-column
  `P1` weight instead) so thin cells are not under-shrunk; planted-truth test
  includes thin non-signal cells (10–50 rows) that must stay within [0.98, 1.02].

### R4. Scale re-baselined (was B4) — piece G
* Coefficients and predictions stay **float64**; the exactness invariant is never
  loosened. Float32 is an internal option for the design only.
* Acceptance replaces "1e-6 coefficients" with: predictions on French motor agree
  to 1e-4 relative, same non-zero set (or a listed difference), holdout
  deviance/Gini agree to 1e-4.
* The memory budget is stated as arithmetic the benchmark asserts (design bytes
  = rows × (4 × dense columns + 4 × categorical variables) × multipliers for
  fit and, separately, CV with `n_jobs=1`). One dtype for all blocks and vectors,
  chosen in `DesignSpec.build`; `P1`, bounds and offset are cast to it.
* A two-day spike decides the approach and the 5M number: a bin-index
  `StepMatrix` tabmat block (cumulative-sum trick) versus fitting on rows
  aggregated by identical design row (exact for Poisson/Gamma/Tweedie).

### R5. Offsets in the RateModel (was B5) — piece C1
`to_rate_model` copies `offset_col`/`offset_is_log` from the fit; `RateModel.predict`
applies the offset like exposure (warns when the column is missing); the
invariant suite includes an offset case. E1 becomes a thin layer on top.

### R6. Should-fix items adopted
S1 (state the offset identity conditions: Poisson, `scale_predictors=False`,
rescaled alpha), S2 (persistence key includes data size/mtime and library
versions; load failure = cache miss), S3 (D2 before D1; knots always from full
training data), S4 (add `RateModel.from_excel` as a product feature), S5 (progress
= one tick per alpha-path point / per CV fold, else spinner with elapsed time),
S6 (golden test on a checked-in 50k-row deterministic subsample so it runs in CI),
S7 (see R3), S8 (`link` in metadata; binomial scorer returns probabilities and
refuses exposure), S9 (order above), S10 (protocol below). Nits: renumber later,
replace `assert isinstance` dispatch with explicit `raise`, drop the unused
`EASY_GLM_MAX_ROWS`, delete `scripts/` investigation files in C2, populate or
drop `Snapshot.metrics`, drag-to-edit acceptance stated up front (edit a point →
table updates → A/E recomputes, in Streamlit 1.57 without a custom component),
smoothing preserves the exposure-weighted mean of **log** relativities.

### R7. Protocol additions
* Any change to an existing tolerance or golden number is itself a blocking
  review item and needs a written reason in the PR.
* The actuarial check is produced by a committed script
  (`scripts/checks/<piece>.py`) that the reviewer re-runs.
* Reviewer findings live in `docs/reviews/<piece>.md`; the actuary's plain-language
  document lives in `docs/checks/<piece>.md`.

### R8. Domain questions
Nine domain questions with defaults are in `docs/checks/00-questions-for-the-actuary.md`.
Building proceeds on the defaults; answers change parameters, not architecture.

### R9. Workstream G decided by the spike (2026-09-02)
Spike deliverables: `docs/spikes/g-scale/` (report, results, re-runnable bench,
prototype `StepMatrix`). Decisions:
* **No float32 anywhere.** With glum 3.4.1 float32 designs stop converging at
  1M+ rows (hit `max_iter`, 0.2–0.4 % from the float64 fit), `coef_` comes back
  float32, and an uncast float64 `sample_weight` segfaults tabmat. Float64 only.
* **Design = tabmat `SplitMatrix` of bin-index `StepMatrix` blocks (new
  `core/stepmatrix.py`, cumulative-sum trick, ~170 lines, 9 `MatrixBase`
  methods + `_cross_sandwich`) plus `CategoricalMatrix` blocks.** Measured: 1M
  rows 0.7 GB / 2.6 s (today 2.8 GB / 3.6 s); 5M rows 2.0 GB / 16 s (today pages
  at 5.6 GB / 103 s); coefficients agree with dense float64 to 1.2e-13 with the
  same non-zero set. Memory scales with variables, not knots.
* **Acceptance for G**: float64 representations agree to 1e-10 with the same
  non-zero set (the exactness invariant is untouched); design bytes formula
  `n·(4·v_step + 8·n_null + 4·v_cat)` asserted by the benchmark; 5M × ~200
  columns under 3 GB peak.
* **Aggregation by identical design row** is exact (2e-13) but compresses
  real data little (1.5× on French motor); ship as opt-in `aggregate=True` for
  coarse designs, not the default.
* Scoring always goes through the float64 rate-table lookup from bin codes,
  never `model.predict(X)`; `spec.build` must stop materialising `hstack`
  transients (3.5 GB at 5M today).
* Known obstacles: `StepMatrix` blocks must precede other blocks in the
  `SplitMatrix` (cross-sandwich dispatch); glum's `check_array_tabmat_compliant`
  needs a one-line shim (upstream PR later); the prototype sandwich is ~3× slower
  than tabmat's C kernel at 5M (acceptable; two optimisations listed in the report).
* **Handed over from piece A (review S5):** the interaction encoder builds a dense
  `n_rows × n_cells` float64 block with a per-cell loop (0.54 GB at 1M rows × 66
  cells; ~8 GB at full French-motor scale with many cells). In G, represent kept
  cells as a single cell-index block (a `CategoricalMatrix` over the kept-cell
  code, with "no cell" as an all-zero row) so memory is 4 bytes per row per
  interaction.

### R10. Actuary's answers to Q1–Q6 (PR #2 review comments, 2026-09-03)
Q1, Q2, Q4, Q6 confirm the defaults. Two answers change the build:
* **Q3 → piece B2 (linear basis penalises slopes).** Replace the hinge basis
  with per-band columns `clip(x − k_j, 0, k_{j+1} − k_j)` so each coefficient is
  the slope *within* band j; the lasso then zeroes slopes (flat sections) rather
  than bends. Table representation `(from, to, log_rel_at_from, slope)` is
  unchanged (slope_j = β_j directly). Monotone constraints on linear terms are
  re-enabled as sign bounds on the slope coefficients (`increasing` → β ≥ 0).
  Re-run the planted-slope recovery, the b-linear actuarial check and update
  Q3 in the questions file. The reviewer's B2 concern (convexity) no longer
  applies because the constraint is on slopes, not on slope changes.
* **Q5 → piece A2 (two-stage interactions, mains frozen).** When a model has
  interactions, `run_model` fits stage 1 (mains only, as today) and stage 2
  (interaction cells only, `fit_intercept=False`, offset = stage-1 linear
  predictor incl. any user offset). `to_rate_model` takes main tables and base
  rate from stage 1 and cell adjustments from stage 2; the exactness invariant
  becomes `RateModel.predict == exp(η₁ + η₂)`. The exported script writes both
  stages. The Model page shows the two stages; the pair search runs on the
  stage-1 residuals (already the case). The `a-interactions` actuarial check is
  re-run: main tables must now be identical with and without the interaction.
  Cell penalty rule (P1 = penalty_weight·0.5/sd) is re-validated in stage 2.
Sequencing: B2 and A2 run right after W3 (hardening), before D3/D4.
* **Q9 (answered 2026-09-03): all numeric variables are step unless explicitly
  specified as not needing knots.** Default unchanged. B2 adds
  `VariableDesign.kind = "continuous"`: one slope on the raw (clamped) value,
  no interior knots — implemented as the linear encoder with a single band, so
  it shares the table type, editor and export. The Design page lists the three
  explicit overrides (categorical, linear, continuous) next to the default.

### R11. README and examples that run (release gate, added 2026-09-03)
The README is the "how do I use this" a visitor lands on; in 0.3 not every
example on it worked, which destroys trust. For 0.4:
* **Every fenced `python` block in `README.md` is executed by
  `tests/test_readme.py`** (blocks extracted in order, run in one namespace in a
  temp working directory, with `load_external_dataframe()` redirected to the
  50k fixture and a `traintest` column added when the block expects one). A
  block that must not run (e.g. `pip install`) is a `bash` block, never a
  `python` block. Blocks may be marked `<!-- readme-test: skip -->` only with a
  reason in the comment; the test asserts fewer than three skips.
* **Every `examples/*.py` runs in CI** on the fixture (`EASY_GLM_DATA` env var
  overrides the download) with a total runtime under two minutes; each example
  asserts its own headline result (e.g. holdout A/E within 0.9–1.1).
* README content, in this order, each block runnable end to end: install ·
  fit in one call · look at the rate tables · A/E on holdout · adjust a
  relativity and see the effect · export Excel, `.easyglm`, the Python script ·
  reload the scorer and score new business · open the workbench on the same
  project · building blocks (DesignSpec → fit_glm → to_rate_model) · interactions
  · piecewise-linear terms · rate-change setup with an offset. Screenshots of the
  workbench from `docs/checks/img/`.
* Sequenced **last before 0.4.0** (after B2, A2, D3, D4) so the examples show
  the final API; reviewed like any piece, and the reviewer runs every block by
  hand as well as via the test.
