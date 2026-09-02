# easy_glm 0.4 — release plan

*Status: agreed 2026-09-02 (decisions in §Decisions). Supersedes the "Phase 3"
list in `docs/WORKBENCH_PLAN.md`. Work happens on a `release-0.4` branch with one
PR per workstream; each PR ships behind green CI.*

## Theme

0.3 made rate tables exact and put the whole workflow in the browser. 0.4 makes
the modelling **expressive enough for a real book** — interactions and
piecewise-linear terms, the two things the original bike model used that 0.3
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

`LinearEncoder(var, knots)` with hinge columns `max(x − k, 0)`: the AGLM
"L-dummies". Curves are continuous, which suits mileage, sum insured,
vehicle value.

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
- Actuarial check: you use the workbench on the bike data for one session; the list of frictions becomes the polish backlog.

**E — modelling extras**
- Tests: with an offset of `log(current premium)`, the fitted relativities equal those of a model on `claims / current premium`-style targets within 1e-8 (algebraic identity); `P1` weights change the number of non-zero terms in the expected direction; target-loss-ratio solver reproduces the target on the training set to 1e-10.
- Actuarial check: the bike rate-change setup (offset = current premium) fitted in the workbench, with the resulting relativities as a rate-change table.

**F — CLI / packaging**
- Tests: `easy-glm run project.json` in a subprocess produces the script, tables, `.easyglm` and report; `mypy` clean on `core`/`workflow`; the 3.14 CI leg passes.

## GUI quality: use it like a professional, then try to break it

Automated page tests prove nothing crashes on the happy path. Two more kinds of
testing run on every workbench change, both scripted with Playwright against a
real server so they are repeatable, plus an unscripted session by a "breaker"
agent whose only brief is to misuse the tool.

### Persona runs (scripted, kept in `tests/e2e/`)

**Actuary — rate review.** Open the bike-style project (SAS-like column names,
a current-premium column). Set roles, recode the PO-score band, add the
`Drvr1Exp_Q/M` derived columns, filter to positive premium, random split, fit
frequency with `log(current premium)` as offset, add `Cover × VehTerms`, look
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

1. **C** legacy removal first — shrinks the surface everything else touches.
2. **G** scale — SplitMatrix/float32 design + chunked scoring, before the encoders
   multiply (interactions widen the matrix).
3. **A** interactions: core + engine + tests → workbench → export.
4. **B** piecewise-linear: engine table type → core encoder → workbench.
5. **D1–D3** run persistence, sample vs full, champion/challenger + Compare page.
6. **D4–D5** HTML report, relativity tooling (smooth / cap / round / undo, snapshot
   diff); **drag-to-edit spike** time-boxed to two days — promoted if it works.
7. **E** modelling extras, **F** CLI, **D6** theme and onboarding.

## Acceptance for 0.4.0

* A model with `A × B` and a linear term round-trips: fit → `RateModel` →
  Excel → script → refit, with `RateModel.predict == fit.predict` to 1e-12.
* The bike model from the original script (interactions, `Drvr1Exp_Q/M`
  derived columns, PO-score recode, mileage linear term) can be built entirely
  in the workbench and exported as a script that reproduces it.
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
