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
