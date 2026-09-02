# Review of the 0.4 release plan (before implementation)

*Reviewer: independent agent. Date: 2026-09-02. Reviewed: `docs/RELEASE_0.4_PLAN.md`
against the 0.3 code on `main` (194 tests pass). Nothing was changed except this file.*

How to read this: section 1 is the verdict, section 2 lists what must change in the plan
before a builder starts, section 3 what should change, section 7 what is already broken
today. Every finding names the failure it prevents. Where a point needs a domain decision
it is repeated in section 6 with the default I would assume.

---

## 1. Verdict

**Approve with changes.** The direction (mains + adjustment matrix, log-linear bands,
legacy removal, scale, persistence, compare) is right and feasible on this codebase, but
five points must be pinned down first: the `.easyglm` format needs a version number and
strict type dispatch, the piecewise-linear term is underspecified in ways that change the
numbers, the interaction cell definition must be tied to the rate-table rows, the scale
targets (float32 to 1e-6, 5M rows in 3 GB) are not achievable as written and must be
re-baselined, and the RateModel must learn about offsets before the rate-change work is
built on it.

---

## 2. Blocking issues (change the plan before building)

### B1. Version the `.easyglm` format and make table-type dispatch strict (workstream C, first PR)

**What.** `RateModel._to_dict()` in `src/easy_glm/engine/rate_model.py` writes no format
version (keys: `metadata, base_rate, current_version, column_mapping, variables,
snapshots`; `current_version` is the snapshot pointer, not a format version).
`RateModel.predict` dispatches with `if config.type == "numeric": ... else:
score_categorical(...)`, so any unknown `type` is scored as categorical.

**Why it matters.** I built a `VariableConfig(type="interaction")` and called `predict`:
it returned 1.0 for every row with no error. Once 0.4 writes `interaction` and `linear`
tables, a 0.3 installation (or a scoring system with the old package) given a 0.4 file
will produce silently wrong premiums for every policy. The plan mentions "JSON
round-trip" for the new types but never a version bump.

**Change to the plan.** Add to workstream C, before A and B: (1) `format_version: 2` in
`_to_dict`, `_from_dict` raises on a newer version and migrates version-less files as
version 1; (2) `predict` dispatches on an explicit map of `type -> scorer` and raises
`ValueError` for anything else; (3) the same bump carries the metadata fields the other
workstreams need (`offset_col`, `offset_is_log`, `link`, `target_is_rate` /
`divide_target_by_weight`), so the format changes once, not four times; (4) a test that a
version-3 file is rejected with a clear message. Also bump `PROJECT_VERSION` to 2 in
`workflow/project.py` with a migration hook: `VariableDesign(**v)` and `ModelConfig(**m)`
crash on unknown keys, so a 0.4 project file opened by 0.3 fails, and a 0.3 project
opened by 0.4 must load unchanged.

### B2. Specify the piecewise-linear term completely (workstream B)

**What.** The plan says "hinge columns `max(x − k, 0)`" and a table
`from, to, relativity_at_from, slope`. Four things that change the fitted numbers are
left to the builder:

1. *The first band.* With hinge columns only, the slope below the first knot is zero
   and the first band is `(−inf, k1)`, so `relativity_at_from` does not exist for it.
   If the builder adds a raw `x` column to get a slope in the first band, the curve is
   unbounded on the left (a 16-year-old or a zero-mileage record gets an extreme
   factor). One of these has to be chosen and written down.
2. *Extrapolation above the last knot.* A log-linear last band continues its slope to
   infinity. A sum insured or mileage 3× the training maximum then gets a relativity
   nobody reviewed. Rating engines normally clamp.
3. *Nulls.* `np.maximum(nan − k, 0)` is `nan`; the design matrix would contain NaN and
   glum would fail or propagate garbage. The step encoder solves this with all-zero
   columns plus an `is null` column; the linear encoder needs the same rule stated.
4. *Base point and monotone.* "Relativity 1.0 on the modal bin" has no meaning for a
   continuous curve; the base must be a specific `x`. Monotone constraints on hinge
   terms are cumulative (`slope_j = Σ_{i≤j} β_i ≥ 0`), not per-coefficient sign
   bounds, so `monotone_bounds` in `core/fit.py` (which raises for non-`StepEncoder`)
   cannot simply be extended; sign bounds on all hinge coefficients force the curve to
   be convex as well as increasing.

**Why it matters.** Each of the four produces a different curve for the same data. The
continuity test in the plan (1e-12 at band edges) would pass for all of them, so the
tests would not catch a wrong choice; only the actuary would, after the fact.

**Change to the plan.** Write the encoder contract into §B:
`LinearEncoder(var, knots, clamp=(lo, hi), null_indicator=True)` where `x` is clipped
to `[lo, hi]` (defaults: training min/max) *before* the hinges; hinge columns at every
knot including `lo`, so the slope in the first real band is a fitted coefficient and the
curve is flat outside `[lo, hi]` (exactly, because the encoder clips too). Nulls: all
hinge columns zero plus the `is null` column, giving them the relativity at `lo` times
the null factor. Table rows: `(from, to, log_rel_at_from, slope)` for each band between
consecutive knots, plus the two flat end rows and the null row; scoring = clip,
`searchsorted`, one `exp`. Base: relativity 1.0 at a stated `x_base` (default the lower
knot of the most-exposed band) recorded in the table. Monotone on linear terms: either
refuse with a clear error in 0.4, or document the sufficient (convex) bound; do not
extend `monotone_bounds` silently. See questions Q1–Q3.

### B3. Tie interaction cells to rate-table rows, including null and Other (workstream A)

**What.** The plan defines cells as "the outer product of the two parent encoders' bin
indicators". For a numeric parent the step columns put nulls in the *lowest bin*
(`StepEncoder.transform`: `NaN >= k` is False) while the rate table and the scorer give
nulls their *own row* (`_bin_rows` appends `FromToRow(None, None)`;
`score_numeric` uses `null_relativity`). Which one is "the bin" of a null is not stated.

**Why it matters.** If the encoder derives bins from the step columns, a null-age policy
contributes to cell `(bin 0, b)` in the fit but is looked up in cell `(null, b)` by the
scorer. `RateModel.predict == fit.predict` then fails only for rows with nulls or unseen
levels, which the planted-truth test will not contain unless required to. The same
applies to unseen categorical levels (`Other` row).

**Change to the plan.** State: the cell index of a row is the *rate-table row index* of
each parent (bins + null row for numeric; levels + Other row for categorical), computed
by the same function as `_modal_bins` in `core/fit.py` (factor it out and reuse it in the
encoder, the table builder and the scorer). Cells below the exposure threshold get no
column and therefore relativity 1.0; the Excel sheet and the editor must show the
training exposure per cell next to the relativity so 1.0 can be read as "no data" vs
"no adjustment". Reword the invariant "with all interaction coefficients forced to zero,
predictions equal the mains-only model" — a refit without the interaction gives different
mains, so as written it is false; the testable statement is "setting every cell to 1.0 in
the RateModel equals the GLM with the interaction slice of `coef` zeroed". Add to the
Project spec: `ModelConfig.interactions: list[Interaction(a, b, min_cell_exposure,
penalty_weight)]` (they are not columns, so `validate`'s "predictors ⊆ predictor-role
columns" and `model_hash` in `app/state.py` must include them). Adjustments on a cell need
a new key shape (`from_a, to_a, from_b, to_b`), which touches `Adjustment`, `Change`,
`update_relativity`, `_mask_for_row` in `ui/metrics.py` and the editor; list these files
in the workstream.

### B4. Re-baseline the scale targets; they are not achievable as written (workstream G)

**What.** Three measured facts against the plan's numbers (glum 3.4.1, tabmat 4.2.1):

1. *"float32 coefficients agree with float64 to 1e-6".* On a 200k-row Poisson toy with
   13 columns, float32 vs float64 differed by 5.8e-4 at default tolerances and 1.6e-4
   with `gradient_tol=1e-10`; float64 at default vs tight tolerance differed by 2.2e-4
   by itself. The optimiser's stopping rule dominates; 1e-6 is not a property of the
   arithmetic and will not be met on French motor either.
2. *Exactness.* glum returns `coef_` **in float32** when the design is float32, and
   `fit.predict` on a float32 design differs from the float64 recomposition of the same
   coefficients by 1e-7 relative. The 1e-10 / 1e-12 invariant `RateModel.predict ==
   fit.predict` (the product's central promise, `TestRateTables`) would fail.
3. *Memory arithmetic.* Step columns `1{x >= k}` are about half ones, so they cannot be
   stored as categorical or sparse; they stay dense. 5M rows × 200 columns × 4 bytes =
   **4.0 GB for the matrix alone**, before the raw frame (~2 GB at 50 columns), and
   glum's CV copies the fold rows (`X[train_idx, :]` in `_glm_cv.py`), adding ~0.8×.
   "< 3 GB peak RSS" is impossible with the proposed representation.

Also: tabmat requires every `SplitMatrix` block to share one dtype (mixing a float32
dense block with a default float64 `CategoricalMatrix` raised `Buffer dtype mismatch`
inside Cython); `P1`, `lower_bounds`, `upper_bounds` and `offset` must be cast to the
design dtype too (`P1` float64 with a float32 design raised). None of this is in the plan.

**Why it matters.** Building to an acceptance criterion that cannot pass ends either with
a quietly loosened test (the exactness tolerance is the one number that must never move)
or with a workstream that never merges.

**Change to the plan.**
- Keep the fitted coefficients in float64 (`GLMFit.coef` already casts; make
  `linear_predictor`/`predict` compute in float64, chunked) so the exactness invariant is
  untouched. Replace "coefficients agree to 1e-6" with: predictions on French motor agree
  to 1e-4 relative, the same set of non-zero coefficients (or a listed difference), and
  holdout deviance/Gini agree to 1e-4.
- State the memory budget as arithmetic the benchmark asserts: design bytes =
  `rows × (4 × dense_columns + 4 × categorical_variables)` plus a stated multiplier for
  fixed-alpha fit and, separately, for CV (sequential folds only; `n_jobs=1` at scale).
- Add a two-day **spike** before committing to numbers: a `StepMatrix` tabmat block that
  stores only the bin index per variable and applies the cumulative-sum trick
  (`X β = cumsum(β)[bin]`, `Xᵀv = Lᵀ(Bᵀv)`, sandwich `Lᵀ(BᵀWB)L`). This is the only
  route that makes 5M rows × 30 step variables cheap (4 bytes per row per variable).
  Alternative to spike alongside: fitting on rows aggregated by identical design row
  (exact for Poisson/Gamma/Tweedie with weights; the classic Emblem trick). Decide the
  5M target from the spike, not before.
- Put the dtype rule in the plan: one dtype for all blocks and vectors, chosen once in
  `DesignSpec.build`.

### B5. Teach the RateModel about offsets before building the rate-change setup (workstreams C and E)

**What.** `fit_glm` accepts `offset_col` and `GLMFit.predict` adds it, but
`to_rate_model` and `RateModel.predict` ignore it (`ModelMetadata` has no offset field).
Measured: with `offset_col="logprem"`, `rm.predict` differs from `fit.predict` by 99.9%.
`run_model` computes A/E, Gini and deviance from `rm.predict`, so a workbench project
with an offset role shows wrong metrics today. The docstring of `to_rate_model` claims
exactness "for any data".

**Why it matters.** E1 ("offset from current premium, fit a change") is built entirely
on this path; every downstream number would be wrong, and the exactness test would fail
on the first offset model.

**Change to the plan.** Add to the C format bump: `ModelMetadata.offset_col` and
`offset_is_log: bool`; `RateModel.predict` multiplies by `exp(offset)` (or by the raw
column when it is a log) and warns when the column is missing, exactly like exposure.
`to_rate_model` copies the fields from the fit. Add the offset case to
`tests/test_invariants.py`. Then E1 is a thin layer: a role "current premium" producing
a derived `log(current)` column and an export labelled "multiplier on current premium".

---

## 3. Should-fix issues

### S1. The offset "algebraic identity" test is only true under conditions the plan does not state (E)

Offset `log(P)` with weight 1 equals modelling `y/P` with weight `P` for the **Poisson**
deviance only, and glum then (a) divides the deviance by `Σw` (so `alpha` must be
rescaled by `ΣP/n`) and (b) standardises columns with the sample weights when
`scale_predictors=True`, which changes the per-column penalty. As written ("within
1e-8") the test fails and the builder's natural fix is to loosen it. State it as: Poisson,
`scale_predictors=False`, `alpha` rescaled, and then 1e-8 is achievable; for Gamma/Tweedie
say the two are *not* expected to match.

### S2. Run persistence needs data identity and library versions in the key (D1)

`model_hash` hashes the spec only. A pickled run under `.easyglm-runs/<hash>.pkl` would
be restored after the parquet file was replaced with new data, and the D1 test ("fit is
restored with identical predictions") would pass. Pickles of glum estimators also break
across glum/numpy versions. Put file size + mtime (or a content hash of the first N MB)
and `easy_glm`/`glum`/`polars` versions in the key; treat load failure as a cache miss;
document that the folder contains pickles (trusted local user, same as derived-column
`eval`).

### S3. Do D2 (sample vs full) before D1 (persistence), and fix the Design preview (D)

`model_hash` includes `data.sample_rows`, so D2 changes every run's key; done after D1 it
invalidates every persisted run. Also: if Design-page previews compute quantile knots on
the sample while the fit computes them on the full training set, the knots the user sees
are not the knots that are fitted. Compute knots on full training data once (quantiles on
5M rows are cheap) and use the sample only for the exposure/rate bars.

### S4. Excel round-trip needs a reader that does not exist (A, B)

The plan tests "Excel matrix sheet reads back to the same relativities" and "Excel
round-trips", but there is no `RateModel.from_excel`; `core/excel.py` only writes. Either
add a reader (then it is a product feature and needs its own tests for hand-edited
sheets) or say the test reads with `pl.read_excel` and compares. Decide; the first is what
actuaries will actually want.

### S5. Progress reporting: glum has no callback (G)

glum's `verbose` uses `tqdm`/`print`, not a hook. Per-iteration progress means capturing
stdout from a background thread; per-fold progress means running the CV folds ourselves,
which loses `deviance_path_`/`coef_path_` that `alpha_path` in `workflow/diagnostics.py`
reads. State the intended granularity ("one tick per alpha-path point of the final fit,
one per fold for CV") and the mechanism, or descope to a spinner with elapsed time.

### S6. Golden French motor test never runs in CI as written

It is "skipped when the cache is absent" and CI has no cache, so the one test that
protects results across refactors would only run on the builder's machine. Cache
`~/.cache/easy_glm` with `actions/cache` on one matrix leg, or check in a 50k-row
deterministic subsample as a test fixture and record golden numbers on that.

### S7. Interactions with `scale_predictors=True` favour thin cells (A)

glum standardises each column by its (weighted) standard deviation. A cell indicator
with 0.1% exposure has a tiny std, so its standardised coefficient buys a large raw
effect for little penalty: thin cells get *less* effective shrinkage than fat ones,
which is the opposite of what an actuary wants. The exposure threshold and `P1` mitigate
but the planted-truth test must include thin non-signal cells (10–50 rows) and assert
they stay at 1.0. Consider `scale_predictors=False` for interaction blocks (glum's `P1`
can carry the per-column weight instead).

### S8. Binomial in the workbench needs the link in the scorer (E3)

`_check_log_link` raises for logit today. Odds relativities are multiplicative, so the
tables work, but `RateModel.predict` must know `link="logit"` to return probabilities and
must refuse to multiply by exposure. Add `link` to the metadata bump in B1 or move E3 to
0.5.

### S9. Sequencing: fit the release to what the actuary asked for first

The plan puts G (open-ended, riskiest) before A and B (the two features the bike model
needs). Interaction cells are exactly the block that tabmat stores cheaply, so A does not
need G first. Suggested order: C (+ B1/B5 format bump) → A core/engine → B core/engine →
D2 → D1 → G (with the spike) → A/B workbench and export → D3 → D7 → E1–E2 → the rest.
After step 5 a coherent 0.4.0 could be cut if time runs out; D4–D6, E3–E4 and the docs
site can slip to 0.4.x without breaking the theme. The owner decided the workbench scope;
this is about order, not scope.

### S10. Working protocol: three additions

The loop is workable. Add: (1) *any change to an existing tolerance or golden number is
itself a blocking review item* and needs a written reason in the PR — this is the single
most likely way an actuary-invisible regression gets in; (2) the actuarial check must be
generated by a script committed with the PR (`scripts/checks/<piece>.py`) so the reviewer
re-runs it rather than trusting pasted numbers; (3) separate the reviewer's findings file
from the actuary's document (`docs/reviews/<piece>.md` for findings,
`docs/checks/<piece>.md` for the plain-language check) — the actuary should not have to
read code-review threads to find their questions. Keep "reviewer may not edit code".

---

## 4. Nits and suggestions

- Workstream letters run C, D, E, G, F; renumber.
- `_bin_rows` and `_modal_bins` dispatch with `assert isinstance(enc,
  CategoricalEncoder)`; asserts vanish under `python -O` and a third encoder kind falls
  into the categorical branch. Use explicit `if/elif/else: raise NotImplementedError`.
- CI sets `EASY_GLM_MAX_ROWS=500`; nothing in `src/` or `tests/` reads it.
- "`pip install easy_glm` is lighter than 0.3": beyond DuckDB, `matplotlib`, `seaborn`,
  `rdata`, `scikit-learn` and `joblib` are hard dependencies; moving the plotting ones to
  the `viz` extra is cheap (`test_imports.py` already asserts lazy import).
- `scripts/` holds eight one-off investigation scripts using the legacy API; delete
  them in C rather than porting.
- `Snapshot.metrics` is never populated; either fill it in `create_snapshot` (useful
  for the D3 compare page) or drop it.
- The plan's "drag-to-edit spike" verdict should state the acceptance up front (e.g.
  "edit a point, table updates, A/E recomputes, works in Streamlit 1.57 without a custom
  component") so "promoted if it works" is decidable.
- D5 "smoothing preserves the exposure-weighted mean relativity": say which mean —
  arithmetic mean of relativities or of log relativities; they differ and only the latter
  keeps the overall premium level when the base rate is not refitted.
- `Project.from_dict` overwrites `version` with `PROJECT_VERSION` unconditionally; the
  migration hook in B1 should branch on the incoming value first.

---

## 5. Missing tests and invariants (one line each)

- **A** Exactness with nulls in *both* parents and an unseen level of the categorical parent, in the same frame.
- **A** Unit: cell index of every row equals the parent's rate-table row index (shared helper, not re-derived).
- **A** `InteractionEncoder(a, b)` and `(b, a)` give identical predictions and transposed matrices.
- **A** `update_relativity` on a cell → snapshot → JSON → `switch_to` restores the exact matrix.
- **A** Excel matrix sheet carries training exposure per cell; lumped cells are 1.0 with exposure below threshold.
- **A** `P1` vector length equals `spec.n_features` and aligns with `spec.features`; an unpenalised main keeps a non-zero coefficient.
- **A** Planted truth includes thin non-signal cells (10–50 rows) that must stay within [0.98, 1.02].
- **B** NaN input produces a finite design (no NaN column) and the null row is used at scoring.
- **B** Below `lo` / above `hi`: scorer equals GLM (extrapolation policy test), including `x = lo` and `x = hi` exactly.
- **B** Slope of band `j` equals the cumulative sum of hinge coefficients up to `j`; continuity at every interior knot.
- **B** Editing one band-end relativity changes exactly the two adjacent slopes and nothing else; exactness holds after the edit.
- **B** `monotone` on a linear term raises (or the documented convex bound is applied) — not silent.
- **C** `import easy_glm; easy_glm.generate_blueprint` raises `AttributeError`; `pip show` has no duckdb; benchmark runner produces rows for all four families.
- **C** `RateModel.from_rate_tables(rate_tables(fit), base_rate)` predicts identically to `to_rate_model(fit)`.
- **C/B1** A `.easyglm` with `format_version: 3` is rejected; a version-less 0.3 file loads and scores identically to before.
- **C/B1** `predict` raises on an unknown `VariableConfig.type`.
- **C/B5** Offset model: `RateModel.predict == fit.predict` to 1e-10; JSON round-trip keeps `offset_col`.
- **G** With a float32 design, `RateModel.predict == fit.predict` still holds to 1e-10 (coefficients and linear predictor in float64).
- **G** Mixed-dtype inputs (`P1`, bounds, offset as float64) are cast, not rejected, and the `SplitMatrix` has one dtype.
- **G** Chunked `spec.build`/`predict` equals unchunked bitwise for chunk sizes 1, 7, and n.
- **G** Benchmark asserts the *design bytes* formula, not only RSS; CV budget asserted separately with `n_jobs=1`.
- **G** CV on the SplitMatrix selects the same alpha as dense float64 on French motor (or within one grid step, stated).
- **G** Lazy loader: same row set as the eager sample for the same seed (compare ids, not counts).
- **D1** Persisted run is ignored when the data file's size/mtime or library versions change; a corrupt pickle triggers a refit, not a crash.
- **D2** `model_hash` excludes sample settings; changing the sample leaves the run valid; `train_rows` equals the full training count.
- **D3** Relativity diff lists exactly bands with |Δ log rel| > tol; two identical runs give an empty diff.
- **D4** Report HTML contains no external `src=`/`href=` to `http(s)://` and a headless browser logs no console errors.
- **D5** Smoothing preserves the exposure-weighted mean of *log* relativities; cap/floor/round are idempotent; undo restores the previous snapshot byte-for-byte.
- **D7** `launch_editor` argument builder is a pure function; test that `--server.port` precedes `--`.
- **D7** Editor default A/E formula is derived from `target_is_rate` metadata; on a count target with exposure, overall A/E on train equals 1.0 within 1e-6.
- **E1** Identity test as specified in S1 (Poisson, unscaled, rescaled alpha) to 1e-8; a Gamma case documented as *not* matching.
- **E3** Binomial scorer returns probabilities in (0, 1); exposure multiplication refused; tables labelled "odds relativity".
- **E4** Base-rate solver is closed form; test with weights and with an existing `base_rate_override`.
- **F** `easy-glm run` on an invalid project exits non-zero; on a valid one writes a `.easyglm` byte-identical to the workbench's.
- **All** `tests/test_invariants.py` parametrised over step, categorical, linear, interaction, mixed, **with offset**, **with nulls and unseen levels** in every case.

---

## 6. Questions for the actuary (domain only; default if unanswered)

- **Q1 Linear terms beyond the data.** Above the largest training value (and below the smallest), should the curve stay flat or keep its last slope? *Default: flat (clamp), as rating engines usually do; the clamp points are shown in the table.*
- **Q2 Where relativity 1.00 sits on a continuous curve.** *Default: at the lower knot of the band with the most exposure, so it is a round, visible number.*
- **Q3 What the lasso should prefer on a curve.** Few *bends* (long straight sections, possibly all sloped) or few *slopes* (flat where the data does not insist)? *Default: few bends (the AGLM hinge basis in the plan); monotone constraints are then not offered on linear terms in 0.4.*
- **Q4 Thin interaction cells.** Minimum exposure for a cell to get its own adjustment? *Default: 0.5% of training exposure per interaction, editable; cells below it show 1.00 with their exposure alongside.*
- **Q5 Mains move when an interaction is added.** In a joint fit the main-effect tables change when `A × B` is added (the split between mains and cells is not unique). Acceptable, or should mains be frozen and the interaction fitted as a second stage? *Default: joint fit, with the before/after main tables shown in the actuarial check.*
- **Q6 Rate-change export.** With offset = log(current premium), should the export read as "multiplier on current premium" (base rate ≈ overall change, relativities = differential changes)? *Default: yes.*
- **Q7 Binomial tables.** Odds relativities with a label, or probabilities by band? *Default: odds relativities; the scorer returns probabilities.*
- **Q8 A/E for counts.** Confirm actual = Σ claims / Σ exposure and expected = Σ fitted claims / Σ exposure for frequency models. *Default: yes; the rate/count flag will be stored in the model file so the editor stops guessing.*
- **Q9 Which bike variables are linear.** *Default: mileage only, as in the original script; everything else step.*

---

## 7. Already broken in the current code (found while reading; the plan should fix these)

1. **Excel export ignores manual adjustments.** `app/ui.py::excel_bytes` calls
   `EasyGLM(run.fit, run.rate_model, run.tables).to_excel(...)` and `EasyGLM.to_excel`
   writes `self.relativities` = the *fitted* tables; the exported script does the same
   (`EasyGLM(fit, rm).to_excel`). Measured: after `update_relativity(..., 3.0)` the
   workbook shows 0.829 while the `.easyglm` scorer uses 3.0. An actuary who ships the
   Excel to the rating team ships unadjusted factors. Fix: `RateModel.to_excel` (already
   adjustment-aware) for the workbench and the script; keep `EasyGLM.to_excel` for the
   fitted view and label it. Add a test that Excel relativities equal `rm.variables`.
2. **Offsets are ignored by the RateModel** (B5). `to_rate_model`'s exactness claim is
   false for any fit with `offset_col`; workbench metrics are wrong for such projects.
3. **Integer-typed categorical columns score as `Other`.** `engine/_scoring.py::
   score_categorical` compares `series.to_numpy()` (ints) to string levels, so every row
   gets the fallback. `DesignSpec.from_data(..., categorical=["VehPower"])` on an integer
   column, or `EasyGLM.fit(categorical=[...])`, breaks exactness (measured 30% error, one
   distinct relativity used). The workbench is protected only because the `types` override
   casts to Utf8 in `prep`. Fix: cast to Utf8 in both scoring paths (`_mask_for_row`
   already does). Add an integer-categorical case to the exactness test.
4. **`RateModel.diff(v1, v2)`** returns `snapshots[v2 - 1].changes` and ignores `v1`;
   D5's "diff view between snapshots" would be built on it.
5. **Unknown table types score silently as categorical** (B1).
6. **`.easyglm` has no format version** (B1).
7. **`launch_editor`** puts `--server.port` after `--` (already in the plan as D7).
8. **Editor A/E default formula** is a metadata gap, not a default: `compute_actual_
   expected` with `sum_weighted` on a count target computes actual = Σ(claims × exposure)
   / Σ exposure and expected = Σ(pred × exposure × exposure) / Σ exposure. The
   `RateModel` cannot tell a rate target from a count target because
   `divide_target_by_weight` is not stored. Fix via the metadata bump (B1), then derive the
   formula; do not just flip the default.
9. **`_bin_rows` / `_modal_bins`** use `assert isinstance` for dispatch (see nits).
10. **CI** sets an environment variable nothing reads (`EASY_GLM_MAX_ROWS`).

None of the above is covered by an existing test, which is itself the finding: the
0.3 exactness suite covers step + categorical, no offset, string-typed categoricals only.
`tests/test_invariants.py` in the plan should be written to cover 1–3 and 8 *before* the
new table types are added, so the builder inherits a suite that already fails on today's
bugs.
