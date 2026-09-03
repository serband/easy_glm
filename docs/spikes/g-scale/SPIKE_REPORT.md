# Workstream G scale spike: how easy_glm 0.4 should build and fit 1M–5M-row designs

Date: 2026-09-02. Machine: Apple M4, 10 cores, 24 GB RAM (about 15 GB free during the
runs; hard stop at 60 % = 14.4 GB per process). Software: the repo's venv — glum 3.4.1,
tabmat 4.2.1, numpy 2.4.5, polars 1.40.1, Python 3.14. Nothing in the easy_glm repo was
modified. Everything here is re-runnable with `bench.py` (see the last section).

---

## 1. Summary for the non-programmer

**The question.** Today easy_glm turns every policy row into a wide row of 0/1 numbers
(about 170 columns for a typical model) stored as 8-byte numbers. At 5 million policies
that table is about 6.7 GB before anything else, and the plan asked whether we should
(a) store it in 4-byte numbers, (b) store the categorical parts more cleverly, (c) add
up identical policies before fitting (the old Emblem trick), or (d) stop storing the
0/1 table at all and keep only, per policy and per rating factor, *which band it falls
in* (a small whole number), letting the fitting engine reconstruct what it needs on the
fly.

**What we found, in one table.** Same model, same data, 5 million synthetic policies,
169 columns, fixed penalty; "peak memory" is the process high-water mark; "difference"
is the largest relative difference in predicted frequency against today's method.

| approach | peak memory | time to fit | difference vs today | verdict |
|---|---|---|---|---|
| today: dense 8-byte table | 5.6 GB (and the machine started paging: 103 s) | 103 s | — | works, but at the edge of this machine |
| dense 4-byte table | 5.9 GB | 180 s, did not converge | 0.4 % | **not viable** |
| categorical blocks compressed, 8-byte steps (tabmat SplitMatrix) | 6.0 GB | 9 s | 0.000 000 000 01 % | viable fallback |
| same, 4-byte | 4.6 GB | 172 s, did not converge | 0.2 % | **not viable** |
| add up identical policies (aggregation) | no gain on this data (ratio 1.00); 1.5× on French motor | — | exact | useful option, not a default |
| **band index only ("StepMatrix"), 8-byte arithmetic** | **2.0 GB** | **16 s** | **0.000 000 000 01 %** | **recommended** |

**What this means.**

* Four-byte storage looked attractive on paper (half the memory) but the fitting engine
  cannot finish the job in 4-byte arithmetic on a million or more rows: it runs to its
  iteration limit and stops 0.1–0.4 % away from the correct answer, and it is 10–20×
  slower. On 200k rows it appears to work only because the stopping rule is loose. We
  should not ship it.
* Keeping only the band index per policy is exact (identical coefficients to 13 decimal
  places), uses one-third of the memory of today's method at 5M rows (2.0 GB, inside the
  plan's 3 GB target), and is fast enough (16 s at 5M, 2.6 s at 1M). It needs about 150
  lines of new code plus a small compatibility shim for the fitting library.
* Adding up identical policies is mathematically exact for every model family we use
  (Poisson, Gamma, Tweedie), and we verified it to 13 decimal places on French motor.
  But real rating data has fine variables (population density, bonus-malus, region), so
  the saving is small: 1.5× on French motor with the standard 20-band design, and
  nothing at all on continuous synthetic data. It should be an *option* for coarse
  designs, not the default path.
* An important side finding for the acceptance criteria: the fitting engine's *default
  stopping rule* alone moves predictions by 0.1–0.4 % (default tolerance versus a tight
  one, both in 8-byte arithmetic). Any acceptance test phrased as "predictions agree to
  0.01 %" between two methods is only meaningful if both use identical arithmetic — which
  the recommended path does (agreement is then 1e-13, not 1e-4).

**Recommendation for 0.4.** Represent step variables as band indices (a `StepMatrix`
tabmat block), categoricals as tabmat `CategoricalMatrix`, all arithmetic in float64,
no float32 anywhere; keep aggregation as an opt-in for coarse designs; make scoring
always go through the float64 rate-table lookup (never through the fitted matrix). On
this machine that gives 1M rows in 0.7 GB / 2.6 s and 5M rows in 2.0 GB / 16 s, with
3-fold CV at 1M in 0.8 GB / 32 s.

---

## 2. Set-up (technical from here on)

* **Data.** Synthetic motor frequency: 6 numeric rating factors (driver age, vehicle
  age, bonus-malus, density, power, mileage; integer-valued, skewed, 1 % nulls in one),
  3 categoricals (20/12/10 levels, skewed, 1 % nulls in one, rare levels lumped to
  `Other`), exposure U(0.05, 1), Poisson claims at ~9 % frequency with planted effects.
  Cached per size in `out/data_{n}.parquet`; spec per size in `out/spec_{n}.json`.
* **Design.** `DesignSpec.from_data(..., n_bins=30, weight_col="exposure")` — quantile
  knots on integer data give fewer knots than requested, so 167 columns at 200k and 169
  at 1M/5M (125–127 step + null columns, 42 categorical columns). This is the
  "~150–200 columns" band of the brief.
* **Fit.** glum `GeneralizedLinearRegressor(family="poisson", link="log", alpha=3e-4,
  l1_ratio=1, scale_predictors=True)`, other settings default; the CV rows use
  `GeneralizedLinearRegressorCV(cv=3, n_alphas=10, min_alpha_ratio=1e-3)`. Target is
  `claims/exposure` with `sample_weight=exposure`, as `fit_glm(divide_target_by_weight=True)`
  does. `alpha=3e-4` gives a properly sparse AGLM solution (95/167 non-zero at 200k).
* **Measurement.** One fresh subprocess per (candidate, size); `ru_maxrss` at exit;
  psutil checkpoints after load / build / fit; a watchdog thread kills the child above
  14.4 GB (never triggered). Predictions for the comparison are always recomposed in
  float64 from the coefficients by rate-table lookup (`spike_lib.predict_from_codes`,
  the same arithmetic `RateModel.predict` uses), so the "difference" columns measure
  *coefficient* differences, not matrix-multiply noise. A separate column
  (`glumVsF64`) records how far glum's own `model.predict(X)` is from that float64
  recomposition.
* **Candidates.** `baseline` = `spec.build()` dense float64 (today's code path);
  `dense64c` = same matrix written straight from bin codes (no `hstack` transients);
  `dense32`; `split64`/`split32` = tabmat `SplitMatrix` (one dense step block +
  `CategoricalMatrix` per categorical, `drop_first=True`); `agg`/`agg_split32` = rows
  aggregated by identical design row; `stepmat64`/`stepmat32` = `StepMatrix` blocks
  (prototype, `spike_lib.StepMatrix`) + null-indicator dense block + `CategoricalMatrix`
  blocks; `*_tight` = `gradient_tol=1e-8` (compared with `baseline_tight`); `*_cv`.

## 3. Measured table

Columns: build/fit/total wall seconds; peak = `ru_maxrss`; design = bytes held by the
matrix representation; it = IRLS iterations (100 = hit `max_iter`, not converged);
nnz = non-zero coefficients; maxRel / p99Rel = max and 99th-percentile relative
prediction difference vs `baseline` (vs `baseline_tight` for `*_tight`); nnzΔ = number
of coefficients whose zero/non-zero status differs; glumVsF64 = `model.predict(X)` vs
float64 recomposition.

```
candidate              n    p status  build_s   fit_s total_s peakGB  desGB  it  nnz maxRelPred p99RelPred nnzΔ glumVsF64  ratio
baseline          200000  167 ok          0.2     0.3     0.7   0.84   0.25   4   95                              1.5e-15
baseline_tight    200000  167 ok          0.1     0.3     0.5   0.78   0.25   5   95    3.6e-03    3.0e-03    0   1.4e-15
dense64c          200000  167 ok          0.1     0.2     0.3   0.80   0.25   4   95    2.5e-14    1.5e-14    0   1.5e-15
dense32           200000  167 ok          0.0     0.1     1.3*  0.55   0.12   4   95    5.7e-06    3.3e-06    0   4.5e-07
split64           200000  167 ok          0.1     0.2     0.4   0.68   0.19   4   95    2.6e-14    1.5e-14    0   1.5e-15
split32           200000  167 ok          0.1     0.1     1.3*  0.49   0.10   4   95    3.8e-06    2.0e-06    0   4.5e-07
agg               200000  167 ok          0.1     0.2     0.3   0.85   0.25   4   95    2.4e-14    1.4e-14    0   1.5e-15   1.00
agg_split32       200000  167 ok          0.1     0.2     0.3   0.54   0.10   4   95    3.2e-06    1.7e-06    0   4.5e-07   1.00
stepmat64         200000  167 ok          0.1     0.3     0.4   0.34   0.02   4   95    2.4e-14    1.4e-14    0   1.5e-15
stepmat32         200000  167 ok          0.1     0.3     0.4   0.33   0.01   4   95    5.8e-07    3.3e-07    0   3.5e-07
dense32_tight     200000  167 ok          0.0    40.8    41.9   0.55   0.12 100   95    5.3e-06    3.1e-06    0   4.2e-07
split32_tight     200000  167 ok          0.1    42.0    43.1   0.49   0.10 100   95    2.3e-06    1.2e-06    0   3.6e-07
stepmat32_tight   200000  167 ok          0.1    44.7    44.8   0.32   0.01 100   95    1.6e-06    1.2e-06    0   3.2e-07
baseline         1000000  169 ok          0.9     1.3     3.6   2.81   1.26   4   76                              1.5e-15
baseline_tight   1000000  169 ok          0.6     1.4     2.4   2.91   1.26   5   75    1.4e-03    1.4e-03    1   1.5e-15
dense64c         1000000  169 ok          0.2     0.9     1.4   3.00   1.26   4   76    2.1e-14    1.3e-14    0   1.7e-15
dense32          1000000  169 ok          0.2     0.6     0.9   1.75   0.63   4   76    2.3e-04    1.1e-04    0   5.1e-07
split64          1000000  169 ok          0.4     0.9     1.4   2.39   0.96   4   76    2.0e-14    1.2e-14    0   1.8e-15
split32          1000000  169 ok          0.3     0.6     1.1   1.46   0.48   4   76    1.1e-04    6.1e-05    0   3.8e-07
agg              1000000  169 ok          0.3     0.9     1.5   3.24   1.26   4   76    2.2e-14    1.2e-14    0   1.9e-15   1.00
agg_split32      1000000  169 ok          0.4     0.7     1.2   1.69   0.48   4   76    1.1e-04    6.1e-05    0   4.1e-07   1.00
stepmat64        1000000  169 ok          0.3     2.2     2.6   0.69   0.08   4   76    2.2e-14    1.2e-14    0   1.5e-15
stepmat32        1000000  169 ok          0.3     2.0     2.4   0.61   0.06   4   76    6.7e-07    3.5e-07    0   3.4e-07
dense32_tight    1000000  169 ok          0.2    31.0    31.4   1.74   0.63 100   75    1.7e-03    1.0e-03    0   4.4e-07
split32_tight    1000000  169 ok          0.3    31.5    32.0   1.45   0.48 100   75    1.6e-03    1.0e-03    0   4.4e-07
stepmat32_tight  1000000  169 ok          0.3    53.1    53.5   0.61   0.06 100   75    3.1e-03    2.0e-03    0   3.6e-07
baseline_cv      1000000  169 ok          0.7    15.9    17.0   3.01   1.26   2   86    (CV alpha 1.83e-4; not comparable)
split64_cv       1000000  169 ok          0.4    15.4    16.0   2.52   0.96   2   86    same alpha and nnz as baseline_cv
stepmat64_cv     1000000  169 ok          0.3    32.0    32.3   0.80   0.08   2   86    same alpha and nnz as baseline_cv
split32_cv       1000000  169 ok          0.4   374.0   374.6   1.55   0.48 100   86    final refit hit max_iter
baseline         5000000  169 ok         13.7    77.6   102.8   5.64   6.30   4   79                              1.5e-15
dense32          5000000  169 ok          1.1   177.1   179.7   5.85   3.15 100   80    3.8e-03    1.8e-03    1   4.2e-07
split64          5000000  169 ok          1.8     5.0     9.1   5.97   4.79   4   79    1.5e-13    8.9e-14    0   1.9e-15
split32          5000000  169 ok          1.7   168.8   171.7   4.63   2.42 100   79    2.2e-03    1.1e-03    0   4.8e-07
agg_split32      5000000  169 ok          2.2   179.2   182.8   4.72   2.42 100   79    2.8e-03    1.5e-03    0   3.8e-07   1.00
stepmat64        5000000  169 ok          1.6    14.2    16.1   2.04   0.39   4   79    1.2e-13    6.8e-14    0   1.8e-15
stepmat32        5000000  169 ok          1.6   291.2   293.1   1.71   0.28 100   79    2.7e-03    1.9e-03    0   3.4e-07
```
`*` total includes the dtype side-probes run inside that child (see §4.2).

**French motor (cached parquet, 677 991 rows, 9 predictors as listed in the brief,
`n_bins=20`, p = 112):**

| item | value |
|---|---|
| distinct design rows / compression ratio | 440 361 → **1.54×** |
| ratio at `n_bins=10` (p=81) / 20 (p=112) / 30 (p=140) | 2.23× / 1.54× / 1.38× |
| ratio without Density / without Density+BonusMalus / without those + Region | 1.78× / 2.31× / 5.84× |
| Poisson, `scale_predictors=True`: aggregated vs row-level, max rel prediction diff | **2.0e-13** (coef 1.2e-13, same 64 non-zeros, 5 iterations both) |
| same with `gradient_tol=1e-8` | 2.2e-13 |
| same with `scale_predictors=False` | 2.0e-14 (46 non-zeros — the unscaled penalty selects a different set, as expected) |
| Gamma severity (synthetic amounts on the 4.6 % of rows with claims, weight = claim count) | 1.1e-15; compression 1.06× |
| row-level default tolerance vs tight tolerance (both float64) | 1.7e-3 |
| time row-level / aggregated | 0.91 s / 0.40 s |

## 4. Findings per candidate

### 4.1 Baseline (dense float64, `spec.build`)
* Exact by definition. At 5M rows the matrix alone is 6.3 GB; `spec.build` adds
  `StepEncoder.transform` transients (bool → float64 → `hstack`, ≈3 × 8 × n × K_max ≈
  3.5 GB for a 29-knot variable) and the 13.7 s build time shows it. Reported peak
  (5.64 GB) is *below* the matrix size because macOS compressed part of it; the fit took
  77.6 s versus 5.0 s for `split64` on the same rows — the machine was paging (swap in
  use: 0.48 GB). So the current path "works" at 5M on a 24 GB laptop only while the
  machine has ~10 GB free, and it is 10× slower than it should be.
* `dense64c` (same matrix, no transients) needs 1.4 s at 1M instead of 3.6 s; the
  transients are pure waste.

### 4.2 Dense float32 — not viable
* **Correctness.** At 200k rows and default tolerance the fit converges in 4 iterations
  and differs from float64 by 5.7e-6. At 1M it is 2.3e-4; at 5M it hits `max_iter=100`
  (177 s) and ends 3.8e-3 away. Tightening the tolerance does not help: at
  `gradient_tol=1e-8` every float32 variant hits `max_iter` already at 200k rows (41–45 s
  instead of 0.3 s) and at 1M ends 1.6–3.1e-3 from the float64 tight fit. The float32
  gradient norm has a noise floor (sums over n rows in float32) that sits above glum's
  stopping criterion once n ≳ 1M; below that, the "agreement" is an artefact of stopping
  early in both arithmetics. This is the reviewer's B4 point 1, now with the mechanism.
* **glum/tabmat defects found.** (i) glum 3.4.1 casts `y` to `X.dtype` but
  `check_weights` calls `sklearn.check_array(sample_weight, dtype=[float64, float32])`,
  which *keeps* a float64 weight vector. tabmat's Cython kernels then receive mixed
  dtypes: `CategoricalMatrix.transpose_matvec` raises `Buffer dtype mismatch`, and
  `DenseMatrix._get_col_stds` (`transpose_square_dot_weights`) **segfaults** (return
  code −11, C- and F-order alike). Work-around: cast `y` and `sample_weight` to float32
  before calling glum. (ii) `P1` in float64 with a float32 design raises `TypeError`
  ("The given P1 cannot be converted..."); `offset` and `lower/upper_bounds` in float64
  are cast silently. (iii) `coef_` and `intercept_` come back as `np.float32`. (iv) A
  `SplitMatrix` mixing a float32 dense block with a float64 `CategoricalMatrix` is
  *constructed* without error in tabmat 4.2.1 (the dtype check reads
  `flatten_matrices[0].dtype` and then compares — it did not raise in our test) and fails
  later inside Cython — i.e. the single-dtype rule is enforced only at use time.
* **Exactness invariant.** `model.predict(X32)` differs from the float64 recomposition
  of the same coefficients by 3–5e-7 (column `glumVsF64`), so `fit.predict` must never
  score through the float32 matrix. This is moot once float32 is dropped, but the same
  rule (score from coefficients in float64, never via `model.predict(X)`) is what keeps
  the `RateModel.predict == fit.predict` invariant exact for every representation.

### 4.3 tabmat SplitMatrix (CategoricalMatrix + dense step block)
* Float64: exact (1.5e-13 at 5M, same non-zero set), fastest fit of all at 5M (5.0 s;
  tabmat's dense sandwich is blocked C code), 1.4 s at 1M. Memory: the categorical
  blocks are 4 bytes/row/variable instead of 8 × levels, but 125 of 169 columns are
  step columns and stay dense, so the saving is only 24 % (4.79 vs 6.30 GB design;
  peak 5.97 vs 5.64 GB, the latter deflated by paging).
* Float32: same non-convergence as dense float32 (100 iterations, 2.2e-3 off at 5M).
* CV at 1M: `split64_cv` 16.0 s / 2.52 GB versus `baseline_cv` 17.0 s / 3.01 GB — no
  slowness in the float64 SplitMatrix CV path. The **374 s** of `split32_cv` is float32:
  the final refit at the chosen alpha ran to `max_iter=100` (fixed-alpha `split32`
  at 1M took 1.1 s only because default tolerance stopped it after 4 iterations; the CV
  path warm-starts along the alpha path and the tolerance is effectively tighter near
  the end). Same alpha and the same 86 non-zeros were selected by all four CV runs.
* Construction notes: `CategoricalMatrix(codes, categories=arange(L+1), drop_first=True)`
  reproduces easy_glm's "reference level has no column, `Other` is the last column"
  layout exactly (`split.toarray() == spec.build(df)` verified).

### 4.4 Row aggregation (Emblem trick)
* **Exactness.** Proven (appendix C) and measured: 2e-13 on French motor Poisson with
  `scale_predictors=True` and `False`, at default and tight tolerance, and 1e-15 on a
  Gamma severity fit. Iteration counts are identical, so the solver path is the same.
* **Compression.** Nil on the synthetic data (1.0000 — 21 bins^6 × 42 levels leaves
  every row unique) and **1.54× on French motor** with the standard design; 2.2× with
  10 bins per variable; 5.8× only after dropping Density, BonusMalus and Region. Modern
  rating data has at least one fine variable (density, credit score, mileage), so the
  realistic gain is 1.3–2×. The grouping itself costs memory (polars `group_by` over
  n rows plus a row→group map: `agg` peaked at 3.24 GB at 1M, above the 2.81 GB
  baseline) and time (0.3 s at 1M) — with no compression it is a pure loss.
* **Consequences.** Not a default path. Offer it as `fit_glm(..., aggregate=True)`
  (or automatic when a cheap `n_unique` probe shows ratio ≥ 2), with: the offset column
  (if any) added to the grouping key; CV folds assigned on *rows* before aggregation (so
  CV has to use easy_glm's own fold loop, not `GeneralizedLinearRegressorCV`);
  diagnostics computed on `pred[group_of_row]`. Family-independent — the same code
  serves Poisson/Gamma/Tweedie/binomial.

### 4.5 Bin-index StepMatrix (cumulative-sum trick) — recommended
* **Interface size.** Small enough to implement in the spike: `spike_lib.StepMatrix`
  is ~170 lines and implements the 9 abstract `MatrixBase` methods plus
  `_cross_sandwich`. It passed equality tests against the dense matrix for `matvec`,
  `transpose_matvec`, `sandwich` (full and with random `rows`/`cols` subsets, error
  ≤ 2e-9 on sums of 200k terms), `standardize` (means/stds to 1e-12), `__getitem__`
  row subsets (CV folds), and full glum fits at 200k/1M/5M including 3-fold CV.
* **Results.** Identical coefficients to float64 dense (1.2e-13 at 5M, same non-zeros).
  Design 0.39 GB at 5M (int32 codes 6 × 4 B/row + null-indicator dense block 6 × 8 B/row
  + categorical 3 × 4 B/row); peak **2.04 GB at 5M**, 0.69 GB at 1M, 0.80 GB for 3-fold
  CV at 1M. Fit time 14.2 s at 5M and 2.2 s at 1M — about 3× tabmat's C dense sandwich
  (5.0 s / 0.9 s) because the prototype is pure numpy (bincounts per IRLS iteration, one
  scipy-sparse product per StepMatrix × dense-block pair, 10 blocks → 45 cross products
  per sandwich). CV at 1M: 32 s vs 16 s. Acceptable as is; two easy optimisations are
  listed under risks.
* **Float32 variant** behaves like the others (non-convergent at 5M) even though the
  StepMatrix sums themselves are float64 — the rest of glum's IRLS runs in the design
  dtype. No reason to keep it.
* **Obstacle.** glum's `check_array_tabmat_compliant` does not know the class and would
  hand it to `sklearn.check_array`; the spike patches that function
  (`spike_lib.patch_glum_validation`). Production options: a one-line upstream PR
  (`isinstance(mat, tm.MatrixBase): return mat`), or the same patch applied in
  `easy_glm.core.fit` at import time behind a glum version pin.

### 4.6 Solver tolerance (cross-cutting)
Default `gradient_tol` vs `1e-8`, both float64 dense: 3.6e-3 (200k), 1.4e-3 (1M),
1.7e-3 (French motor), with one coefficient changing zero-status at 1M. Whatever the
representation, this is the reproducibility floor of a *refit* unless tolerances are
pinned. R4's "predictions agree to 1e-4" must therefore be stated between two float64
representations of the *same* solver run (where it is 1e-13), not as a float32
criterion.

## 5. Memory arithmetic

Notation: `n` rows; `p` design columns; `p_s` step + null columns (125–127 here);
`v_s` step variables (6); `v_c` categorical variables (3); `L` levels per categorical.
Design bytes measured in `results.json` (`design_bytes`) match these formulas exactly.

| representation | design bytes | 5M × 169 | measured peak at 5M |
|---|---|---|---|
| dense float64 (today) | `8·n·p` (+ build transients ≈ `24·n·K_max`) | 6.3 GB (+3.5 GB transient) | 5.64 GB, paging |
| dense float32 | `4·n·p` | 3.15 GB | not viable |
| SplitMatrix float64 | `8·n·p_s + 4·n·v_c` | 4.79 GB | 5.97 GB |
| SplitMatrix float32 | `4·n·p_s + 4·n·v_c` | 2.42 GB | not viable |
| **StepMatrix + Categorical, float64 (prototype)** | `4·n·v_s + 8·n·v_s(null cols) + 4·n·v_c` | 0.39 GB | **2.04 GB** |
| same with null folded into the code and int16 codes (follow-up) | `2·n·v_s + 4·n·v_c` | 0.12 GB | ≈1.8 GB |
| aggregation, ratio r | any of the above with `n/r`, plus `≈ 40·n` for the grouping | r = 1.0–1.5 realistic | — |

Process peak ≈ raw frame + design + glum working set. The working set is ~20 float64
vectors of length n (`y`, weights, eta, mu, gradient rows, Hessian rows, active-row
masks, …) plus the int32 codes: measured **≈170–180 B/row** (stepmat64 at 5M: 2.04 GB
peak − 0.82 GB after load − 0.39 GB design ≈ 0.83 GB / 5M). The synthetic raw frame is
≈120 B/row in memory (11 columns incl. strings). CV with `n_jobs=1`: glum copies the
training fold (`X[train_idx, :]`), i.e. up to +2/3 of the design for dense/split (measured
+0.13–0.20 GB at 1M) and only +2/3 of the *codes* for StepMatrix (measured +0.11 GB).

**Budget formula for the recommended path** (float64 StepMatrix + Categorical):

    peak_bytes ≈ frame_bytes + n · (4·v_s + 8·n_null + 4·v_c) + n · 180   (+ n · (4·v_s + 4·v_c)·2/3 for CV)

## 6. Achievable budgets on this machine (24 GB, cap 14.4 GB)

| rows | today (dense f64) | SplitMatrix f64 (fallback) | **StepMatrix f64 (recommended)** |
|---|---|---|---|
| 1M | 2.8 GB / 3.6 s; CV 3.0 GB / 17 s | 2.4 GB / 1.4 s; CV 2.5 GB / 16 s | **0.7 GB / 2.6 s; CV 0.8 GB / 32 s** |
| 5M | 5.6 GB reported but paging / 103 s (needs ≈10 GB free) | 6.0 GB / 9.1 s | **2.0 GB / 16 s** (CV extrapolated ≈2.5 GB / ~3 min) |
| headroom | ~7M rows before the 14.4 GB cap (with transients) | ~11M rows | ~30M rows by arithmetic (0.36 GB per 1M rows incl. frame); practically limited by the raw frame and fit time (~3 s per 1M rows) |

The plan's "5M × ~200 columns in < 3 GB peak" is **met by the StepMatrix path with
float64** (2.04 GB at 169 columns; column count barely matters for StepMatrix since
memory scales with variables, not knots) and not met by any float32 variant or by the
SplitMatrix alone.

## 7. Recommendation for 0.4

1. **Representation:** tabmat `SplitMatrix` whose blocks are, in this order: one
   `StepMatrix` per step variable (band index per row), one dense block holding the null
   indicators (or fold the null into the StepMatrix as an extra column in a follow-up),
   one `CategoricalMatrix` per categorical (`drop_first=True`, code 0 = reference, last
   code = `Other`). Block order matters (`_cross_sandwich` dispatch, appendix A).
2. **Dtype:** float64 everywhere. Delete the float32 option from the plan; do not
   accept float32 designs in `DesignSpec.build`. `P1`, bounds and offset stay float64,
   `coef_` stays float64, and the exactness invariant is untouched.
3. **Scoring:** `GLMFit.predict`/`linear_predictor` compute `intercept + Σ table_v[code_v]`
   in float64 from the codes (what `RateModel` already does), chunked by rows for free
   (codes are cheap to compute per chunk). Never `model.predict(X)`. This makes
   `RateModel.predict == fit.predict` hold by construction (measured 1e-15 here) and
   removes the "second full matrix" that §G's chunked-scoring item worried about.
4. **Aggregation:** ship as an opt-in `aggregate=True` (or auto when a probe shows
   ratio ≥ 2), exact for all families, offset in the key, CV folds on rows. Not default.
5. **Acceptance criteria (replace R4's):** for every design in the invariant suite,
   `StepMatrix`/`SplitMatrix` fit coefficients equal the dense float64 fit to 1e-10
   (same glum settings, same data) and the non-zero set is identical; 5M × ~170 columns
   fits in < 3 GB peak (assert the arithmetic in the benchmark and measure on CI Linux,
   where RSS is not deflated by memory compression); French-motor holdout deviance/Gini
   identical to 1e-10, not 1e-4.
6. **Fallback:** if the glum validation shim is judged unacceptable, ship `split64`
   (exact, fastest, 6 GB at 5M) and defer StepMatrix to 0.4.x — the code is the same
   apart from the step blocks.

## 8. Risks

* **glum private-API shim.** `check_array_tabmat_compliant` is not public; pin
  `glum==3.4.*` and open the upstream PR early (one `isinstance(mat, MatrixBase)` line).
  Detect breakage with a test that fits a two-block SplitMatrix through `fit_glm`.
* **StepMatrix speed.** 3× slower than tabmat's C sandwich at 5M (14 s vs 5 s). Two
  cheap fixes if it matters: (a) one multi-variable `StepBlock` holding all step
  variables (turns 45 pairwise cross products into a handful of joint bincounts),
  (b) `np.bincount` on a combined `code_a·(K_b+2)+code_b` key is already the hot path —
  a small numba/Cython kernel would make it C-speed. Neither is needed for 0.4.
* **`rows`/`cols` subset semantics.** glum passes `active_rows` when `hessian_approx>0`
  and `active_cols` always; both are implemented and tested against dense with random
  subsets, but the test must live in the repo's suite, not only in the spike.
* **Measurement caveat.** On macOS `ru_maxrss` understates when the compressor is
  active (baseline at 5M). The benchmark should assert the arithmetic and record
  `t_fit` as a paging detector (fit time per row should not grow with n).
* **Aggregation subtleties.** Offsets and any per-row quantity must join the key; CV
  needs row-level folds; per-row diagnostics need the `group_of_row` map; the reported
  deviance constant differs (compare deviance *differences* only).
* **Float32 stays tempting.** Document the four defects in §4.2 in the code so nobody
  re-adds it; the segfault in particular is silent data-dependent memory unsafety.
* **`_modal_bins` at 5M.** It loops in Python over `to_list()` for categoricals — O(n)
  Python; at 5M that is tens of seconds. Replace with the same `replace_strict` codes.

## 9. Code changes the builder needs

`src/easy_glm/core/design.py`
* `Encoder.codes(series) -> np.ndarray[int32]` as the primary method: `StepEncoder`:
  `searchsorted(knots, x, side="right")`, null → `K+1`; `CategoricalEncoder`:
  `cast(Utf8).replace_strict({level: i}, default=L)`, null → `L`. Keep `transform`
  (dense) implemented *from* codes for tests, tiny data and the exported script.
* `Encoder.n_codes`, `Encoder.tables(coef_slice) -> np.ndarray` (the float64 lookup
  table: `[0, cumsum(step), null]` / `[0, *levels, other]`) — shared by `GLMFit`,
  `RateModel` and `tables.py`.
* `DesignSpec.codes(data) -> dict[str, np.ndarray]`.
* `DesignSpec.build(data) -> tm.SplitMatrix` (StepMatrix blocks first, null dense block,
  CategoricalMatrix blocks; float64 only) and `DesignSpec.build_dense(data)` (today's
  behaviour, written straight into a preallocated array — no `hstack`).
* `DesignSpec.linear_predictor(codes, coef, intercept)` in float64 (chunk-safe).
* New module `core/stepmatrix.py`: the `StepMatrix` class from `spike_lib.py`
  (matvec / transpose_matvec / sandwich / `_cross_sandwich` for Step, Categorical,
  Dense / getcol / toarray / astype / `_get_col_stds` / `__getitem__` / names), plus the
  glum validation shim `install_glum_shim()` guarded by the glum version.

`src/easy_glm/core/fit.py`
* `fit_glm`: `design = spec.build(data)` (SplitMatrix); call `install_glum_shim()`
  once; optional `aggregate: bool | "auto"` branch (group codes + `w`, `w·y`, offset
  key; fit on `W`, `ybar`; store `n_train_rows` = original rows and the ratio in
  `GLMFit`); everything else unchanged (bounds, `P1`, monotone, CV via glum for the
  non-aggregated path).
* `GLMFit.design_matrix` → returns the tabmat matrix (or drop it); `linear_predictor`
  and `predict` → `spec.codes` + `spec.linear_predictor` in float64 with the stored
  offset, in row chunks (`chunk_rows=500_000`) — no `model.predict`.
* `_modal_bins` → reuse `spec.codes` + `np.bincount` (removes the Python loop).
* `coef` unchanged (float64 view). Progress hook unchanged.

Tests to add: StepMatrix equals dense for the three ops with random `rows`/`cols`
(from `test_lib.py`); `fit_glm` on SplitMatrix equals `fit_glm` on `build_dense` to
1e-10 for the invariant-suite designs; aggregated fit equals row-level fit to 1e-10 for
Poisson, Gamma and Tweedie with weights; a `-m slow` 5M benchmark asserting the
arithmetic of §5 and a fit-time-per-row bound.

---

## Appendix A. What glum actually needs from a design matrix

glum 3.4.1 wraps whatever it is given with `tabmat.as_tabmat` and then only uses
(`_glm.py`, `_solvers.py`, `_linalg.py`, `_glm_cv.py`):

| call | where | notes |
|---|---|---|
| `X.standardize(w, center, scale)` | `_utils.standardize` | base-class method; needs `transpose_matvec(w)` (means) and `_get_col_stds(w, means)` |
| `X.matvec(coef, cols)` | linear predictor, `StandardizedMatrix.matvec` | `cols` = active set |
| `X.transpose_matvec(v, rows, cols, out)` | gradient, `StandardizedMatrix.transpose_matvec` | `rows` = active rows when `hessian_approx > 0` |
| `X.sandwich(d, rows, cols)` | `build_hessian_delta` → `_safe_sandwich_dot` | `rows`/`cols` subsets; may return a `dia_matrix` |
| `X[train_idx, :]` | `_glm_cv.py` 663/670 | row subset per fold (a copy for dense/categorical) |
| `X.dtype`, `X.shape`, `get_names` | everywhere | one dtype per `SplitMatrix` |
| `X.unstandardize()`, `X.toarray()` | end of fit; `_glm.py` 1671 only with diagnostics | |

`tabmat.MatrixBase` abstract methods: `matvec`, `transpose_matvec`, `sandwich`,
`getcol`, `toarray`, `astype`, `_get_col_stds`, `__getitem__`, `get_names` (9). Not
abstract but required inside a `SplitMatrix`: `_cross_sandwich(other, d, rows, L_cols,
R_cols)`. `SplitMatrix.sandwich` calls `mat_i._cross_sandwich(mat_j)` only for `i < j`,
and `DenseMatrix._cross_sandwich` / `CategoricalMatrix._cross_sandwich` raise
`TypeError` for an unknown block type — so a custom block must come *first* in the
block list and implement the cross products with Dense, Categorical and its own type.
`_combine_matrices` merges Dense blocks with each other (and Sparse with Sparse) but
leaves unknown types alone, so the ordering survives construction.

glum's `check_array_tabmat_compliant` (`_validation.py`) passes through only
`SplitMatrix`, `CategoricalMatrix`, `StandardizedMatrix`, `DenseMatrix`, `SparseMatrix`;
any other `MatrixBase` goes to `sklearn.check_array`. `predict` calls it with
`copy=True` (copies `SplitMatrix.indices`, cheap) — another reason not to score through
glum.

## Appendix B. StepMatrix: the three operations

For one step variable with knots `k_1 < … < k_K` and per-row bin code
`b_i ∈ {0..K}` (`b_i = #{j : x_i ≥ k_j}`; null rows carry code `K+1` and appear in no
step column), column `j` is `1{b_i ≥ j}` for `j = 1..K`.

* `X β`: `c = concat([0], cumsum(β), [0])`; result `c[b]` — one gather.
* `Xᵀ v`: `t = bincount(b, weights=v)[0..K]`; `S = reverse_cumsum(t)`; result `S[1..K]`.
* `Xᵀ diag(d) X`: with `S` computed from `d`, `(XᵀDX)[j,l] = S[max(j,l)]` — a `K×K`
  table filled from a length-`K` vector.
* Cross with another StepMatrix: joint `bincount(b₁·(K₂+2) + b₂, d)` then a 2-D
  reverse cumsum; with a `CategoricalMatrix`: joint bincount with its `indices`, 1-D
  reverse cumsum along the bin axis (drop the first column if `drop_first`); with a
  `DenseMatrix` block `Y`: per-bin column sums of `d·Y` (sparse indicator matmul), then
  reverse cumsum.
* `rows` subsets: index `b` and `v`/`d` by `rows` first; `cols` subsets: slice the
  result (zero-fill `β` outside `cols` for `matvec`).
* Column stds: columns are 0/1, so `var = mean − mean²` (tabmat's own trick for
  `CategoricalMatrix`).
* `__getitem__(rows, :)` returns `StepMatrix(b[rows])` — a CV fold costs `n_fold`
  int32s, not `n_fold × K` floats.

All sums go through `np.bincount`, i.e. float64 regardless of the declared block dtype.

## Appendix C. Why aggregation by design row is exact

glum minimises `(1/Σw) · Σᵢ wᵢ · dev(yᵢ, μᵢ)/2 + α · penalty(β)` after rescaling `w`
to sum to 1. For every exponential-dispersion family the unit deviance is
`2·(y·(θ(y) − θ(μ)) − b(θ(y)) + b(θ(μ)))`, so the part that depends on `β` is linear in
`y`: `Σᵢ wᵢ·(−yᵢ·θ(μᵢ) + b(θ(μᵢ)))`. Rows with identical design rows share `μ`, hence
group `g` contributes `W_g·(−ȳ_g·θ(μ_g) + b(θ(μ_g)))` with `W_g = Σwᵢ` and
`ȳ_g = Σwᵢyᵢ / W_g`. Objective, gradient and Hessian are identical up to a constant, the
weighted column means/stds used by `scale_predictors=True` are identical, and
`ΣW_g = Σwᵢ` keeps glum's alpha scaling identical. Holds for Poisson, Gamma, Tweedie,
Gaussian, inverse Gaussian and binomial; with an offset only if the offset value is part
of the grouping key. Not identical: the deviance constant (`y·log y` terms) and any
per-row diagnostic — those use `pred[group_of_row]`. CV folds must be assigned to rows
*before* aggregating.

## Appendix D. Files and how to re-run

* `bench.py` — runner (parent spawns one child per candidate; `--sizes`, `--candidates`,
  `--table-only`, `--skip-french`; child mode `--run <candidate> --n <rows>`; the
  float32 dtype probes run as sub-subprocesses so a segfault is recorded, not fatal).
* `spike_lib.py` — data generator, `var_codes`, `predict_from_codes`, dense/Split
  builders, `aggregate`, `StepMatrix`, `patch_glum_validation`.
* `test_lib.py` — equality checks of every representation against `spec.build` and of
  the StepMatrix operations against dense (run it before trusting a change).
* `results.json` — every record (timings, RSS checkpoints, dtypes, diffs, notes,
  tracebacks); `out/` holds cached data/specs and per-candidate coefficient/prediction
  arrays (503 MB; safe to delete).
* `pilot.py`, `debug32.py`, `debug32b.py` — the alpha pilot and the float32 fault
  isolation scripts, kept for reference.

Full default plan: `python bench.py` (≈25 min on this machine, dominated by the
non-converging float32 runs at 5M; `split32_cv` is excluded from the default 1M plan
because it takes 6 min and is known not to converge — pass `--candidates split32_cv` to
repeat it).
