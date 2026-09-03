# Review of piece W2 — workbench pages for interactions and piecewise-linear terms, persona e2e runs

*Reviewer: independent. Branch `release-0.4`, commits `6ac7c66`, `acbcf4b`, `15cfb26`
(`git diff dbb722a..HEAD`). Contract: `docs/RELEASE_0.4_PLAN.md` §A/§B workbench bullets,
§D item 7, §"GUI quality" (persona runs, break-it catalogue), §Revisions R2/R3.
Date 2026-09-03.*

## 1. Verdict

**Changes requested — three blocking items; all are small, none touches the engine.**

What the pages show is right. On the French-motor fixture (50,000 policies, 70/30 split,
`DrivAge × VehPower` at 0.5 % cell exposure, `Density` piecewise-linear) I checked, outside
Streamlit and through AppTest, that: the Design-page cell preview keeps exactly the cells the
fitted encoder keeps (86 of 168, exposures equal to 1e-9); the Rate-tables heatmap is the
`RateModel`'s cell matrix (21 × 8, equal); a cell edit lands as `Adjustment(cell=True)` and
`run.predict` moves by the factor (1.37, to 1e-15) on the 169 holdout rows in that cell and by
nothing anywhere else; a node edit on `Density` moves that node, re-derives the slopes of the
two adjacent bands and no others, and the engine rule `slope = Δlog(rel)/Δx` holds on every
band afterwards; the chart's curve equals `score_linear` to 4e-16 and `RateModel.predict`
along a `Density` sweep to 1.5e-5; the A/E-by-pair matrices sum to the overall actual
(607.0), expected (600.04) and exposure (7,950.15) on three pairs. The pair-search statistic
is sound: on a null book (mains only, three seeds, six pairs each) |z| ≤ 1.5 in 17 of 18 cases
and 2.9 once; the planted `DrivAge × Region` block ranks first with z = 4.3–4.5 and the
runner-up ≤ 1.2; the IPF converges to 1e-15 in its five sweeps; the degrees of freedom are
counted on the kept cells after the expected < 3 filter, which is correct; zero-exposure
bands, one-level categoricals, constant, all-null, boolean and 30-row inputs all return
without a crash. Every break-it action below ended in a message or a graceful fallback; the
three server logs contain no traceback. Suite 398 passed + 1 skipped, ruff and black clean,
golden untouched, the check script's text mode reproduces `docs/checks/w2-pages.md`, the
five screenshots exist at 87–110 KB.

The blocking items: (B1) the documented persona command fails — with the relative
`EASY_GLM_SERVER_PYTHON=.venv/bin/python` that `AGENTS.md` prescribes, the actuary run dies in
`run_python` with `FileNotFoundError` before the "exported script reproduces the scorer"
assertion; the builder's "2 passed" only holds with an absolute path. (B2) `pytest tests/e2e`
from the repo venv (no Playwright) exits 1 with a traceback instead of skipping. (B3) the
three relativity editors display **truncated numbers**: with `step=0.01`, Streamlit's
`NumberColumn` truncates the cell to two decimals despite `format="%.4f"`, so the linear
editor shows *working 1.0000* on every row next to *fitted 1.0005 … 1.0040* — the screenshot
in the actuary's own check document shows this — and the interaction editor shows 0.830 for a
fitted 0.8306. It also limits what can be typed to two decimals. That is misleading output in
the one control the actuary is meant to trust.

## 2. Blocking

### B1. The documented persona command does not run the actuary assertions

**What.** `tests/e2e/conftest.py:33` takes `EASY_GLM_SERVER_PYTHON` verbatim. `Server` runs
it with `cwd=ROOT`, so a relative path works there; `_helpers.run_python` runs it with
`cwd=e2e_dir` (a temp folder), so the same relative path fails.

**Failure scenario.** Exactly the command in `AGENTS.md` and in the task brief:

```
EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <playwright-python> -m pytest tests/e2e -q
→ FAILED tests/e2e/test_persona_actuary.py::test_actuary_rate_review
  FileNotFoundError: [Errno 2] No such file or directory: '.venv/bin/python'
1 failed, 1 passed in 61.35s
```

The failure happens at the last step, after every browser assertion has passed, so the one
assertion the plan names ("exported script reproduces predictions") is the one that never
runs. With `EASY_GLM_SERVER_PYTHON=$PWD/.venv/bin/python` both tests pass (61.5 s).

**Exact fix.** In `tests/e2e/conftest.py`:

```python
SERVER_PYTHON = str(
    (ROOT / os.environ.get("EASY_GLM_SERVER_PYTHON", sys.executable)).resolve()
    if not os.path.isabs(os.environ.get("EASY_GLM_SERVER_PYTHON", sys.executable))
    else Path(os.environ["EASY_GLM_SERVER_PYTHON"]).resolve()
)
```

(or simply `cwd=ROOT` in `run_python` and pass the temp folder to the script explicitly). Add
a one-line test that `Path(SERVER_PYTHON).is_absolute()`.

### B2. Without Playwright, `pytest tests/e2e` crashes instead of skipping

**What.** `conftest.py:24–28` calls `pytest.importorskip` and `pytest.skip(allow_module_level=True)`
at conftest import time. When the path `tests/e2e` is given on the command line, pytest
imports that conftest while parsing arguments, and a `Skipped` raised there is not handled:

```
.venv/bin/python -m pytest tests/e2e -q
→ Traceback (most recent call last): … _pytest/outcomes.py … raise skipped
  Skipped: could not import 'playwright.sync_api': No module named 'playwright'
exit code 1
```

`pytest -q` from the root is fine (the conftest is loaded during collection and the
directory is skipped — that is the "1 skipped" in the suite), so CI is not broken today, but
the command a developer types after reading `AGENTS.md` is.

**Exact fix.** Replace the two module-level skips with collection control:

```python
_ENABLED = bool(os.environ.get("EASY_GLM_E2E")) and importlib.util.find_spec("playwright") is not None
if not _ENABLED:
    collect_ignore_glob = ["test_*.py"]
```

and import `playwright.sync_api` inside the `browser` fixture. Add
`tests/test_w2_pages.py::test_e2e_skips_cleanly_without_playwright` that runs
`subprocess.run([sys.executable, "-m", "pytest", "tests/e2e", "-q"], env=without EASY_GLM_E2E)`
and asserts return code 0 / 5 and no `Traceback` in the output.

### B3. The editors show truncated relativities (and only accept two decimals)

**What.** `pages_tables.py` gives the editable column `step=0.01` in all three editors
(step/categorical `working`, linear `working (at band start)`, and every column of the
interaction cell grid, with `format="%.3f"`). Streamlit's docs say a printf format keeps its
own precision, but the frontend truncates the value to the step's decimals first. I isolated
it: a three-column data editor with the same number 1.0005 shows **1.0005** (no step),
**1.0000** (`step=0.01`, `format="%.4f"`); 0.8306 shows 0.8300; 1.2345 shows 1.2300.

**Failure scenario.** Open Rate tables → `Density (linear)` on the check project with no
adjustments. The grid reads *fitted 1.0005 / working 1.0000*, *1.0008 / 1.0000*, … down the
column (see `docs/checks/img/w2_tables_linear.png`); "manual adjustments 0" in the header.
The actuary concludes the working table has been flattened, or that the fitted numbers are
not what is being applied. I chased this for an hour myself before proving (AppTest spy on
`st.data_editor`) that both columns carry identical numbers. Likewise the interaction grid
shows 0.830 where the heatmap hover says 0.8306, and typing 1.125 to cap a cell is stored as
1.13 (the step also rounds the entry).

**Exact fix.** `step=0.0001` (or drop `step`) on the three `NumberColumn`s at
`pages_tables.py` lines ~113, ~138 and ~229 (`format="%.4f"` on the cell grid too, to match
the hover). The 0.3 step editor had the same `step=0.01`; fixing it here is one more line.
Regenerate the five screenshots afterwards (`scripts/checks/w2_pages.py --write`) — the
current linear-table screenshot is the misleading one.

## 3. Should-fix

**S1. The pair-search z-score is Poisson-only and is shown for every family.** `signal =
(Σ(A−E')²/E' − d)/√(2d)` assumes `Var(A) = E'`, i.e. claim counts. With actual = amounts
(a severity or Tweedie model, both offered on the Model page) the same code returns
z ≈ 11,000 on a null book (I fed it the null actual/expected × 2000). Either divide the
Pearson sum by a dispersion estimate (the fit's Pearson φ̂ = Σ(y−μ)²/V(μ)/(n−p) is the
standard one; or, model-free, the exposure-weighted median of the per-pair `X²/d` which is
≈ 1 under the null) or show the search only for `family == "poisson"` with
`divide_target_by_weight`, and say so in the caption. Add a test with amounts.

**S2. The heatmap under "Search pairs" is not the table that was scored.** The search bands
numerics into 8 quantile bands (`knots` is not passed, deliberately, at
`pages_diagnostics.py:381`), but `_pair_heatmap` below it *does* pass the model's knots, so
for `DrivAge` the search scores 8 × 8 cells and the picture shows 21 × 8, and `worst_cell`
(`≥ 63.0 | [64.0, 80.0)` in my session) names bands the heatmap does not have. Pass the
same banding to both (call `_pair_heatmap(..., knots={}, levels=levels, n_bins=8)` for the
search's picture, or let the search take the model's knots merged to ≤ 8 bands).

**S3. "No data" and "no adjustment" cells look the same.** The interaction table carries all
168 cells (82 of them without their own column, 28 with zero training exposure); the heatmap
paints every 1.00 white and the grid lets the user type an adjustment into a cell no policy
ever fell in. R3 asks that "1.00 reads as *no data* vs *no adjustment*". The hover does say
so, but the colour does not: paint cells without a column blank (pass a mask to
`matrix_heatmap`, or `None` in `cell_grid["current"]` for keys whose exposure is 0) and mark
below-threshold cells in the grid caption ("82 cells had too little exposure and are 1.00 by
construction"). The Excel matrix sheet has the same question; check it shows exposure.

**S4. Persona runs assert less than the plan lists.**
- Actuary: the offset step is skipped (the conftest builds `current_premium` and supports
  `offset=True`, but the actuary project is written with `offset=False`); "cap one
  relativity" is skipped; after the reload only "Fitted and up to date" is asserted, not
  that adjustments survived. Typing into the glide grid *is* possible from Playwright (my
  break-it session did it: click the cell, `Enter`, type, `Enter`), so the cap step can be
  scripted; failing that, seed one `cell=True` adjustment in the project file and assert
  "manual adjustments 1" and the cell value after reload.
- Data scientist: `assert "DrivAge × VehPower" in text or "×" in text` is always true (the
  caption contains "×"). Assert on the *Show* selectbox's default (the top pair) instead, or
  plant a cell in the fixture so the top pair is known. "Metrics agree with
  `workflow.model_metrics`" is not asserted; the metrics table is a canvas, so run
  `model_metrics` in `run_python` on the reloaded scorer and compare the Gini shown in the
  sidebar/caption text.
- `settle()` carries 0.8 s of fixed sleeps per call (≈ 25 s across the two runs); acceptable,
  but a `wait_for` on `[data-testid="stStatusWidget"]` detaching would be both faster and
  stricter.

**S5. Design preview and search do not say which rows they use.** The interaction preview
runs on all training rows (correct — it matches the fit) but the caption only says
"Preview on 34,868 training rows"; when a sample is active the page's top caption explains
that knots come from full data, the interaction section should repeat it for the cells.

**S6. Export page with a vanished data file (new session) says "This model is not fitted"**
rather than "Could not load …" like the other pages; `require_data` is not called on that
page before the fitted check. Cosmetic, but the actuary would look in the wrong place.

## 4. Nits

- Colour bar is in log units ("log ratio 0.6 / 0.2 / −0.2"); actuaries read ratios. Set
  `tickvals=log([0.5, 0.8, 1, 1.25, 2])`, `ticktext=["0.50", "0.80", "1.00", "1.25", "2.00"]`.
- Linear curve chart uses a linear x-axis 0–27,000 for `Density`, so the 20 bands below 1,000
  where all the exposure is collapse into the first pixel; offer a log axis when `hi/lo > 100`.
- Typing `-2` into a linear node becomes `2` (Streamlit drops the sign under `min_value`) and is
  saved as 2.0 — the message "a negative relativity is not meaningful" in `apply_row_edits`
  can never fire from the grid. Harmless, but the unit test for it tests a path the UI cannot
  reach.
- After a refused cell value the same error is re-shown on every rerun until the user retypes
  the cell (the editor keeps the 0). One sentence in the message ("retype the cell to clear
  this") would help.
- `pair_bins` slider label "Bands (not in model)" is truncated in a 400 px viewport to
  "Bands (not in m…" — fine, but "Bands" alone would do.
- `residual_pair_search` with NaN in `expected` (a broken prediction) drops the NaN cells and
  reports z ≈ 11 on a null book; guard with `np.isfinite` and say "prediction has NaN".
- The `A/E rows` radio is keyed `tables_ae_rows` in both `_main_effect` and `_interaction`;
  fine today (only one renders) but fragile.
- `CHANGELOG` says the persona runs take "about 30 s each": true (30 s + 30 s), but only with
  the absolute path (B1).

## 5. Missing tests

1. The B1/B2 regressions: `SERVER_PYTHON` absolute; `pytest tests/e2e` exits 0/5 without
   Playwright.
2. Design preview == fitted encoder: `InteractionEncoder.from_data(...)` built the way the
   page builds it equals `run.spec[name]` on `cells` and `exposure` (I did it ad hoc: equal).
3. Cell edit → `run.predict` ratio equals the factor on rows in the cell and 1.0 elsewhere
   (the existing test checks `cell_grid["current"]`, not predictions).
4. Chart polyline == `score_linear` on the same x (equal to 4e-16 here) and the fitted vs
   working overlay after a node edit.
5. Null distribution of the pair search: three seeds, mains only, `max |z| < 3.5` and the
   planted pair first with z > 3 (the existing test covers the planted case only).
6. Amounts / non-Poisson input to the search (S1) once the decision is made.
7. `pair_matrices` totals equal `totals(...)` including null rows (I checked with nulls
   injected into `VehPower`: actual 607 = 607, the `Other / Unknown` column present).
8. A persona-level assertion that adjustments survive the reload (S4).

## 6. Break-it log

Live app (`streamlit run … --project=<check project>`, port 8611), Playwright chromium
1400 × 900 unless stated. Severity: crash / data loss / misleading / cosmetic / none.

| Action | Outcome | Severity |
|---|---|---|
| Empty project (no `--project`): all nine pages | "Load a data file on the Project & data page first." on seven pages; Project page renders; Export says "Create and fit a model first." | none |
| Data file deleted after load, same session: Split, Model, Rate tables, then **Fit** | All render from the cached frame; the fit succeeds ("Fitted and up to date") | none (expected: frames are cached) |
| Data file deleted, new browser session: all pages | "Could not load …/gone.parquet: … Check the path on the Project & data page." on seven pages; Export shows the "not fitted" message instead (S6) | cosmetic |
| Remove a predictor used by an interaction, then Design / Model / Tables / Diagnostics | Not reachable from the browser (roles live in a canvas grid); AppTest `test_pages_survive_a_removed_predictor` covers it: no exception; Model page warns "parents no longer among the predictors", Tables/Diagnostics say which columns are missing | none |
| Interaction with the same variable twice | "Pick two different variables."; Add button disabled | none |
| Duplicate pair (`DrivAge × VehPower` again) | "DrivAge × VehPower is already in the model."; Add disabled | none |
| Clamp lo 5000 > hi 100, Apply | "Clamp lo must be below clamp hi" plus the list of knots now outside the (inverted) range; design unchanged | none |
| Custom knots 5 and 9000 outside clamp [10, 3000] | "Knots outside the clamp range: 5, 9000 (clamp 10 – 3000); move them inside or widen the clamp" | none |
| Custom knots `100, abc` | "Knots: 'abc' is not a number" | none |
| Custom knots `100, 100, 1000` | Accepted, de-duplicated to `100, 1000` | none |
| Interaction cell → 0 | "[25.0, 28.0) \| [5.0, 6.0): an adjustment must be above 0 (was 0); change not saved" | none |
| Interaction cell → −1, → `abc`, → `1e12` | Grid refuses the entry (min 0 / not a number / too many digits); the earlier "was 0" message persists until the cell is retyped | cosmetic |
| Interaction cell → 2 | Saved as a cell adjustment; "manual adjustments 1"; heatmap and A/E update | none |
| Reset all (interaction) | Adjustments cleared, button disabled afterwards | none |
| Linear node → 0 | Grid refuses (min 1e-4); nothing saved | none |
| Linear node → −2 | Stored as **2.0** (sign dropped by the grid); curve shows the spike; adjustment listed | cosmetic (nit) |
| Linear node → `abc` | Refused by the grid | none |
| Linear node → 1.5, then Reset all | Saved, slopes re-derived on both sides; reset clears | none |
| Linear editor display (no edits) | *working* column shows 1.0000 on every row against *fitted* 1.0005–1.0040 (B3) | **misleading** |
| A/E by pair: same variable in Rows and Columns | "Pick two different variables." | none |
| Search pairs (6 predictors, 15 pairs) | Table with z-scores, `Show` heatmap; but heatmap banding ≠ scored banding (S2) | misleading (minor) |
| Refresh the page 1.5 s into a fit | After reload: "Fitted and up to date" (the fit finished server-side and was persisted); no exception | none |
| Browser back (Rate tables → Design) and forward | Both pages render, correct titles, no exception | none |
| 400 px viewport: Design, Diagnostics, Rate tables | Render; sidebar collapses; `scrollWidth > clientWidth` is **false** (no body overflow); heatmaps and grids scroll inside their containers | none |
| Server logs (three servers) | 0 tracebacks | — |

## 7. What I re-ran

- `pytest -q` (repo venv): **398 passed, 1 skipped**, 174 s. `ruff check .`: clean. `black --check .`: 84 files unchanged. `git status` clean; no golden files in the diff.
- `EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <pw-python> -m pytest tests/e2e -q`: **1 failed, 1 passed** in 61.4 s (B1). With `$PWD/.venv/bin/python`: **2 passed** in 61.5 s (actuary 30 s, data scientist 30 s); no `Traceback` in either server log.
- `.venv/bin/python -m pytest tests/e2e -q` (no Playwright): exit 1 with a traceback (B2).
- `scripts/checks/w2_pages.py` (text mode) vs `docs/checks/w2-pages.md`: identical apart from a trailing newline. Screenshots: 5 files, 87–110 KB each.
- Page correctness script (French-motor fixture, `Interaction("DrivAge","VehPower",0.005)`, `Density` linear, α = 5e-4): preview cells = fitted cells (86/168, exposures equal); heatmap = `cell_matrix` (21 × 8); cell edit ×1.37 → predict ratio 1.37 on 169 rows, 1.0 elsewhere (max deviation 0.0); node edit ×1.3 on `[33, 50)` → slopes of `[20,33)` and `[33,50)` change, all others unchanged, rule holds on every band; chart vs `score_linear` 4.4e-16 (working) / 3.3e-16 (fitted); `RateModel.predict` sweep vs curve 1.5e-5; A/E-by-pair totals 607.0 / 600.044 / 7,950.15 on three pairs and with nulls injected.
- Pair search: null (n = 20,000, mains only), seeds 1–3, six pairs: z ∈ {1.46, 1.07, 0.59, 0.41, −0.54, −1.39}, {2.92, 1.01, 0.55, 0.49, −0.21, −0.83}, {1.11, 1.07, 1.05, 0.75, −0.04, −0.75}. Planted (+0.6 log on age < 35 × R2): top `DrivAge × Region` z = 4.26 / 4.26 / 4.52, runner-up 0.67 / 0.03 / 1.23. IPF margin misfit after 5 sweeps ≤ 1e-15; no post-IPF cell below 3. Pathological inputs (zero-exposure band, one-level categorical with and without `levels`, constant numeric, all-null numeric and string, boolean, no weight, 30 rows, all-zero expected): no crash. Amount-scale input: z = 11,081 (S1); NaN in expected: z = 11.1 (nit).
- Streamlit isolation of B3: `NumberColumn(format="%.4f", step=0.01)` renders 1.0005 → 1.0000, 0.8306 → 0.8300, 1.2345 → 1.2300; without `step` the same numbers render exactly.

## 8. Re-check of commit `c5da2ee` (2026-09-03)

Scope: `git diff 15cfb26..c5da2ee` only (23 files; app pages, `grids.py`, `charts.py`,
`workflow/diagnostics.py`, `ui.flash`, `tests/e2e/*`, four regenerated screenshots).

**Final verdict: accepted.** All three blocking items are fixed and verified on the live app;
the deviation on B3 (no `step` at all instead of `step=0.0001`) is the right call.

What I re-ran:

- Documented persona command, relative interpreter exactly as in `AGENTS.md`
  (`EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <pw-python> -m pytest tests/e2e -q`):
  **2 passed in 69.7 s** (actuary 31 s, data scientist 35 s). `_server_python()` joins a
  relative path onto the repo root without `resolve()` — correct, since resolving a venv
  symlink would lose the venv's packages.
- `.venv/bin/python -m pytest tests/e2e -q` without Playwright: "no tests ran", **exit 5, no
  traceback** (`collect_ignore_glob` in the conftest; the import of `playwright.sync_api`
  moved into the `browser` fixture).
- Full suite on the repo venv: **403 passed**, 175 s (the e2e folder is now ignored rather
  than skipped, hence no "1 skipped"). `ruff check .` clean, `black --check .` 84 files
  unchanged. `git status` clean apart from this review.
- Editors at 4 dp, live app on the check project (screenshots
  `scratchpad/recheck/recheck_linear.png`, `recheck_interaction.png`, and the regenerated
  `docs/checks/img/w2_tables_*.png`): the linear grid now reads *fitted 1.0005 / working
  1.0005*, *1.0008 / 1.0008*, … *1.0233 / 1.0233* — identical columns, as the numbers are; the
  interaction grid shows 0.8306, 0.8810, 0.9834, 1.0140, 0.8029, 0.6452 where it showed 0.830,
  0.880, 0.980, 1.010, 0.800, 0.640. On the B3 deviation: I had verified in isolation that
  `step=0.01` truncates; the builder reports every `step` value truncates the grid display
  under this Streamlit, so `format="%.4f"` with no `step` is the only correct option — and it
  also removes the two-decimal limit on what can be typed. Accepted.
- Should-fix items, by reading the diff: S1 — `pearson_dispersion` on totals, applied to
  the Pearson sum and to the `min_expected` filter; φ̂ = 1 for Poisson, caption says "read it
  as a ranking" otherwise; `test_pair_search_dispersion_scaling` shows the ×2000 case
  reproduces the count-scale z to 1e-9. Sound. S2 — the shown pair now uses the search's
  8-band grid (`knots={}`, `n_bins=8`) and says "search bands" in the title. S3 —
  zero-exposure cells are `None` in `cell_grid`, blank in the heatmap (the regenerated
  screenshot shows the `Other / Unknown` row and `≥ 11.0` column blank) and refused in
  the editor with "no policy ever fell in this cell". S4 — the actuary run now uses the
  `log_current_premium` offset, caps `VehGas` row 2 to 1.05 through the grid overlay editor
  (`edit_grid_cell`, verified against "1 adjustment"), asserts the adjustment and the fit
  survive the reload and that the downloaded scorer carries the offset column and two
  snapshots; the data-scientist run refits the stale v1, asserts the *Show* selectbox holds a
  real pair, and recomputes the holdout Gini from the downloaded scorer to within 5e-4 of the
  one on screen. `settle()` now waits for the status widget to attach/detach (0.15 s fixed
  wait instead of 0.8 s). S5, S6, the ratio colour-bar ticks, the "retype the cell" hint and
  the log-x option are in. `ui.flash()` is a sensible fix for notices lost to `st.rerun()`
  on Streamlit ≥ 1.63 and has a one-shot test.

One new item, carried forward as **should-fix (not blocking)**:

- **The "below the exposure threshold" count in the interaction caption is wrong.**
  `cell_grid["n_below_threshold"]` counts every cell with exposure whose fitted and current
  value are exactly 1.00. On the check project the caption says "a further **116** cells were
  below the exposure threshold"; the encoder kept 86 of 168 cells, 28 have no exposure, so
  **54** were below the threshold and the other 62 are kept cells whose lasso coefficient is
  zero. Fix: compute it from the encoder (`len(run.spec[var].cells)` is available on the
  page: `n_all − kept − n_nodata`) and say the rest are "kept cells with no adjustment fitted",
  which is the more interesting number for the actuary. One line in `pages_tables._interaction`
  plus the caption; drop the heuristic from `cell_grid` or rename it.

Nits noticed on the re-check: `log_x` is gated on `enc.lo > 0`, so `Density` (clamp lo = 0)
still gets the linear axis where all the exposure sits below 1,000 — use the smallest
positive band edge as the axis floor instead; `_drop_refused_adjustment` both draws the
error and queues a flash, so when the caller does *not* rerun (a fit) the message appears
twice, once now and once on the next run.
