# Review — C2 legacy removal

Reviewer: independent (Claude), 2026-09-02. Scope: commits `d5d09ed`, `170ae2a`,
`1301ca9` (`git diff 1935e25..HEAD`) against the C2 brief in
`docs/RELEASE_0.4_PLAN.md` (§C, sequencing C2, R6 nits) and S6 of
`docs/reviews/00-plan-review.md`. Interpreter `.venv/bin/python`, plus a fresh
`uv venv` for the dependency checks. Nothing outside this file was modified.

## Verdict

**Approved with one blocking fix.** The removal is complete (no trace of the
blueprint / DuckDB path or its scripts remains in code, tests, examples, docs,
packaging or CI), `from_rate_tables` reproduces `to_rate_model` exactly — with an
offset, nulls, unseen levels and integer-coded categoricals — the Gini change is
mathematically right and empirically stable, the golden test is real and cannot be
skipped, the base install is lighter, and the suite, lint and formatter are green.

One hole must be closed before merge: `from_rate_tables` silently accepts a
categorical table that lists the same level twice and rates that level with
whichever row came last. For a feature whose only purpose is hand-built tables that
is a silent mis-rating, and the fix is three lines. Everything else below is
should-fix or cosmetic.

## Blocking

### B1. Duplicate categorical level is accepted silently (last row wins)

- **What.** `RateModel._config_from_table` builds the categorical `cat_map` by
  iterating the rows; a repeated `from` value overwrites the earlier relativity with
  no error or warning.
- **Failure scenario.** An actuary types a `VehGas` table by hand (or pastes it from
  a spreadsheet) with `Regular` on two rows — 0.90 from an old version and 1.05
  from the new one. `from_rate_tables` accepts it, `Regular` is scored at 1.05, the
  table shows both rows, and nothing anywhere says the two disagree. Verified:
  `cat(["A","A","B"], [0.9, 1.5, 1.1])` is accepted and `A` scores 1.5.
- **Exact fix.** In `_config_from_table`, after building `rows` for a categorical:

  ```python
  levels = [r.from_ for r in rows if r.from_ is not None]
  dupes = sorted({lv for lv in levels if levels.count(lv) > 1})
  if dupes:
      raise ValueError(f"Table for {name!r} lists level(s) {dupes} more than once")
  ```

  Also reject more than one `(None, None)` row in either table type (same
  silent-overwrite pattern for the null / Other relativity). Add the two cases to
  `test_from_rate_tables_rejects_bad_tables`.

## Should-fix

### S1. Fixture recipe is not committed, and the docstring's recipe is incomplete

`tests/test_golden.py` says the fixture is "rows sampled with seed 20260902 from
`freMTPL2freq`". That alone does not reproduce it: sampling the cached frame with that
seed gives a different 50k rows. The fixture is exactly
`df.sort("IDpol").sample(n=50_000, seed=20260902).sort("IDpol")` on the cached
parquet (`~/.cache/easy_glm/46131676743eedbc.parquet`, 677,991 rows) with `IDpol`
cast to text — I found this by trial. Per R7 the golden numbers must be
re-derivable by a reviewer, so commit the recipe: a small
`scripts/make_golden_fixture.py` that writes the fixture from
`load_external_dataframe()` and asserts it equals the checked-in file, and state
the two sorts in the docstring. (A pytest that re-derives it may skip when the cache
is absent; that is the one place a skip is acceptable.)

### S2. An integer-coded categorical table is mis-detected as numeric

Type is inferred purely from the dtype of `from`. A hand-built `VehPower` table with
levels 4…12 typed as integers (the natural way to type them, and what an Excel
reader will produce) is treated as a numeric band table and rejected with
"must start with an open lower band and end with an open upper band" — a message
that has nothing to do with the actual problem. This is the C1 integer-categorical
bug re-entering through the hand-built-table door, and S4 (`from_excel`, planned)
will hit it head-on. Fix: when the dtype is numeric but every non-null row has
`from == to`, either treat the table as categorical with `str()` levels (matches
C1's "levels are text everywhere") or raise
"looks like a categorical table with numeric level codes; store the levels as
text". Test both `Int64` and `Float64` level columns.

### S3. Unsorted numeric bands are rejected with a misleading message

The tiling check runs in row order, so a valid table whose rows are shuffled or
descending fails with "must start with an open lower band". Sort the bands by
`from_` (open lower end first) before checking, or change the message to "rows must
be in ascending band order". Sorting is safer for spreadsheet-sourced tables.

### S4. `examples/scoring_editor.py` step 4 does not run (found while reading the examples)

Pre-existing, not introduced by C2, but the task asked whether the examples still
run. The example sets `rm.column_mapping = {"VehAge": "vehicle_age", ...}` (model
variable → data column). `RateModel.predict` and `tests/test_engine.py::TestColumnMapping`
use the opposite direction (`{"driver_age": "DrivAge"}`, data column → model
variable). Verified: with the example's direction `predict` raises
`Column 'VehAge' not found in data`. Flip the dict in the example (or document the
direction in the `column_mapping` docstring and flip it). The other five examples
use only APIs that exist (`use_cv=` is a still-supported deprecated alias).

### S5. `scripts/setup_dev.{py,sh,ps1}` are broken and the README points at them

All three install from `requirements-dev.txt`, which does not exist in the repo.
README §Install (development) says "or: `python scripts/setup_dev.py`". Either
replace the install line with `uv pip install -e ".[dev]"` in all three, or delete
them and the README line (the one-liner above them already does the job). The
`extend-exclude` for `setup_dev.py` in `pyproject.toml` goes with them.
`FIXES_SUMMARY.md` (dask-ml / requirements.txt saga) is likewise stale and can go.

### S6. Package description still advertises the removed feature

`pyproject.toml` `description` reads "… blueprint generation, preprocessing, model
fitting, rate table extraction & plotting". That is the PyPI one-liner. Suggest
"LASSO-regularised GLMs for insurance pricing: exact rate tables, a browser editor
and a portable scoring model. Built on glum."

### S7. Benchmark test does not assert the C2 acceptance criterion

Plan §C: "the benchmark runner produces easy_glm rows on all four families". The
runner does (verified in the fresh venv: 8 non-null easy_glm rows, statsmodels and
catboost rows `FAIL`), but `test_run_benchmarks_with_small_dataset` only checks the
method *names* appear — which they do even on failed rows — and
`test_benchmark_metrics_are_positive` only needs one non-null row. Add: for each of
the four datasets, both easy_glm rows have non-null `Deviance`.

## Nits

- `core/plots.py` still takes a `blueprint` argument and documents "legacy
  `{var: [levels...]}` blueprint-based tables". Drop the parameter and the second
  branch; every caller now passes `rate_tables` output with a `label` column.
- `EasyGLM.blueprint` ("Legacy view of the spec") survives and two tests use it.
  Harmless, but the CHANGELOG says "no blueprint"; either rename to `knots_and_levels`
  or leave with a note.
- `docs/checks/c2-legacy-removal.md` prints the measured tables-vs-GLM difference
  (`1.1e-15`). It is machine noise: my run printed `1.0e-15`, so the document is not
  byte-reproducible. Print only the bound ("below 1e-12: yes").
- Same document: "the old blueprint / DuckDB route that produced the mis-rated
  bottom bands in 0.2". I could not find that finding anywhere in the repo; cite it
  or drop the clause. The document is otherwise plain-language; the function names in
  backticks are unavoidable but the `pip install` line could move to a footnote.
- Install-footprint wording (CHANGELOG, check doc): "scikit-learn dropped" and
  "8 required packages". scikit-learn is no longer *declared* but glum still installs
  it (confirmed in the fresh venv). Say "no longer a direct requirement" so the
  actuary is not surprised to see it in `pip list`.
- CHANGELOG: "the SQL transform helpers" means nothing to a non-programmer; "the
  helpers that rewrote data inside the old database engine" or simply drop it.
- `from_rate_tables` docstring: say what happens when the `(None, None)` row is
  absent — categoricals get an Other row at 1.0 (silently), numerics get no null row
  and nulls raise at scoring. Both are reasonable; both should be written down.
  Consider a warning for the categorical case.
- `test_golden.py` docstring: "a refit on the same machine reproduces coefficients to
  1e-15" is accurate but "bitwise" it is not — see the numbers below; worth saying
  so explicitly so nobody tightens `RTOL` later.
- README §1 "Optional: matplotlib charts" should mention `pip install "easy_glm[viz]"`
  next to the call, since a base install now raises there.
- README quick-start uses `np.random.rand` (unseeded) for the split while every
  example seeds `default_rng(42)`; use the seeded form.
- The realistic Gini worry, for the record: quantising to 1e-12 relative can in
  principle split a tied cell when its score sits within an ulp of a rounding
  boundary. I could not make it happen with realistic (per-cell) noise — 0 changes
  in 300 perturbed runs — and a gap-based alternative was worse (46/300), so no
  change requested; a comment noting the residual case is enough.

## Missing tests

1. Gini tie pooling has no unit test. Add: (a) the pooled value equals the average
   of the unpooled value over all row orders on a 7-row tied example (mine:
   pooled 0.4464285714285714; unpooled min 0.375, max 0.5179, mean 0.4464…); (b) 50
   random row permutations give one distinct value; (c) `gini(a, a)` is exactly
   `1.0`, `gini(a, 3a)` is `1.0`, a constant prediction is `0.0`.
2. `from_rate_tables`: duplicate level (B1), two `(None, None)` rows (B1),
   integer-coded categorical (S2), shuffled/descending bands (S3), and the
   `rate_model_tables(rm)` → `from_rate_tables` round trip (currently only
   `rate_tables(fit)` is covered; both pass today).
3. Golden: `from_rate_tables(rate_tables(fit), base_rate(fit))` on the golden fit
   equals `to_rate_model(fit)` — one line, and it ties the C2 promise to the golden
   data.
4. `tests/test_imports.py` guards only matplotlib. Extend the subprocess check to
   `seaborn`, `duckdb`, `rdata` and `streamlit` so the lazy-import contract is the
   one the CHANGELOG states.
5. Benchmark per-family non-null rows (S7).
6. Fixture provenance (S1).

## What I re-ran, and the numbers

**Suite, lint, format** (`.venv`): `pytest -q` → 233 passed, 16 warnings, 161 s.
`ruff check .` clean; `black --check .` 68 files unchanged. Separately:
`test_app.py + test_invariants.py + test_c1_foundations.py + test_imports.py` →
62 passed; `scripts/checks/c1_foundations.py` runs and prints the C1 tables.

**Removal completeness.** `grep` over src, tests, examples, scripts, docs, pyproject
and `.github` for `blueprint|all_ratetables|prepare_data|PreparedData|EasyGLMModel|
transforms|core.model|core.prepare|core.ratetable|duckdb|new_scoring_prototype|
launch_ui|test_ui_demo|sklearn|seaborn|requirements`: the only hits are the
deliberate "upgrading from 0.2/0.3" notes in README/AGENTS/CHANGELOG, the
`EasyGLM.blueprint` property, `plots.py`'s legacy branch, the `viz` extra, and the
three `setup_dev` scripts (S5). `git ls-files scripts` = the two check scripts plus
`setup_dev.*`. CI installs `.[dev]` only; no DuckDB anywhere.

**`from_rate_tables` vs `to_rate_model`** (4,000-row synthetic book: numeric with 5%
nulls, string categorical with two rare levels lumped to Other, integer-coded
`power` categorical, exposure weight; scored with 5 unseen levels and 3 nulls
injected): max |tables / exact − 1| = **0.0** without and with `offset_col`; vs
`fit.predict` 1.0e-15 / 1.6e-15; `from_glm_model` vs exact 0.0;
`rate_model_tables(rm)` → `from_rate_tables` round trip 0.0; metadata dataclasses
equal (offset_col `logcur`, `offset_is_log` True, link `log`,
`divide_target_by_weight` True); null relativity and Other fallback (1.0163 in the
offset case) carried across identically.

**Adversarial tables.** Gap → "has a gap or overlap at 5.0"; overlap → same
(clear); missing `relativity` → "lacks columns ['relativity']"; null relativity →
clear error; only a `(None, None)` row → "has no bands"; `label`/`is_base` columns
ignored; `Int64` from/to accepted and cast to float; categorical `to` null accepted
(`to = from`); empty dict accepted (no variables); **duplicate level accepted** (B1);
integer-coded categorical → misleading error (S2); shuffled or descending bands →
misleading error (S3); closed first band (`from = 0`) → rejected with a clear
message (acceptable); numeric without null row + null at scoring → "did not match
any bin"; categorical without Other row → Other appended at 1.0, unseen and null
score 1.0 (with the C1 unmatched-levels warning firing).

**Gini.** On a 7-row example with two tie groups: code 0.4464285714285714, brute-force
pooled trapezoid 0.4464285714285714, average of the old unpooled value over all
5,040 row orders 0.44642857142857145 (min 0.375, max 0.5179) — pooling gives the
order-free expectation, as it should. 50 random permutations → one distinct value.
`gini(a, a) = 1.0`, `gini(a, 3a) = 1.0`, constant prediction 0.0 (normalised and
raw). Noise: per-cell ulp perturbation of the rates (the real between-run
mechanism), 15k rows / 5k cells, 300 trials → 0 changes; per-row independent ulp
noise (harsher than reality) → max drift 5.0e-6 vs 1.7e-4 for the old code. No
tolerance or golden elsewhere was touched to accommodate it: `test_workflow.py` is
not in the diff; the only numeric-literal changes in tests are new `rtol=1e-12` /
`1e-10` assertions in the rewritten `test_engine.py` integration tests (tighter
than before, which had none).

**Golden test** (`tests/test_golden.py`, three separate processes): 5 passed each
time (0.85 s). Hex of the computed numbers across the three runs — `holdout_gini`
`0x1.5963e67dfa23ep-2` all three (bitwise identical); `base_rate` …`582b8`, …`582b8`,
…`582b3`; `holdout_ae` …`d8bc`, …`d8ba`, …`d8b8`; `holdout_dev_explained` …`0b30`,
…`0b10`, …`0b30`; integer counts 112 / 61 exact. So the fit is *not* bitwise
reproducible on this machine (BLAS threading), varying by ≤ 5e-16 relative — nine
orders of magnitude inside `RTOL = 1e-6`; the tolerance comment in the file is
justified and the Gini, which was the thing that used to wobble, is now exact.
Missing fixture: `pl.read_parquet` raises `FileNotFoundError` in the module fixture,
so all five tests **error** (not skip). Fixture is tracked (`git ls-files`), 373,622
bytes, `IDpol` sorted, 50,000 rows, and reproduces exactly from the cache as
described in S1.

**Benchmark** (fresh venv, statsmodels and catboost absent, `n_rows=300`): easy_glm
(no CV) and (CV) rows non-null for poisson, gamma, gaussian, binomial (deviances
193.2 / 8.34, 8.31 / 4.80, 15.98 / 10.70, 0.607 / 0.386; `NParams` 651); statsmodels
and catboost rows `FAIL` as designed. Identity link for gaussian is set in
`_fit_easy_glm` with the comment "gaussian benchmark responses can be negative", and
mirrored in `test_gaussian_family_fits`.

**Dependencies** (fresh `uv venv` under the scratchpad, `uv pip install -e ".[dev]"`,
resolved to Python 3.14): `import easy_glm` succeeds; after import `sys.modules`
contains no matplotlib, seaborn, duckdb, rdata, streamlit, statsmodels or catboost —
it does contain scikit-learn, which glum imports (hence the wording nit).
`find_spec`: matplotlib, seaborn, duckdb, statsmodels, catboost absent; sklearn,
rdata present. `plot_all_ratetables(...)` raises
`ImportError: plot_all_ratetables needs matplotlib and seaborn: pip install "easy_glm[viz]"`;
after `uv pip install ".[viz]"` the same call renders (Agg backend).

**Actuarial check.** `scripts/checks/c2_legacy_removal.py` exits 0 (no `GOLDEN
MISMATCH`); its output is identical to `docs/checks/c2-legacy-removal.md` except
the measured noise figure (`1.0e-15` vs `1.1e-15`) and a trailing blank line. The
"Install footprint" section is derived from `git show v0.3.0:pyproject.toml`, so it
is honest about declared dependencies (11 → 8).

**Docs.** CHANGELOG is accurate against the diff and mostly plain-language (nits
above). README and AGENTS module map list `core/split.py`, mark `plots.py` as
viz-extra, describe `from_rate_tables` correctly, and the AGENTS tests table matches
the files present (`test_blueprint.py`, `test_nulls.py`,
`test_model_and_ratetable.py` are gone from both disk and the table). R6 nits are
done: investigation scripts deleted, plotting deps moved, `Snapshot.metrics`
populated by `run_model` and `rebuild_rate_model` and round-tripped through JSON
(covered by `test_snapshot_metrics_are_stored_and_round_trip`).

## Re-check needed

B1 (with its tests) and S1. The rest can ride along or land in a follow-up; none of
it changes a number.

## Re-check (commit `4543e9f`, 2026-09-02)

Scope: `git diff 1301ca9..4543e9f` (`0283a3b` in between is the coordinator's
ruff/black exclusion of `docs/spikes`). Same interpreter; nothing outside this
file modified.

**Verdict: Approved.** B1 is closed, S1–S7 are done, the wording nits are
addressed, and every missing test I listed now exists and passes. No number changed.

### Item by item

| Item | Status | Evidence |
|---|---|---|
| B1 duplicate level / duplicate null row | Fixed | `cat(["A","A","B"])` → `Table for 'r' lists level(s) ['A'] more than once`; two `(None, None)` rows → `has 2 rows with both 'from' and 'to' empty; only one null / Other row is allowed` — both for categorical and numeric tables. Covered by `test_from_rate_tables_rejects_duplicates`. |
| S1 fixture recipe | Fixed | `tests/fixtures/make_french_motor_50k.py --check` → `fixture matches recipe: True`, exit 0 (select → sort IDpol → sample(50k, seed 20260902) → sort IDpol → cast text). `test_fixture_matches_its_recipe` re-derives it when the cache exists and skips otherwise — the one acceptable skip. Docstring states the recipe. |
| S2 integer-coded categorical | Fixed | `Int64` and `Float64` tables with `from == to` on every row become `categorical` with text levels `'4','5','6'`; scoring `[4, 6, 9, None]` → `[0.9, 1.1, 1.0, 1.0]`; fractional codes `0.5/1.5` keep their text form. A numeric-typed table with any `from != to` row still takes the band path, so `rate_tables` output is untouched (equivalence below is still 0.0). Parametrised test added. |
| S3 shuffled / descending bands | Fixed | Shuffled and descending inputs are re-ordered to `(None,0),[0,5),[5,10),[10,None)` and score identically to the ordered table; a null row placed first is kept and its relativity (1.3) applied to nulls; negative edges and an edge at exactly 0.0 sort correctly (I probed the `from_ or 0.0` sort key for that). Test added. |
| S4 `examples/scoring_editor.py` | Fixed | Mapping flipped to dataset-column → model-variable, with a comment; `RateModel.__init__` now documents the direction. |
| S5 `setup_dev.*`, `FIXES_SUMMARY.md` | Fixed | All four files deleted; README and AGENTS install lines updated; the ruff exclusion for `setup_dev.py` removed. No stale references remain (`grep` for `setup_dev|FIXES_SUMMARY|requirements` over `*.md`/`*.toml` hits only the CHANGELOG line that records the deletion). |
| S6 package description | Fixed | New one-liner without "blueprint". |
| S7 benchmark acceptance | Fixed | Test now asserts eight easy_glm rows, one per family and CV mode, with no null `Deviance`. |
| Nits | Done | `plots.py` `blueprint` parameter and legacy branch removed (raises a clear error for tables without `label`/`relativity`); check doc prints only "below 1e-12: yes" and is now byte-identical to the script output apart from a trailing newline; "mis-rated bottom bands" clause dropped; scikit-learn wording corrected in CHANGELOG and check doc; "SQL transform helpers" jargon removed; README `[viz]` hint and seeded split; golden docstring now says the fit is not bitwise reproducible and `RTOL` must not be tightened; missing-Other-row behaviour documented and warned (`Table for 'r' has no Other row …; unseen levels and nulls will score at 1.0`). |
| Missing tests 1–6 | Done | `TestGiniTies` (pooled = mean over all 5,040 orders, order independence, perfect/scaled/constant); `from_rate_tables` rejection and coded-categorical tests; `rate_model_tables` round trip in `test_from_rate_tables_matches_to_rate_model`; `test_golden_hand_built_tables_match_exact_tables` (rtol 1e-12 on the golden holdout); `test_imports.py` parametrised over matplotlib, seaborn, duckdb, rdata, streamlit, plotly; benchmark per-family rows; fixture recipe test. |

### Error messages, as a hand-table author would see them

Gap: `band ending at 5.0 is followed by a band starting at 6.0; bands must tile the
line with no gaps or overlaps`. Overlap: `… ending at 5.0 is followed by a band
starting at 4.0 …`. Two open-lower or two open-upper bands: caught by the same
check (`… followed by a band starting at None …`) — correct, if slightly terse.
Closed first band: `must start with an open lower band (a row whose 'from' is empty
covers everything below the first edge)`. No open upper band: the mirror message.
Categorical with `from != to`: `must have 'from' == 'to' on every level row`.
All clear enough for a non-programmer reading a spreadsheet.

### Re-run numbers

- Full suite: **248 passed**, 17 warnings, 165 s. `ruff check .` clean; `black --check .`
  68 files unchanged.
- `from_rate_tables` vs `to_rate_model` on the 4,000-row synthetic book (nulls, unseen
  levels, exposure weight), without and with `offset_col`: max |ratio − 1| = **0.0**;
  `rate_model_tables` round trip 0.0; vs `fit.predict` 1.3e-15 / 1.1e-15.
- `scripts/checks/c2_legacy_removal.py`: exit 0, golden match on all six quantities;
  output identical to `docs/checks/c2-legacy-removal.md` (trailing newline aside).
- `tests/fixtures/make_french_motor_50k.py --check`: True.

### Remaining (non-blocking, for the record)

- Scoring an integer-coded categorical with a *float* column (`4.0` against level
  `"4"`) still falls to Other — that is the C1 text-comparison contract and the C1
  warning fires loudly (`100% of rows matched none of its 3 trained levels`). The new
  `_level` helper normalises the table side (`4.0 → "4"`) but not the data side;
  if a future piece normalises integer-valued floats at scoring, do it in
  `score_categorical` so both sides agree.
- The tiling error for two open-lower bands says "starting at None"; a dedicated
  message ("more than one row has an empty 'from'") would read better. Cosmetic.
