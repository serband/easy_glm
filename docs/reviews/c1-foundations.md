# Review — piece C1 (foundations)

Reviewed: commits `8a758e0`, `8bab954`, `05207c1` on `release-0.4` (diff `41e0dbf..HEAD`).
Reviewer had not seen the builder's reasoning; everything below was re-run independently
with `.venv/bin/python` (Python 3.14) on 2026-09-02.

## 1. Verdict

**Approve with changes.** Every R1/R5 and §7 item in scope is implemented as specified and the
exactness claim holds on every adversarial case I could construct (genuine 0.3 files included);
the changes requested below are about the editor's A/E default on one API path, the check script
overwriting its own golden document, and missing tests — none makes a shipped number wrong on the
documented paths.

## 2. Blocking findings

None.

## 3. Should-fix

### S1. Editor A/E default is wrong when a count model's RateModel has no `exposure_col`
- **What.** `src/easy_glm/ui/app.py` (lines 91–106) derives `sum_over_weight` from
  `divide_target_by_weight and weight_col` alone. `compute_actual_expected`
  (`src/easy_glm/ui/metrics.py`) applies the *same* formula to the target and to `rm.predict(data)`,
  so the derivation is only right when predictions already include exposure, i.e. when
  `metadata.exposure_col` is set.
- **Failure scenario (measured).** Poisson count model, `divide_target_by_weight=True`,
  `rm = to_rate_model(fit)` (no `exposure_col`), `rm.launch_editor(data)`: overall train A/E shown
  as **0.599** (= mean exposure). With `exposure_col="Exposure"` the same model shows 0.99997.
  `EasyGLM.fit` and the workbench (`exposure_for`) always set `exposure_col`, so the documented
  paths are fine; the bare `to_rate_model` → editor path is not, and it was equally wrong in 0.3.
- **Fix.** In `app.py` derive `sum_over_weight` only when
  `meta.divide_target_by_weight and meta.weight_col and meta.exposure_col`; when
  `divide_target_by_weight` is true and `exposure_col` is `None`, either have
  `compute_actual_expected` score with `rm.predict(data, exposure_col=weight_col)` or show a
  sidebar warning that A/E is per unit of weight. Add the D7 test (see §5).

### S2. `scripts/checks/c1_foundations.py` overwrites the committed golden document, and its first golden number is not reproducible
- **What.** The script writes `docs/checks/c1-foundations.md` unconditionally. Re-running it (as
  R7 requires the reviewer to do) produced a diff on a tracked file: `max_rel` came out
  **8.9e-16** where the committed document says **1.2e-15**; all other numbers matched.
  The last digit of a 1e-15 residual is BLAS/threading noise, not a golden number.
- **Failure scenario.** Every reviewer re-run dirties the working tree; a reviewer who does not
  notice commits a "changed golden number" without a written reason, which R7 makes a blocking item.
- **Fix.** Print to stdout by default and write the document only behind `--write`; report the
  exactness residuals as a bound (`< 1e-12`, with the raw value in the stdout log) rather than as
  two significant figures. Keep the rest as is.

### S3. `Project.from_dict` warns on unknown keys only inside dataclass sub-objects
- **What.** `_build` (`src/easy_glm/workflow/project.py`) warns for unknown keys in
  `source`, `recodes`, `derived`, `split`, `defaults`, `variables`, `penalty`, `models[*]`. Unknown
  keys at the top level, inside `data`, inside `design`, or inside an adjustment entry are dropped
  silently (verified: `{"future_top": 1}` and `data["future_data_key"]` produce no warning).
- **Failure scenario.** A 0.4.x project with a new top-level section (e.g. `runs`) opened by 0.4.0
  loses the section on the next save with no message.
- **Fix.** Apply the same "unknown keys → warning" check to `raw`, `raw["data"]`, `raw["design"]`
  and the adjustment dicts (a five-line helper call in each place), and mention in the warning that
  the keys will not be written back.

### S4. Integer-vs-float categorical mismatch is silent (pre-existing, made visible by this piece)
- **What.** `score_categorical` now compares as strings, which is correct and matches the encoder
  (`CategoricalEncoder.transform` also casts to Utf8). But `pl.Int64` 4 becomes `"4"` and
  `pl.Float64` 4.0 becomes `"4.0"`.
- **Failure scenario (measured).** Model trained on an integer `VehPower`, scored on the same
  column delivered as float (what pandas does to any integer column that acquires a null): **100 %**
  of rows fall into the Other row. `rm.predict == fit.predict` still holds (both are wrong the same
  way), so the invariant suite cannot see it; the actuary sees a flat relativity and no message.
- **Fix (either).** (a) In `_scoring.score_categorical` and `CategoricalEncoder.transform`, cast
  float columns whose values are all integral to Int64 before Utf8; or (b) cheaper and safer: warn
  in `RateModel.predict` when a categorical variable sends more than, say, 50 % of rows to Other
  while its training table has more than one level. Option (b) also catches renamed levels.

### S5. `GLMFit.predict` silently drops a missing offset column; `RateModel.predict` now warns
- **What.** `src/easy_glm/core/fit.py:181` — `if offset is None and self.offset_col and self.offset_col in data.columns` — falls through without a message.
- **Failure scenario.** A holdout frame without `logprem`: `fit.predict` and `rm.predict` agree
  (both omit the offset), the exactness test passes, and a rate-change model is evaluated without
  its offset. Same warning as `_apply_offset` is enough.

## 4. Nits

- `RateModel._from_dict` accepts a file whose `variables[*].type` is unknown; the error only comes
  from `predict`. `rate_model_tables`/`to_excel`/`_mask_for_row` treat anything non-numeric as
  categorical, so an "interaction" table from a newer file exports as a categorical sheet without
  complaint. Validating `type in _SCORERS` at load time gives the clear message one step earlier.
- `_metadata_from_dict` drops unknown metadata keys silently and `_to_dict` rewrites the file, so a
  key added by a 0.4.x patch without a version bump is lost on save. A warning, as for projects,
  would be consistent with R1.
- `RateModel._migrate` does not touch `snapshots[*].metadata`; `switch_to` tolerates it via
  `_metadata_from_dict`, so it works, but a v1 file rewritten as v2 still has v1-shaped snapshot
  metadata. Cosmetic.
- `RateModel.diff` lists rows present in `v2` only; rows present in `v1` but absent from `v2`
  are not reported. Tables do not change shape today, so no failure scenario yet.
- `int(raw.get("format_version", 1))` raises `TypeError` on `"format_version": null`. Treat
  `None` as 1.
- Two `assert isinstance` dispatches remain outside the two named in §7.9:
  `src/easy_glm/app/pages_design.py:233` and `src/easy_glm/workflow/export.py:44`.
- `test_versionless_0_3_file_loads_and_scores_identically` builds its "0.3 file" from
  `_to_dict()` of the new code and pops keys, so its snapshots still carry v2 metadata. It passes,
  but a real fixture (see §5) is a stronger guard; I generated one from the `v0.3.0` tag and it
  loads identically.
- Invariant 3 in `tests/test_invariants.py` (`rm.to_excel` reflects adjustments) passes on the
  pre-C1 code too — `RateModel.to_excel` was already adjustment-aware; the real §7.1 bug was in
  `app/ui.py::excel_bytes` and the exported script. The guards for those are the two tests in
  `TestDiffAndExports`; the docstring of `test_invariants.py` should not claim invariant 3 covers
  the bug.
- Untracked files `model.easyglm`, `demo_model.easyglm`, `my_model.easyglm` sit in the repo root.
  Not committed, so not a scope problem, but a `.gitignore` entry for `*.easyglm` at the root
  would prevent an accident.
- `README.md:86` still shows `eglm.to_excel(...)` as the headline Excel export; since that is
  now labelled "fitted (pre-adjustment)", add one sentence pointing to `rate_model.to_excel` for
  tables that include editor changes.
- The `fitted` column changes the Excel layout of every per-variable sheet (new column between
  `label` and `relativity`). Any consumer reading by column position breaks. Worth one line in the
  0.4 changelog.

## 5. Missing tests (one line each)

- **D7** — default A/E formula derived from metadata: with `divide_target_by_weight=True`,
  `weight_col == exposure_col`, `compute_actual_expected(..., formula=<derived>)` gives overall train
  A/E = 1 within the fitter's tolerance (I measured 3e-5 at default `gradient_tol`; 1e-6 as written
  in §5 is not achievable without tightening the solver — state the tolerance used).
- **D7** — the same with `exposure_col=None` must *not* report 0.6 (whatever fix S1 chooses).
- **C/B1** — a real 0.3 fixture (`.easyglm` written by the `v0.3.0` tag, with two snapshots and one
  adjustment) checked into `tests/fixtures/`, loaded and scored against saved predictions.
- **C/B1** — `_from_dict` on a file with an unknown `variables[*].type` gives the clear message
  (currently only `predict` does).
- **C/B5** — offset column with nulls: `rm.predict` and `fit.predict` are NaN on the same rows
  (measured 8/8; nothing asserts it).
- **C/B5** — `GLMFit.predict` warns when `offset_col` is missing (after S5).
- **All** — `tests/test_invariants.py` integer-categorical case should have nulls in the integer
  column (`VehPower` has none; `_data` only nulls `DrivAge` and `Region`). My run with 38 null
  `Int64` values passed at 4e-16, so it is a one-line change to the fixture.
- **Project** — unknown top-level / `data` / `design` keys warn (after S3).
- **Excel** — `rate_model_tables` on a model with no snapshots writes no `fitted` column and does
  not crash (works today; untested).

Items in §5 under **C** proper (`generate_blueprint` gone, `from_rate_tables` rebuild) belong to
C2 and are correctly absent here.

## 6. What I re-ran and what I saw

- Full suite: `226 passed, 88 warnings in 117.57s`. `ruff check .`: clean. `black --check .`:
  83 files unchanged.
- `git diff 41e0dbf..HEAD -- tests`: only `tests/test_design_fit_tables.py:501–502` changed, from
  a four-column to a five-column list (`fitted` added) plus an equality assertion between `fitted`
  and `relativity`; **no tolerance or golden number changed**. `docs/` diff is the new check
  document only; the plan is untouched. CI diff removes only the unused `EASY_GLM_MAX_ROWS` (no
  reader in `src/`, `tests/`, `scripts/*.py`).
- New tests against the pre-C1 tree (`git archive 41e0dbf`, `PYTHONPATH`): `test_invariants.py`
  **6 failed, 9 passed** (`categorical_integer`, `mixed`, `mixed_with_offset` × predict and JSON),
  exactly as the builder reported; `test_c1_foundations.py` fails at import. The new tests are not
  tautologies.
- `scripts/checks/c1_foundations.py`: all asserts pass. Numbers: rate tables vs GLM **8.9e-16**
  (document says 1.2e-15 — see S2), with offset **2.0e-15**, VehPower distinct relativities
  **6 of 13**, Excel value **3.0000**, holdout A/E **1.0190**, Gini **0.3056**, deviance explained
  **4.57 %**, alpha **0.00100**, first five driver-age bands 0.8805 / 0.6217 / 0.6217 / 0.6217 /
  0.6594 — all identical to the document except the first residual. I restored the document with
  `git checkout` afterwards.
- Genuine 0.3 round trip: `.easyglm` written by the **`v0.3.0` tag** (two snapshots, one
  adjustment, `switch_to` exercised) loads under `warnings.simplefilter("error")` with no warning;
  `predict` (current, `version=1`, `exposure_col=None`) is **bitwise identical** to the old code's
  saved predictions; rewritten file has `format_version: 2` and the four new metadata keys;
  `clone()` and `switch_to` preserve metadata; version 3 is refused with
  "This .easyglm file is format version 3; this easy_glm reads up to version 2. Upgrade easy_glm to
  open it."
- Genuine 0.3 project (`Project.to_dict()` from the `v0.3.0` tag, `version: 1`): loads with no
  warning, becomes version 2, `validate()` empty, model config intact.
  `from_dict` reads `version` before `_migrate` overwrites it (checked the code path, and
  `{"version": 3}` is refused).
- Adversarial exactness (`rm.predict(df, exposure_col=None)` vs `fit.predict(df)`, max relative
  difference): Int64 categorical with 38 nulls **4.4e-16**; levels literally named `"None"` and
  `"nan"` alongside real nulls **3.3e-16** (and after JSON round trip); float categorical
  **2.2e-16**; mixed + offset + unseen level + nulls **1.4e-15** with `exposure_col=None`, default
  exposure and explicit exposure; `offset_is_log=False` on the raw premium column **1.4e-15**
  (survives JSON); Gamma severity with offset **6.7e-16**; offset column absent → one `UserWarning`
  ("Offset column 'logprem' not found …") and no crash; nulls in the offset column → NaN on the same
  8 rows in both.
- Editor: `streamlit.testing.v1.AppTest` on `src/easy_glm/ui/app.py` with the migrated 0.3 file
  and with a v2 offset file plus data parquet — **0 exceptions** in both; default formula
  `sum_weighted` for the 0.3 file (`divide_target_by_weight` unknown) and `sum_over_weight` for the
  v2 count model; `editor_args` puts `--server.port 8765` before `--`.
- Actuary document (`docs/checks/c1-foundations.md`): readable and honest. Three wording fixes:
  (1) "Vehicle power: distinct relativities used, 6 of 13 rows" does not demonstrate the fix — the
  table always had 6 distinct values; replace with "share of holdout policies scored with the Other
  row for vehicle power: 0 % (was 100 %)", which the script can compute directly;
  (2) "the editor no longer guesses" is true only for files written by 0.4 — for 0.3 files the
  editor still falls back to the old default, say so; (3) drop the backticked file names and column
  names (`fitted`, `relativity`, the script path) from the prose — the actuary does not need them,
  and the plan asks for no code in this document.
