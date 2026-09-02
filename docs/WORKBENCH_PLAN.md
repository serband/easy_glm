# easy_glm Workbench — design plan

*Status (2026-09-02): Phase 1 (workflow engine) and Phase 2 (Workbench v1, all
nine pages, AppTest smoke tests, Playwright-verified on the French motor data)
are built on branch `core-rewrite`. Phase 3 items are open. This document is the
reference for the GUI programme; update it as decisions change.*

## 1. Goal

An Emblem-class pricing workbench in the browser: an actuary or data scientist
can run the **entire** GLM workflow without writing code — load data, define
variables, find leakage, split, design factors, fit, diagnose, adjust, export
rate tables — and at any point **export the whole workflow as a runnable Python
script** that reproduces the model with the public `easy_glm` API.

Non-goals for now: multi-user server deployment, authentication, a general BI
tool. Those become possible later because of the architecture below.

## 2. Principles

1. **The spec is the product; the GUI is a view of it.** Every click edits a
   declarative `Project` document (JSON). The engine executes the spec. The
   script exporter renders the spec. Nothing lives only in widget state.
2. **Engine first, pixels second.** Every capability exists as a tested,
   GUI-agnostic function in `easy_glm.workflow` before it gets a screen.
3. **Exact and reproducible.** Fit on training rows only; the exported script
   must reproduce the GUI model's predictions to floating-point precision
   (this is a test, not an aspiration).
4. **Fast enough for real portfolios.** Sub-second interactions on ~1M rows by
   caching on spec hashes and computing on samples where exactness is not
   needed (exploration), never for the fit or the tables.
5. **Swappable front end.** Streamlit today; the `Project` + `workflow` layer
   is the stable API a React/FastAPI front end could target later.

## 3. Front-end decision

**Streamlit multipage (`st.navigation`) over a `Project` spec.**

Why: the package already ships a working, tested Streamlit relativity editor;
the user base is Python-only; Streamlit 1.57 has multipage navigation,
dialogs, fragments (partial reruns), editable data grids, Plotly with selection
events and a theming API — enough for an Emblem-style workflow. Build time to a
usable v1 is weeks shorter than a React front end.

Known limits and the exit path: Streamlit cannot do canvas-style drag editing
of curves or true multi-user sessions. If those become must-haves, add a
FastAPI service exposing `easy_glm.workflow` and a React client; the spec and
engine are unchanged. The GAMChanger-style drag editing on the roadmap will be
prototyped as a Plotly `editable` shape component inside Streamlit first.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ easy_glm.app (Streamlit)  — pages, widgets, charts, no maths  │
├──────────────────────────────────────────────────────────────┤
│ easy_glm.workflow         — Project spec, steps, exporter     │
│   project.py   Project / DataSource / Variable / Split / ...  │
│   prep.py      load → rename → recode → derive → filter → split│
│   explore.py   univariate summaries, leakage report           │
│   diagnostics.py  deviance, lift/gini, double lift, A/E by any│
│                variable, alpha path, residual factor search   │
│   registry.py  multiple named fits per project (champion/     │
│                challenger)                                    │
│   export.py    Project → Python script; Project → JSON        │
├──────────────────────────────────────────────────────────────┤
│ easy_glm.core             — DesignSpec, fit_glm, rate_tables,  │
│                             to_rate_model, Excel export       │
│ easy_glm.engine           — RateModel (scoring, snapshots)    │
└──────────────────────────────────────────────────────────────┘
```

Session state holds one `Project` plus cached artefacts (the loaded frame,
prepared train/holdout frames, fits). Autosave writes `project.easyglm-project`
(JSON) after each change; a project can be reopened and every page re-renders
from it. Heavy steps are memoised with `st.cache_data` keyed on the hash of the
relevant part of the spec.

## 5. The Project spec

```jsonc
{
  "version": 1,
  "name": "bike_2026_07",
  "data": {
    "source": {"type": "parquet|csv|sas7bdat|xlsx", "path": "...", "options": {}},
    "sample": {"rows": null, "seed": 42},            // exploration sample only
    "renames": {"cPrem_nep": "premium"},              // applied first
    "roles": {"ClaimNb": "target", "Exposure": "weight", "traintest": "split",
              "IDpol": "id", "DrivAge": "predictor", "...": "ignore"},
    "types": {"VehPower": "categorical"},             // overrides inference
    "recodes": {"cVarN": {"mapping": {"0": "B", "1": "B"}, "default": "keep|Other|<value>"}},
    "derived": [{"name": "Drvr1Exp_Q", "expr": "pl.when(pl.col('Drvr1Lic')=='Q').then(pl.col('Drvr1Exp')).otherwise(0)"}],
    "filters": ["pl.col('cPrem_nep') > 0", "pl.col('aPrem') < 5000"],
    "split": {"mode": "column|random", "column": "traintest", "train_value": 1,
              "fraction": 0.7, "seed": 42}
  },
  "design": {
    "defaults": {"n_bins": 20, "min_level_share": 0.0025, "null_indicator": true},
    "variables": {
      "DrivAge": {"kind": "step", "knots": "quantile|integer|[...]", "monotone": null},
      "Region":  {"kind": "categorical", "levels": "auto|[...]", "min_level_share": 0.01}
    }
  },
  "models": {
    "freq_v1": {
      "family": "poisson", "link": "log", "target": "ClaimNb", "weight": "Exposure",
      "offset": null, "divide_target_by_weight": true,
      "predictors": ["DrivAge", "Region", "..."],           // subset of role=predictor
      "penalty": {"alpha": null, "cv": 5, "n_alphas": 20, "l1_ratio": 1.0},
      "monotone": {"BonusMalus": "increasing"},
      "base": "modal", "base_rate_override": null,
      "adjustments": [{"variable": "DrivAge", "from": 25, "to": 30, "relativity": 0.95}]
    }
  },
  "champion": "freq_v1",
  "exploration": {"leakage": {"ignored": ["IDpol"], "acknowledged": ["cVarN"]}}
}
```

Rules: `roles` are the single source of truth for what a column is; `predictors`
in a model must be a subset of role=predictor columns; `design.variables` is
keyed by *post-rename* names; anything not listed inherits `design.defaults`.
The spec is versioned; loaders migrate old versions.

## 6. Workflow and screens

| # | Page | User does | Engine (workflow) | Exported code |
|---|------|-----------|-------------------|---------------|
| 1 | **Project & Data** | New/open project; pick file (parquet/csv/sas7bdat/xlsx); preview; row/col counts; memory; optional exploration sample | `prep.load_source` | `df = pl.read_parquet(...)` / `pd.read_sas` |
| 2 | **Variables** | Assign roles (target/weight/exposure/offset/split/id/predictor/ignore); rename; override type; recode levels via an editable mapping grid; add derived columns with a polars expression and live preview; add row filters | `prep.apply` (renames→recodes→derived→filters) | `.rename`, `.with_columns`, `.filter` |
| 3 | **Explore** | Univariate panel per variable: exposure histogram, target mean by band (train), missing %, cardinality; **Leakage report** ranking every candidate with reasons; one click sets role=ignore or "acknowledge" | `explore.univariate`, `explore.leakage_report` | comment block listing ignored variables |
| 4 | **Split** | Column split or random split (fraction, seed); shows train/holdout exposure, target rate balance | `prep.split` | `train = df.filter(...)` |
| 5 | **Design** | Per predictor: kind (step/categorical), knot strategy (quantile n / integer range / custom list, editable), null indicator, level share threshold, monotone direction; preview: exposure per bin + target mean per bin on train; design size counter | `DesignSpec.from_data` + overrides | `DesignSpec({...})` written out explicitly (knots as literals) |
| 6 | **Model** | Family/link, target/weight/offset, penalty (alpha slider or CV), predictors checklist, fit button; results: alpha chosen, deviance train/holdout, non-zero terms, **regularisation path chart** (deviance vs alpha, coefficients vs alpha) | `fit_glm`, `diagnostics.alpha_path`, `registry` | `fit_glm(train, spec, ...)` |
| 7 | **Diagnostics** | A/E by variable (train vs holdout, with exposure bars), lift chart (deciles), Gini/AUC, double lift vs a benchmark column (e.g. current premium), calibration; **residual factor search**: A/E by every *unused* variable to spot missing factors; champion vs challenger overlay | `diagnostics.*` | not exported (report only) |
| 8 | **Rate tables** | Existing relativity editor (baseline vs working, snapshots, A/E recompute); adjustments are recorded in the spec; export Excel / `.easyglm` | `rate_tables`, `to_rate_model`, `RateModel` | `to_rate_model`, `update_relativity(...)` per adjustment |
| 9 | **Export** | Download: Python script, project JSON, Excel rate tables, `.easyglm`, HTML report | `export.to_script`, `export.to_report` | — |

Navigation is free (any page any time); each page shows a status chip for its
upstream prerequisites (e.g. Model page: "design uses 12 predictors, 2 changed
since last fit → refit").

## 7. Leakage detection (Explore page)

For every column with role predictor or unassigned, computed on the training
sample, each producing a score and a human-readable reason:

* **Single-factor strength**: one-variable GLM (same family as the champion,
  step/categorical encoding) — % of null deviance explained. > 40 % → *suspicious*,
  > 80 % → *almost certainly leakage*.
* **Target proxy**: |Spearman ρ| with target > 0.9 (numeric); target is constant
  within levels (categorical, η² > 0.95).
* **Identifier-like**: distinct/rows > 0.9 (or > 0.5 with no numeric order).
* **Post-outcome naming**: name matches `claim|incur|paid|loss|settl|recover|
  reserve|cost|date|dt_` → *check*.
* **Missingness signal**: single-factor strength of the null indicator alone.
* **Degenerate**: constant, or one level with > 99.5 % exposure.

The report is a table (variable, flags, score, reason); actions: *ignore*
(role=ignore, recorded in `exploration.leakage.ignored`) or *acknowledge*
(keep, recorded, no longer nags).

## 8. Script export

`export.to_script(project, model="freq_v1") -> str` renders a linear,
readable script with section comments mirroring the pages:

```python
# 1. Data
df = pl.read_parquet("policies.parquet").rename({...})
df = df.with_columns([...recodes..., ...derived...]).filter(...)
# 2. Split
train = df.filter(pl.col("traintest") == 1)
# 3. Design (explicit knots so the script is self-contained)
spec = DesignSpec({"DrivAge": StepEncoder("DrivAge", [21, 24, ...]), ...})
# 4. Fit
fit = fit_glm(train, spec, "ClaimNb", family="poisson", weight_col="Exposure",
              divide_target_by_weight=True, alpha=0.00031, monotone={...})
# 5. Rate tables + adjustments
rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")
rm.update_relativity("DrivAge", 25.0, 30.0, 0.95)
rm.to_json("freq_v1.easyglm"); EasyGLM(fit, rm).to_excel("freq_v1.xlsx")
```

The exporter writes the **resolved** alpha (never `cv=`) so the script is
deterministic; a comment records that CV chose it. Test: run the exported script
in a subprocess and assert its `.easyglm` predictions equal the in-session
model's on the holdout.

## 9. Engine additions (all tested, no Streamlit imports)

* `workflow/project.py` — dataclasses + JSON (de)serialisation + validation
  (roles consistent, predictors ⊆ predictor-role columns, one target, ...).
* `workflow/prep.py` — `load_source`, `apply_variables` (rename/recode/derive/
  filter, expressions evaluated with a restricted namespace `{pl, np}`),
  `split`, `train_holdout`.
* `workflow/explore.py` — `univariate(df, var, target, weight, n_bins)`,
  `leakage_report(df, project)`.
* `workflow/diagnostics.py` — `deviance`, `lift_table`, `gini`, `double_lift`,
  `ae_by_variable` (any variable, quantile bands for numerics not in the
  model), `alpha_path(fit)`, `residual_factor_search`.
* `workflow/registry.py` — `ModelRun` (spec snapshot + fit + tables + metrics),
  compare runs.
* `workflow/export.py` — `to_script`, `to_json`, `to_report` (HTML).
* `core` additions: `DesignSpec.with_overrides`, integer-knot strategy helper.

## 10. Delivery phases and acceptance

**Phase 1 — workflow engine (this session).** Project spec; prep steps; leakage
report; diagnostics; script exporter; tests including the script round-trip.
*Accept when*: `pytest` green; exported script reproduces a French-motor model.

**Phase 2 — Workbench v1 (this session, continues next).** `python -m
easy_glm.app` launches the multipage app; all nine pages functional on French
motor data; autosave/reopen works; existing editor embedded as the Rate tables
page. *Accept when*: a model can be built end to end without code and the
downloaded script runs.

**Phase 3 — polish.** Theming and layout, champion/challenger overlays, HTML
report, CLI (`easy_glm gui`, `easy_glm run project.json`), performance pass on
1M+ rows, drag-to-edit prototype, two-way interactions once the core supports
them.

## 11. Testing strategy

* Engine: unit tests per function on synthetic data; property: prep is
  idempotent given the same spec; leakage report flags a planted proxy and an
  ID column; diagnostics agree with hand calculations on tiny frames.
* Exporter: golden-file test for script text; round-trip execution test.
* App: smoke tests with Streamlit's `AppTest` for each page (renders without
  exceptions on a small project); Playwright screenshot run in CI as an
  artefact, not an assertion.

## 12. Risks and mitigations

* *Streamlit rerun cost with big frames* → keep frames in session state, cache on
  spec hashes, fragments for widgets that only touch one panel.
* *Expression injection in derived columns* → evaluate with a namespace limited
  to `pl`, `np` and no builtins; documented as "trusted local user" anyway.
* *Spec drift between GUI and script* → the exporter is the only renderer and is
  tested by execution, not by string comparison alone.
* *Scope creep* → every page ships in its minimal form first; polish is Phase 3.
