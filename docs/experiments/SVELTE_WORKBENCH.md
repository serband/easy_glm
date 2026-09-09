# Svelte workbench experiment

Branch: `codex/svelte-workbench`, based on main `0828b89` (0.452).
This is a runnable Variables → Model → Diagnostics → Rate tables slice.
It does not replace the full workbench.
No version bump, publication, merge or migration of saved workspaces is part of it.

## Try it

From this checkout, using the Python environment you use in Positron:

```sh
python -m pip install -e ".[desktop]"
python -m easy_glm.desktop --port 8765
```

Open `http://127.0.0.1:8765`. With no project argument, it uses a generated
12,000-row motor sample, without downloading data. To inspect a saved project:

```sh
python -m easy_glm.desktop --project /path/to/project.easyglm-project.json --port 8765
```

Or from a Python / Positron session:

```python
from easy_glm.desktop import launch
process = launch(data=df)  # pandas or Polars; opens the default browser
# Alternatively: process = launch("/path/to/project.easyglm-project.json")
# To stop this experiment:
process.terminate()
process.wait()
```

The helper selects a loopback port, uses the same interpreter, isolates the child
from injected PYTHONPATH/PYTHONHOME, and prints the actual URL and server log.
It checks readiness and the identity of the new process before returning. Node
is not required to run the installed package. On Windows, use the same commands
in an installed Python environment; this has **not been tested on Windows or
inside Positron**. Corporate browser/network restrictions remain undiagnosed.

Applied edits stay in the server's in-memory project copy. Browser reloads read
that copy, but stopping the server loses it. **Export project** downloads the
applied project specification for retention. Unsaved browser drafts are lost on
refresh. Source files and existing fit caches are never written by the API.
Save / Close / Resume and durable workspaces remain proposals.

## What works

- Compact table of source names, editable final names, roles and modelling types.
  Search and role filters operate locally. The long variable list renders at most
  28 rows at a time (change previews render at most 18); it represents wide datasets as a vertical variable list.
- Role-only JSON shows all ten roles: six singleton `null` defaults and four
  group `[]` defaults. It uses stable source column names. Omitted columns become
  ignored, matching the established bulk semantics; explicit unassigned stays
  unassigned. JSON changes preserve table names and types.
- Switching editors reconciles the draft locally. Invalid JSON blocks switching
  until corrected or reset. Reset restores the last applied snapshot. Preview
  validates the entire draft; Apply is a separate explicit action.
- Shared `workflow.variables` rules implement atomic rename swaps and canonical
  `Project.rename_column` / `apply_role_change` behaviour. The Streamlit page
  re-exports the existing helpers, so existing callers/tests continue to work.
  Changing a role does not add a predictor to a model; removing eligibility clears
  affected model references/interactions with a notice in Preview. No edit refits.
- Existing `workflow.explore.univariate` supplies one distribution chart. Hover
  shows band/level counts; zoom expands its horizontal scale. It uses applied
  settings, includes row filters/recodes/types, and limits exploration to the first
  50,000 source rows. It is a row-count distribution, not an A/E diagnostic or a
  representative sample guarantee. Categorical previews use the engine's 25-level
  limit. Unsupported/preparation errors appear as messages.
- Concurrent tabs use revision checks: stale Apply returns a conflict message.
  Reload applied settings discards that tab's draft and reads the latest copy.

## Architecture and local boundaries

Svelte 5 compiles to static JS/CSS using Vite. Those assets are checked into
`src/easy_glm/desktop/static` and included in the Python wheel. All scripts,
styles and fonts are local/system resources. No CDN, telemetry or external
service is needed for the UI. The existing Python package dependencies are still
installed; this experiment does not yet slim the full package's dependencies.

FastAPI serves the UI and a small same-origin API on **127.0.0.1 only**. It
validates the exact Host and Origin, refuses cross-site fetches, uses a
per-process API token, disallows framing, and bounds incoming drafts to 8 MB.
No API accepts arbitrary file paths or executes user code beyond the already
configured workflow expressions in a project opened by the launcher. This is a
local single-user application, not a network server or a sandbox for untrusted
project files. Other processes running as the same user are inside its boundary.

Browser drafts, filtering, role counts, view switches, resets and chart hover/zoom
run in JavaScript. Preview/Apply call the shared workflow layer. A short lock
protects atomic project changes. Plot aggregation uses FastAPI's worker thread
pool and releases the project lock before computation. Fits run in a separate Python process using immutable project/data copies in a
temporary directory. Requests only start jobs or poll their status. Progress,
cancellation and setting fingerprints prevent an obsolete result from being
used. The worker calls canonical preparation, `workflow.run_model`, diagnostics
and rate-table functions. One fit runs at a time. Completed results are temporary
and are lost when the server stops; existing Streamlit caches are untouched.

## Acceptance goals and measured results

These measurements describe the original Variables slice, not the expanded model
UI or corporate laptops:

| Measure | Goal | Observed on this Mac |
| --- | --- | --- |
| Fresh server process ready, French 50k sample | median under 2 s | 1.418 s; five runs, range 1.406–1.430 s |
| 2,000-column initial browser load, server already ready | under 500 ms | 123 ms including initial plot |
| Local variable edit to paint | median under 100 ms | 23.2 ms; maximum 32.6 ms |
| Table to JSON to paint | median under 100 ms | 32.2 ms; maximum 32.5 ms |
| Reset to paint | median under 100 ms | 32.4 ms; maximum 32.6 ms |
| Draft edit/switch/reset network requests | zero | zero |
| Rendered variable rows on 2,000 columns | at most 40 | at most 28 |
| Compiled JS + CSS before HTTP compression | under 150 KB | approximately 65 KB (about 25 KB gzip) |

Measurement environment: macOS 26.6.2 arm64, Python 3.14.7, Chrome headless.
Fresh Python processes were used, **OS disk caches were not flushed**. The
first observed server launch took 1.98 s. Browser timing samples use DOM events
and two requestAnimationFrame callbacks (a paint opportunity, not a physical
display measurement), 12 samples per interaction; source fixture is 2,000 columns
× 2,000 rows. Raw samples: `svelte-browser-timings.json`. These are not comparative
Streamlit measurements and do not establish Windows performance.

## Build and verification

Frontend contributors need Node/npm; users of a built wheel do not:

```sh
cd frontend
npm ci
npm run check
npm run build
npx playwright install chromium  # only on a test machine without its browser
npm test
npm test -- --config playwright.wide.config.js
npm test -- --config playwright.session.config.js
npm test -- --config playwright.models.config.js
```

For an installed Chrome, set `PLAYWRIGHT_CHANNEL=chrome` in the test environment.
The two configurations start isolated synthetic test servers on ports 8770/8771.
Python API tests require the desktop extra plus httpx:

```sh
python -m pip install -e ".[desktop,dev]" httpx
pytest -q tests/test_desktop_api.py tests/test_variable_sync.py tests/test_w3_hardening.py
python scripts/bench_desktop.py --project /path/to/project.json --runs 5
python -m build --wheel
```

API integration tests cover atomicity, model references, stale revisions,
malformed drafts, host/origin/token protections, payload limits and scan-free
wide metadata. Browser tests exercise table/JSON sync, reload, reset, singleton
reassignment, explicit Apply, graph zoom and no external requests. A separate
wide test checks virtualization, search/edit/apply of the last column and timing.

Validation on 9 September 2026: full Python suite **937 passed, 1 skipped,
1 slow deselected** (372 s). After the final local-boundary/launcher additions,
all **12 focused API/launcher tests** passed. **Four browser integration tests**
passed (three Variables flows and one wide-schema flow). Black, Ruff, mypy
(20 modules), Svelte checks and the production build passed. The final wheel was
installed into a separate Python environment; `launch(data=df)` and bundled
assets worked with Node absent from PATH (2.12 s for that readiness + HTTP smoke).
No Windows, Positron, CI matrix or corporate-network result is claimed.

During local development, macOS marked the editable-install `.pth` file hidden,
so Python 3.14 ignored it. The isolated checkout uses the source symlink prescribed
in AGENTS.md; the separately installed wheel did not require that workaround.

Reference implementation choices follow [Vite's production build guidance](https://vite.dev/guide/build)
and [FastAPI's middleware guidance](https://fastapi.tiangolo.com/advanced/middleware/).

## Model and results slice (9 September 2026)

Open **Design & models** after applying Variables. Choose an existing split column, or
select **Seeded random** and **Apply split**. A generated random split does not
need a source column with the split role. The page reports actual training and
holdout row counts and missing prerequisites.

Create or select a model, choose predictors and their inferred/step/linear/
continuous/categorical designs, family/link, target, weight, offset, fixed alpha
or cross-validation, L1 ratio and table base. **Create model** / **Save model**
apply settings; **Fit model** starts a separate worker. Navigation and Variables
drafts remain available during fitting. The job shows progress, errors and
**Cancel fit**. Existing interactions, monotone rules, custom knots/clamps,
penalty weights and adjustments survive basic edits; unsupported combinations
are refused. Editing these advanced settings is outside this slice. Design
kinds and defaults are project-wide, matching the canonical project model.

After completion, **Diagnostics** shows training/holdout totals, A/E, Gini,
deviance measures and an actual-versus-expected risk-decile chart. **Rate tables**
shows canonical exported rows, including linear slopes, with bounded rendering
and paging. Results are available only for a completed fit matching the applied
settings. Variable, split or model changes invalidate affected results and never
start a fit automatically.

Table editing and detailed diagnostics are described in the restoration below.
Champion comparison, result exports and durable fit recovery remain unimplemented. The
existing **Export project** downloads the applied specification. Server restarts
retain neither in-memory edits nor fits unless the specification was exported
and supplied at launch. Open a fresh tab for a new UI build; keep an older tab
open until any unsaved draft has been copied.

Model browser checks cover split/create/fit, responsive navigation, both metric
subsets, a real lift chart, continuous slope tables, invalidation and Variables
draft retention. Backend checks exercise the actual worker, cancellation,
readiness and preservation of expert settings. Existing session and wide-schema
regressions are also run. See the final validation record below.

## Remaining work

Windows/Positron validation, advanced design and interaction editing, comparison,
exports, and an explicit durable Save / Close / Resume
policy are separate follow-ups before any replacement of Streamlit defaults.

## Session recovery correction (9 September 2026)

An open tab could survive a development-server restart with its old API token.
The server correctly rejected that token, but the client fetched credentials
only on initial mount; even its old "Reload applied settings" action reused the
expired token. The exact reported alert came from this token check. A direct
request for the VehBrand plot returned 200 with the current token and the same
reported 403 message with a stale token. This does not diagnose the separate
Windows/corporate-network issue.

The client now re-bootstraps same-origin credentials once on an explicit
`session_expired` response, sharing concurrent bootstrap requests. Session
responses and fetches are non-cached. Read requests recover without replacing
table drafts or raw JSON, including incomplete JSON. Host/Origin refusals still
fail; no general 403 bypass or token persistence was added.

Every edit is bound to a server session as well as a revision. After a restart,
an old Apply is refused even if both servers happen to be at revision 0. The
applied baseline is refreshed while the draft is kept; the user previews it again
before applying. A different project at the same address cannot inherit that
draft. Reconnect keeps the draft; **Discard draft and reload** explicitly replaces
it, and **Download draft** retains names/types plus raw role JSON when needed.
No page reload is used for recovery by the corrected client.

The one-time upgrade cannot replace JavaScript already running in an old tab.
Keep that tab open if it contains a draft and open the workbench URL in a new tab.
Copy any old draft before refreshing the old tab; this change does not claim to
recover unobservable drafts from an already-loaded old client.

Verification: 13 API/launcher checks and nine browser cases passed, including
actual process restarts, bare URL, refresh/new tab, plot selection, table/JSON
draft preservation, stale Apply rejection, Preview/Apply after recovery, origin
refusal, and a simulated different-project response. Existing Variables and wide
schema tests remained green. No additional core/modelling changes were made.

## Model slice validation record

The 11 browser cases passed: two model flows, three Variables cases, five session
recovery cases and one wide-schema case. All 19 focused API/worker/launcher
checks passed, including cancelling a running process, worker failure, and
invalidation after variable, model and split edits. Svelte reported zero errors
and warnings; Black, Ruff, mypy and the production build passed. An installed
wheel completed a real child-worker fit with Node absent from PATH.

A separate copy of the French motor 50,000-row fixture fitted with the existing
two-stage DrivAge × BonusMalus interaction and returned nine rate tables plus
training and holdout diagnostics. No live review settings were changed for this
check. Visual inspection covered model setup, diagnostics and linear rate tables.
The expanded compiled JS/CSS totals 100,509 bytes (35,715 bytes gzip). The repeated
wide-schema check measured 24.9 ms median edit-to-paint, 32.3 ms table-to-JSON,
32.5 ms reset and 28 rendered rows. These remain Mac-only measurements.

The live review upgrade backed up the old assets and applied project, and
verified exact project-spec equality after restart. Original project/data/cache
hashes, main branch state and the Streamlit health endpoint were checked.

Full Python regression: **945 passed, 1 skipped, 1 slow deselected** in 316 s.
Two additional failure/invalidation tests and the strengthened running-process
cancellation check passed in the final 19-test focused run. Benchmark convergence
warnings and dependency deprecation warnings were reported; no tests failed.

## Restored diagnostics and table editing

The Diagnostics page now includes variable A/E charts and values, two-variable
A/E heatmaps, missing-factor search and missing-interaction search. Searches run
only on training rows through the existing noise-adjusted residual functions.
Inspect a candidate before adding it; **Add and review model** updates the model
(and promotes an eligible unassigned factor), then returns to setup for an
explicit refit. No search changes model settings by itself.

The Rate tables page supports individual row and interaction-cell relativity
edits, moving-average and isotonic smoothing, cap/floor, decimal/increment
rounding, base-rate rebalancing, undo/redo, named snapshots and reset to fitted.
Changes are previewed before applying. Charts show actual rates, the original
fitted rates, current adjusted rates and proposed rates. The preview reports
real training expected claims before/after and the fitted total. A/E tables
retain exposure counts; empty groups are omitted from the rate chart.

The original fit is retained in an application-owned temporary worker artifact.
Reviews run in separate processes and re-use that fit: they do not call the GLM
solver. Worker inputs and output paths are internal, never supplied as browser
file paths. Review previews are bound to both project revision and fit identity,
so a stale tab or a new fit cannot apply an old preview. Reads, searches and
previews do not mutate the project; Apply changes only adjustments/base rate.
Named snapshots store those settings in the exported project. Undo/redo holds
up to 50 steps in the server session and includes the base rate.

Canonical table edit rules are shared with Streamlit. A failed multi-row edit
is refused atomically; smoothing leaves null/Other rows unchanged, requires an
explicit meaningful order for categories, and refuses interaction tables.
Manual interaction-cell editing is supported where exposure exists. Rebalancing
logit probabilities by scaling their base rate is refused. Existing fit warnings
remain visible after table edits.

Navigation stays usable during reviews. Closing a panel cancels its unfinished
review, and late asynchronous responses cannot start an orphan review. Variables
continues to keep its unsaved table and JSON drafts. The one-time live upgrade
backs up the applied specification and rebuilds its previously fitted models;
subsequent table adjustments use those retained fits without refitting. Fits and
undo history are still temporary, so Export project remains the retention path
for applied settings and named snapshots.

Restoration validation: full suite **950 passed, 1 skipped, 1 slow deselected**
in 361 seconds; the final focused desktop run passed **23 checks**, including
interaction inclusion and cell editing added after full-suite collection. The
12 browser cases cover the restored workflow, model setup, Variables, restarts
and wide schemas. The restored flow also passed at an 884-pixel viewport without
page overflow. Wide-variable edits remained approximately 25 ms median, with 28
rows rendered. Svelte checks, Black, Ruff and core/workflow mypy passed. Main,
source data, original project/cache hashes and the separate Streamlit session
were preserved. No push, merge or release is part of this experiment.

Final installed-wheel smoke: a real fit and background variable A/E review passed
with Node absent from PATH. The check exposed a pre-existing launcher edge case
when a random identity began with a dash; passing it as one option/value argument
fixed it, and both launcher regression checks passed. A fresh browser tab now
recognises existing fits immediately. Adding a previously unassigned factor also
updates the clean Variables view, while an existing draft remains intact.

Rate-table presentation: the relativity chart is primary, followed by a collapsed
**Rate table** section containing edits, paging, All columns and row-preview
actions. Collapsing keeps the mounted grid and its draft edits. Numeric factors
use connected staircase lines for step bands or curves following the exported
log slopes for linear bands; clamp bands stay flat and nulls are separate points.
Categorical factors use paired fitted/current bars, including numeric-looking
category levels. Chart kind follows applied design overrides and prepared dtype
metadata using the workflow encoder rules, never labels or table-row values.
Interaction cells retain their matrix view. Charts cover the displayed page;
training exposure aligns below relativity and subset exposure below A/E.

Validation covers step jumps, linear slopes, null/clamp mapping, categorical type
resolution, collapsed order and draft retention, preview/apply/undo, model and
Variables navigation, and visual checks at 884 × 773. This update changes static
assets only; the live applied project, fit identities and server session survive
without restarting or refitting. Current-page navigation buttons remain hidden.

## Workflow layout alignment

Reference: the running Streamlit Model page and `app/main.py`,
`pages_variables.py`, `pages_model.py`, `pages_diagnostics.py`, and
`pages_tables.py`. The Svelte colours and asynchronous workers remain intact.

| Streamlit pattern | Svelte layout |
| --- | --- |
| Ordered Workflow navigation and setup checklist | Project & data, Variables, Explore, Model, Diagnostics, Rate tables, Export; applied setup progress in sidebar |
| Project context before setup | Read-only project/data overview, applied settings expander and next-step actions |
| Roles, names and types separate from exploration | Variables retains grid/JSON/preview; distribution moves into Explore |
| Model definition → Factor design → Fit and results | Same section order, in-page links, collapsed shared defaults and a split expander by fit settings |
| Compact metrics then diagnostic tabs | A/E by variable (default), A/E by pair, Lift, Residual factors/interactions |
| Chart, table and optional adjustment tools | Chart first; expandable editable Rate table below; tools/snapshots remain expandable |
| Export as a workflow step | Applied project JSON download with precise contents and limitations |

Model selection is shared across the model/results pages; Variables drafts stay
mounted across navigation. Diagnostic tab switching retains its component and
uses background reviews for the selected view. Navigation and chart/table edits
remain client-side, with revision checks unchanged.

This is layout alignment for the implemented subset, not full feature parity.
Source upload/selection, recodes/derived/filter editors, leakage exploration,
detailed knot/clamp/monotonicity/interaction editors, champion comparisons,
double lift, coefficients/regularisation paths and Excel/report/script/scorer
exports remain in Streamlit. Project & data lists these gaps; saved settings
continue to be retained. No placeholder pages imply that these features work.

Alignment validation: 13 browser cases passed across restored reviews, model
setup, Variables/workflow navigation, real restart recovery and the wide-schema
editor. At 2,000 rows × 2,000 columns, at most 28 rows rendered; median edit paint
time was 24.2 ms. Model and diagnostics layouts were visually inspected at
884 × 773. Svelte check/build and whitespace checks passed. The static-only live
update retained the applied project, fit identities/status and server session;
original main, project/data/cache hashes and Streamlit health were verified.

### Diagnostics parity restoration

Diagnostics now follows the original workbench's analytical workflow using its
existing workflow functions. The diagnostics gap list above describes the earlier
layout checkpoint; champion comparisons, double lift, coefficients and paths are
now implemented.

| View | Restored behaviour |
| --- | --- |
| Shared comparison | Sidebar default and page selector across Diagnostics, Compare and Rate tables; designate the project champion without refitting |
| Metrics and facts | Training, holdout and all rows; target totals, deviance, Gini, fitted settings and recorded table-version metrics; family limitations explained |
| A/E by variable | Fitted bands/levels, actual/fitted/current/challenger rates, training and holdout, aligned exposure; numeric lines and categorical bars |
| A/E by pair | Fitted main-factor grouping, current/challenger A/E heatmaps, exposure and cell values; temporary numeric bins for unfitted variables |
| Lift and double lift | Equal-exposure lift for both models; double lift against a challenger or a null benchmark calibrated only on training rows |
| Residual factors | Full signal statistics, multi-factor selection, inspection, add to model then review/refit |
| Interactions | Residual search, inspection with the search's coarse numeric bins and fitted categorical levels, add then two-stage refit |
| Regularisation | Separate stages and L1 ratios, CV/training deviance, selected penalties and retained coefficients; fixed-alpha table retained |
| Coefficients | Kept/all original fitted coefficients with complete CSV download |
| Relativity comparison | Canonical union of numeric edges and matching categorical/cell labels; base-rate change and tolerance |

Analysis runs on demand in a worker. Comparisons require the same target and
exposure basis. Requests and editable previews are bound to both fit identities;
refitting either invalidates an old comparison. Gini is unavailable for signed
actuals/predictions; double lift also requires a positive benchmark. Binomial Gini
is an exposure-weighted ordering measure, not ROC AUC. The rate-table workbench
continues to support log/logit links only.

A private launcher option `--restore-session` supports this local upgrade from
application-owned fit artifacts. It validates applied model settings and raw data,
rebuilds adjusted tables and results from saved coefficients, and does not refit.
It is not a public import format or an HTTP endpoint. Undo/redo stacks remain
session-only: this upgrade was rehearsed with an empty live undo/redo history;
project adjustments and named snapshots remain in the exported project.

Validation: the full Python suite passed 958 tests (one skipped, one deselected),
with the final focused parity/edit checks covering subsequent challenger-staleness
and heatmap refinements. Fourteen browser cases passed, including a two-model CV
flow, both stages of the regularisation path, search/add/refit, smoothing/edit/undo,
actual server restarts and draft retention. A numeric-to-categorical switch checks
that exposure bars follow the new groups. At 884 × 773 the charts and controls were
visually checked; the 2,000-column editor retained 28 rendered rows and 24 ms median
edit response. Svelte check/build, Black/Ruff and core/workflow mypy passed.

The live French motor session was upgraded from a fresh private backup after a
separate-port rehearsal. The applied project, Frequency fit identity and original
2.402-second elapsed time, rate-table rows and train/holdout metrics were preserved.
Live variable A/E, lift, double lift, path and coefficient queries passed. Original
main, project/data/cache hashes and the Streamlit session were left intact.

### Rate adjustment workflow restoration

The adjustment methods now sit directly below the primary relativity chart,
before the collapsed manual grid: Moving average, Isotonic smoothing, Cap / floor
and Round. Each exposes only its relevant parameters and a Preview adjustment
button. Manual editing has a direct button and supports several row edits in one
preview; linear rows represent band-start nodes, and interaction tables offer a
matrix of kept cells on the current page (large matrices use the row editor).

A preview shows current and proposed relativities, current/proposed A/E, the true
training expected-total change and any base-rate change, followed by Apply and
Discard. Changing tool parameters invalidates the old preview. Applying refreshes
the rates and A/E while retaining the tool settings and snapshot controls.
Smoothing shows the preserved weighted log mean separately from the monetary
impact, excludes null/Other rows and requires explicit meaningful-order
confirmation for categoricals. Linear slopes use the canonical node derivation.

Undo/redo, rebalance, reset this variable and reset all are beside the tools.
Current versus fitted expected totals make the off-balance visible. Snapshots
support save, restore, comparison (including original fitted/current versions),
CSV differences and confirmed deletion. Tools and edits continue through the same
canonical row/cell adjustment rules; they never refit the model.

Validation covered 26 new Python cases, including eight tool configurations for
step, linear and categorical factors, lifecycle actions and restoration of a
two-step undo/redo history. The final focused run passed 37 tests; an earlier
broader run passed 115. All 15 browser cases passed, including manual multi-row
edits, every tool configuration, snapshots, categorical ordering and interaction
cell editing with current/proposed A/E. At 884 × 773 the controls and previews
were visually checked. The wide editor retained 28 rendered rows with a 23.9 ms
median edit response. Svelte check/build, Black/Ruff and core/workflow mypy passed.

The private upgrade path now also restores validated session undo/redo history.
A separate-port rehearsal restored and exercised both undo steps and both redo
steps without changing the fit. The live upgrade preserved the applied smoothing
adjustments and captured the prior fit and history in a private backup. A new fit
was initiated in the live session after the restart; that newer fit was retained,
with its normal empty undo/redo history. Final read-only checks confirmed the
project, table values and train/holdout/all diagnostic totals, and unapplied live
previews passed.

### Combined regularisation chart

Each stage/L1 path now combines CV and training deviance on the left axis with
retained coefficient counts on the right, following the Streamlit reference.
CV standard-deviation bars and selected-penalty markers are retained. The shared
log-alpha axis uses at most five compact scientific labels; deviance uses a padded
data range instead of starting at zero. Each combined chart has one expandable
numeric table. Missing CV values omit that series, and fixed-alpha paths retain
the single point. The live two-stage French fit was visually checked at 884 × 773,
with non-overlapping ticks and no horizontal overflow; project, jobs and review
history were unchanged by the static update.
The focused browser parity case covers CV and fixed-alpha single-point paths,
selected markers, non-overlapping scientific ticks and both fitted stages. Svelte
check/build and `git diff --check` passed. Only frontend components and static
assets changed; the server and existing draft tabs were not restarted or reloaded.

### Rate table section grouping

Rate tables now has one Relativities card containing its single chart, visible
adjustment tools, preview impact/actions, history/snapshots and expandable manual
table. A proposal replaces that chart with Current/Proposed values instead of
adding a second chart. Actual versus expected is a separate card underneath, with
its own subset selector, shared preview values, exposure and expandable table.
The explanatory relativity note stays with the relativity chart. Other diagnostics
retain their layout. A live 884 × 773 preview/discard check confirmed both sections
update, only one relativity chart exists, and project, fit and history are unchanged.
The focused table-tools flow passed (all tool modes, manual apply/undo, snapshots
and section geometry), as did the interaction/diagnostic parity case with an
explicit preview-ready wait. Svelte check/build and diff checks passed. This was
a static-only update, without restarting the server or reloading existing tabs.

### Moving-average diagnosis and preview clarity

The live DrivAge three-band calculation was independently reproduced from current
band values and training exposures: every proposed band matched within 1e-12;
endpoints used shorter windows and Other / Unknown was untouched. For 32–34,
current 0.65199094 and its two neighbours give weighted geometric mean 0.66470448,
then the common re-centring factor 0.999735314 gives 0.66452854. The existing flat
upper-age bands remain flat. Step boundaries are retained because scoring remains
a step table; smoothing does not silently convert the factor to a continuous curve.

The chart now explicitly distinguishes the applied table from an unapplied preview.
Tool selection alone does not calculate a preview; unlike the older Streamlit tool
panel, this workflow has an explicit Preview action. Completed previews scroll to
the updated chart instead of past it to the impact panel. Help explains exposure
weighting, geometric averaging, re-centring, current-table input, band-count windows
and endpoint/null handling. No numerical or scoring semantics changed.

### Automatic adjustment previews

Tool selection and valid parameter changes now preview automatically after a
350 ms debounce; the separate tool Preview button is removed. Initial page entry
has no selected tool and starts no adjustment work. Apply remains explicit.
Apply, Discard and navigation disarm pending previews; another deliberate tool
selection or parameter edit starts a new one. A generation guard rejects old
responses and cancels superseded workers; Apply cannot use pending/invalid or
superseded parameters. Manual row drafts block tools and survive preview discard.

The compact controls show applicable parameters, one contextual-help disclosure,
and expected-total impact with Apply/Discard directly underneath. Preview updates
do not scroll or move focus. Relativity and A/E sections retain shared proposal
state. A dedicated browser test delays an old completion, checks latest-only
application after rapid changes, invalid bounds/windows, focus/scroll, no automatic
restart after Apply/Discard/navigation, and manual draft preservation. A live
French preview/discard check confirmed exact project, fit and history preservation.
Svelte check/build and diff checks passed; mathematical/scoring code is unchanged.
Final validation: automatic-preview race/draft case passed (24.3 seconds), and the
existing all-tool/manual/snapshot/undo flow passed (1.4 minutes). A manual preview
is also invalidated if its row draft changes before Apply, preventing an old
preview from clearing a newer draft. Existing tabs and the live server were not
restarted; the updated controls were opened in a fresh tab.

### Diagnostics precision and containment

Diagnostics no longer repeats the metrics/model-facts comparison block. That
block remains on the dedicated Compare page. A shared display formatter limits
numbers to three decimal places, uses compact scientific notation for small
nonzero values, preserves integer counts, normalises negative zero and renders
nonfinite values as a dash. Metrics, totals, chart values/axes/hover, preview values
and diagnostic/rate table displays share the rule. Editable numbers, underlying
arrays and CSV exports retain full precision.

Range labels are compacted for display, retain original identities in tooltips,
and are disambiguated if rounding would merge bands; a range that would falsely
become zero-width uses a unique band label. Numeric axis labels use lower edges
with complete ranges available in tooltips/tables. Width constraints on cards and
tables contain scrolling inside the work area instead of under the sidebar.
Three formatter tests passed. Live read-only checks at 884 × 773 covered metrics,
both regularisation stages, tables/hover, full-precision CSV download, coefficients,
A/E and Compare containment, with exact project/fit/history preservation.
The focused two-model browser parity case passed (38.3 seconds), including block
visibility, three-decimal path values and comparison width assertions. Svelte
check/build and diff checks passed. Updated Diagnostics was opened in a fresh tab;
existing draft tabs and the live server were not reloaded or restarted.

### Automatic pair A/E

Pair A/E recomputes automatically from either variable, relevant temporary bins,
or model/subset/challenger context. The separate Show pair A/E button is removed;
two equal-height dropdowns share a responsive grid. The full request identity
includes both fit identities and context as well as pair/bins; server reviews
already enforce those fit bindings. Debouncing, cancellation and generation guards
prevent superseded results from being displayed. Pending, duplicate and invalid
pairs clear the previous heatmap; duplicate pairs request nothing and show a brief
hint. Hidden pair controls do not launch jobs. Fitted groups ignore temporary bins.

The dedicated browser test passed (25.3 seconds), including a delayed old response,
rapid variable/bin changes, duplicate/invalid clearing, subset/model/challenger
switches, no hidden-tab requests, aligned controls and preserved fit identities.
A read-only live check at 884 × 773 verified the same selection/context behaviour
and exact project/fit/history preservation. Three-decimal display and contained
heatmaps remain in place; no fitting or scoring code changed.
Final checks: the two-model/search/refit/interaction-edit parity case passed
(42.3 seconds). The pair test with a deterministic temporary server refusal and
delayed old result passed (27.5 seconds). Navigation now briefly retries only the
explicit "review is running" refusal while the old view's worker finishes; all
context guards remain active. Svelte check/build and diff checks passed. This was
a static-only update with no server restart or changes to existing draft tabs.

### Focused two-model comparison

Compare now requires two selected, applicable fits and shows selected-subset metrics
with challenger-minus-baseline deltas, changed model settings, both applied base
rates, and the canonical aligned relativity differences. It no longer repeats the
single-model KPI cards, fit banner, diagnostic tabs or saved-version tables.
Response, exposure/scaling, family/link and Tweedie-power mismatches show guidance
instead of comparable metrics. A/E is described in terms of closeness to one;
there is no automatic winner or champion promotion. Numeric band labels use the
shared display precision while CSV data and original hover labels remain exact.

Validation: Svelte check and static build; comparison/format unit tests; existing
two-model parity workflow; focused 884 × 773 browser checks for missing and
incompatible challengers, subset deltas, containment and navigation. The live
single-model session was checked read-only, preserving the exact project, fit jobs
and review history; no live refit or server restart was required.

### Trailing point moving average

The Svelte Moving average now uses an equal-weight arithmetic average of the
current point and preceding N−1 points, with available points at the start.
For example, [1, 1, 1, 4, 4, 4] with window 3 produces [1, 1, 1, 2, 3, 4].
There is no exposure weighting, logarithmic averaging or recentering. Windows
1–25 include even sizes. The existing public log-space moving-average function
is retained separately; the desktop tool calls `smooth_trailing_average`.

Numeric band charts join the band sample values with lines and points, as the
original Streamlit chart did. Scoring still uses constant values within each
band. Linear factors average distinct nodes and retain canonical log slopes;
Null / Other is excluded. Preview starts from current tables and applies only
through the existing explicit adjustment action, without refitting.

Validation covers endpoint arithmetic, future-point causality, exposure
invariance, window 1/even windows, ordered categoricals, unique linear nodes and
slope continuity; desktop preview/money/history tests and automatic-preview
race tests. A live BonusMalus preview independently matched every trailing mean,
rendered nine joined band points with no vertical staircase, and was discarded.
The exact live project, fitted jobs and review history were unchanged.

### Original fit and adjusted slots

Rate charts now retain two permanent series: Original fit and Adjusted. An
active preview replaces only the adjusted series; Discard returns it to the
applied values. A/E uses Actual, Original fit and Adjusted with matching model
colours. Tool previews now start from the original fitted factor, rather than
compounding the current table. Applying replaces eligible rows of that factor's
adjustment overlay; other factors, base-rate overrides, Null / Other manual
adjustments and saved snapshots remain intact. Manual row edits retain other
rows in the adjusted overlay.

Candidate impact and Apply eligibility still compare against the applied book.
Thus an original-equal candidate can remove a prior adjustment. Sequential-tool
checks cover this case, retained other-factor/null/base/snapshot values, exact
original coefficients and predictions, and successive manual edits. Backend
preview/history checks and automatic-preview race checks pass. Read-only live
verification matched the Original fit to the retained fit artifact, checked
window 3 and window 1 previews, and verified stable series during preview,
switching and discard without changing the project, fit artifact or history.

### Fitted-variable A/E cache

Fit completion now writes aggregate A/E rows for ordinary fitted main effects,
using shared full-frame predictions for training, holdout and all rows. The
immutable original cache is separate from the applied-model cache. Keys include
fit/raw artifact identity, preparation and grouping settings, the entire model's
adjustments/base override, and any challenger's fit and adjustment basis.
Challenger predictions are grouped using the selected model's fitted groups.
Original summaries survive adjustment changes; any factor/base change invalidates
all adjusted factor summaries. Undo can reuse an earlier valid cache.

Precomputation excludes unfitted columns and pairs and is bounded to 128 factors,
500 groups per factor and 10,000 total groups; eight adjusted cache versions are
retained. Cache failure falls back to canonical on-demand diagnostics and cannot
fail a successful fit. Tool previews do not precompute all factors. The server
returns cached results without starting a worker, even while unrelated review
work is running. The browser keeps bounded aggregate packets; ordinary variable
and subset switches then make no request. Selection generations prevent a late
uncached response from replacing a newer cached choice.

Measured on the live 50,000-row French motor model at 884 × 773: ordinary variable
switches fell from 1,795–1,800 ms (one cold switch 4,542 ms) to 11–22 ms, with zero
warm review requests. Preparing eight fitted factors took 226 ms; adjusted and
original packets together occupied 166,717 bytes. The standalone measurement
process peaked at 307.56 MiB including interpreter, imports, data and model.
The live fit was warmed read-only. Its existing server was deliberately not
restarted, preserving the session/history: a fresh browser's first load still
costs about 1.87 seconds there. The server cache fast path is active on the next
normal launch and was verified with an isolated server.

Checks cover exact canonical numeric/linear/categorical and probability-link
rows, null/empty groups and all subsets; other-factor/base/challenger invalidation;
original reuse, fit-time readiness and failure fallback; rapid cached switching,
a delayed uncached response, applied-edit invalidation and existing preview races.

### Explicit adjustment application

The adjustment panel now starts with **Choose adjustment…**. Selecting moving
average, isotonic smoothing, cap/floor, rounding or manual rows only edits local
settings. The original and adjusted curves and A/E remain unchanged until Apply.
One Apply calculates and commits the result; there is no tool-preview confirmation.
Manual row changes are applied together in one undo step. Existing persisted
changes are labelled **Saved adjustments** and are retained on entry.

Tool replacements still start from the original fitted factor; manual changes
still overlay the applied table. Other factors, base rate, null/Other rows and
snapshots retain their established behavior. Pending actions are guarded against
double clicks, changed selections, cancellation and failed calculations. Once
commit starts, cancellation is disabled. A successful save followed by a failed
chart refresh reports that the adjustments were saved. Candidate data is parsed
before edit history is mutated.

Validation covers explicit Apply, all tool options, manual batch undo, snapshots,
failed calculations, stale/cancelled results, double clicks and failed refresh
after saving. Read-only live checking found zero tool requests or commits from
changing methods/options or drafting manual rows; chart markup and the complete
project/jobs/edit-history snapshot were unchanged. The live server was not
restarted and the user's fitted model and saved adjustments were preserved.
