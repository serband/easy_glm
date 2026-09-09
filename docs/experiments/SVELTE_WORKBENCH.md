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

Rate-table layout review: the editable grid now appears directly after variable
selection, before adjustment controls and A/E. At 884 × 773 the first bands and
row-preview action are visible without scrolling. A separate fitted/current
relativity chart uses canonical row values, log slopes for linear bands, and
flat clamp bands; interaction tables use labelled cells. Training exposure is
shown under the relativity chart and selected-subset exposure under A/E. Charts
cover the displayed table page. Default columns omit redundant boundary fields;
All columns reveals them. Existing row preview/apply/undo semantics are unchanged.
Navigation actions omit the current page, and selecting a diagnostic variable
refreshes A/E without an additional redundant button.
