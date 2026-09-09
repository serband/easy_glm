# Svelte workbench experiment

Branch: `codex/svelte-workbench`, based on main `0828b89` (0.452).
This is a runnable Variables slice, not a replacement for the full workbench.
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
pool and releases the project lock before computation. There is **no fitting
endpoint** yet. Long fits in the next slice must use a separate worker process,
with immutable project revisions, progress/cancellation and stale-result checks;
never run fits inside a request handler or block the UI loop.

## Acceptance goals and measured results

These targets are goals for this first slice, not promises for corporate laptops:

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

## Next slices, in order

1. Review this Variables interaction on the actual Windows/Positron laptop; collect
   readiness and edit timings plus any browser or server errors. Investigate the
   existing network problem separately using its logs.
2. Design and model selection, preserving independent roles versus model terms;
   virtualise both axes when a raw-data or interaction matrix grid is introduced.
3. Process-based fitting jobs: polling/progress, cancellation, revision-bound
   results, and a proven idle UI while a long fit runs. Reuse `workflow.run_model`.
4. Diagnostics and rate tables with current engine tooling and undo/snapshots;
   test that main and two-stage interaction invariants survive the new UI.
5. Explicit durable workspace design, conflict policy, Save / Close / Resume and
   packaging/dependency review. Only then consider replacing Streamlit defaults.

Do not infer completion of later slices from the sidebar placeholders.
