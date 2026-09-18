# Variables-page binning controls

Status: implemented on `codex/variable-binning`, based on released v0.463 (`ef3b432`). Plan critiqued by GPT-5.6-luna, built by GPT-5.6-sol, independently reviewed by GPT-6-astra. No release or README changes.

## User workflow

Add **Numeric binning** below the Variables table. Rename **Role JSON** to
**Variables JSON**, because it now contains roles, split mapping and binning.

- **Default number of bins**: initially 20, with the current 2–200 range.
- A searchable list of numeric predictors shows each variable's effective setting.
- Each variable offers **Use default**, **Automatic — choose number of bins**,
  or **Custom cuts** (comma-separated numbers). An existing integer-cut strategy
  must also remain visible and editable without conversion to automatic bins.
- Compact read-only summaries appear in the searchable numeric predictor list.
  The main roles table stays compact; the binning panel owns editing.
- Preview one selected variable on demand: exact intervals, row counts, missing
  count and, when available, exposure. Do not scan every column on every keystroke.
- Reuse **Preview changes → Apply changes**. Drafts do not change fits or charts.
- Move the editable numeric default off Model; leave a summary and link to Variables.
  Factor type, predictor selection and interactions remain on Model.

Settings apply across the project, not separately to each model. Show this in the
panel and list the fitted models that Apply will mark stale.

## JSON proposal

Add this sibling section to the existing roles JSON (source column names):

```json
"binning": {
  "default_bins": 20,
  "overrides": {
    "DriverAge": { "method": "quantile", "bins": 10 },
    "VehicleAge": { "method": "cuts", "cuts": [0, 1, 2, 3, 4, 5] }
  }
}
```

Use an explicit method: `quantile` with `bins`, `cuts` with `cuts`, or `integer`.
A count and custom cuts are mutually exclusive. Integer mode may carry an
optional `fallback_bins` value to preserve an existing per-variable quantile
fallback; otherwise it inherits the project count. A null or omitted override
means inherit when a binning section is supplied. UI and backend validation
must use exactly this contract.

Keep `project.design.defaults.n_bins` and `project.design.variables` as the
canonical persisted settings. This editor is a projection of those settings,
not a second independent configuration. Preserve kind, clamps, penalties,
monotonicity and all other fields the binning editor does not manage.

Legacy JSON with no `binning` section must preserve existing settings. For the
new section, define complete replacement of its managed overrides: removing an
entry returns that variable to the project default. Advanced/inactive settings
must remain represented, so editing roles cannot silently erase them. Resolve
source names through the rename mapping atomically, including rename swaps.

## Meaning and validation

- Automatic cuts use the same full, prepared training rows and quantile routine
  as fitting. These are row quantiles, not equal-exposure quantiles.
- The count is requested, not guaranteed: ties can reduce the actual number.
  Show requested and actual counts separately; missing is a separate bucket.
  Report non-finite data separately and use the actual encoder for row assignment;
  do not introduce silent data filtering or numeric-string conversion here.
- For a step factor, cuts `0,1,2,3,4,5` mean `<0`, `[0,1)`, `[1,2)`, `[2,3)`,
  `[3,4)`, `[4,5)`, `>=5`. Exactly-at-cut values enter the band to the right.
  Do not silently turn cuts into category values or discard empty tail bands.
- Reject non-finite, duplicate or unordered custom cuts and invalid bin counts.
  Show a warning for empty bands, constant or entirely missing training data.
- For piecewise-linear factors, cuts are slope-change points, not flat rating
  bands. Validate against the effective clamp range rather than silently dropping
  cuts. Preview must match the actual encoder.
- Continuous factors have no internal cuts. Categorical factors use levels.
  Explain when saved numeric settings are inactive and retain them across type
  changes, rather than deleting unrelated design fields.
- A training split must be valid before computing a data preview. Saving a
  syntactically valid binning draft should not require a fitted model.

## Apply and fit safety

Include binning-only edits in the server's change detection; currently Variables
Apply commits only when its change list is nonempty. Apply roles, renames, types,
split and binning in one validated transaction.

Current desktop fit keys include the whole shared design, so design changes
conservatively invalidate all fitted models. Preserve that safe behaviour and
make it explicit in preview; selective cache invalidation is outside this pass.
Do not refit automatically. Warn about adjustments and snapshots tied to old
bands; do not carry incompatible values onto new boundaries or silently delete
saved work. Confirm this lifecycle against the existing refit handling before
coding it.

## Delivery and checks

1. Plan and critique: parent agent drafts; lower-tier GPT-5.6-luna critiques.
2. After user review, GPT-5.6-sol implements in an isolated branch.
3. GPT-6-astra independently checks the diff, statistical semantics and failure
   cases. Parent reconciles the reviewer findings with the builder.
4. Browser review: edit defaults and custom cuts, switch table/JSON, Apply,
   reload, fit, inspect rate-table boundaries and export/reopen the project.
5. Tests cover legacy JSON, rename/type changes, numeric strings, custom
   boundaries including zero/negatives, missing values, tied quantiles, linear
   clamps, stale fits, adjustments, and a wide dataset without eager previews.
   Existing projects must fit identically until settings are explicitly changed.
6. Show the user the working preview. No release or README changes in this task.

## Independent critique incorporated

GPT-5.6-luna supported the bounded proposal and emphasised explicit JSON methods,
legacy round-trip safety, atomic renames, training-only previews using fitting
logic, linear clamp validation, and stale-fit/adjustment handling. These are
acceptance requirements above. The new UI targets the desktop workbench; shared
schema changes also need Streamlit regression coverage. Do not promise a precise
refit list until the actual cache keys used by each frontend have been checked.

## Implementation and validation record

- Uses the existing canonical design fields; no core encoder or fitting semantics
  changed. Newly entered linear cuts are checked against clamps at Apply. Existing
  imported designs retain their prior behavior until edited.
- Empty cuts remain valid for linear single-slope and inactive settings; newly
  entered empty cuts for active step factors are rejected.
- Invalid text and method choices stay together in the local draft. Correcting
  a field commits the complete valid draft. Late previews and unchanged model
  polling cannot erase or replace a newer Variables review.
- Searchable list is virtualized. A 2,000-predictor browser check confirms no
  eager bin-preview requests while editing; existing wide-table timings passed.
- Dedicated browser acceptance passes: cut boundaries, row counts, JSON/table
  sync, invalid-draft recovery, reload, fit, rate tables, stale results, export
  and reopen, late response rejection, and a 799-pixel viewport.
- Existing Variables and model browser suites pass, plus 14 JavaScript unit tests,
  Svelte checks, production build, Python formatting/lint and core/workflow types.
- Full Python run: 1,336 passed, one skipped and one deselected; it exposed one
  split-validation regression. Restoring the original mandatory split check was
  then verified by all 56 affected split/binning/desktop API/model tests. The
  separate Variables/refit/review/export regression selection passed 56 tests.
- Final independent review reports no remaining blockers, including snapshot
  restoration safety and all-missing custom-cut intervals.
