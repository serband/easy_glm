# AGLM numeric methods

Status: implemented and locally verified on `codex/aglm-numeric-methods`.
Unreleased.

## Scope

Add the numeric methods from R `aglm` 0.4.1 as explicit choices. Keep EasyGLM's
existing step, piecewise-linear and continuous methods, defaults and fitted
models unchanged. Fix the confirmed rare-value binning and small-range rounding
bugs separately.

The reference is the installed R package, rather than a description from another
LLM. `scripts/generate_aglm_reference.R` records its version and generates
`tests/fixtures/aglm_0_4_1_reference.json`. CI can check those results without R.

## Numeric choices

| Choice | Basis | Outside the break grid |
| --- | --- | --- |
| Existing step | Threshold indicators | Existing end bands |
| Existing linear/continuous | Amount within each band; coefficients are slopes | Existing flat tails |
| `aglm` | Original numeric value plus a capped, normalised ramp for every interval | Extend by default; optional flat tails |
| `aglm_l` | Original numeric value plus absolute distance from each interior break | Extend by default; optional flat tails |

AGLM's full grid includes both endpoints. Automatic grids use evenly spaced
quantile probabilities and interpolated **training** quantiles, with 100 points
by default, capped by the number of training rows. Duplicate points are removed.
These points are separate from the existing number of step bands. User-supplied
grids must contain at least two finite, strictly increasing values.

Keep every reference column, including AGLM's deliberately redundant columns.
Apply ordinary per-column regularisation, multiplied by any explicit variable
penalty weight. Do not apply the existing band-rise penalty to AGLM columns.
Compare objective functions and predictions against glmnet before claiming fit
parity; equal numeric alpha/lambda values alone are insufficient evidence.

Flat tails clip the original numeric value as well as the ramp/absolute-distance
columns. The default missing-value extension uses zero numeric columns and a
separate null indicator. Without that indicator, missing values are rejected.
Infinities are rejected. R parity fixtures use finite data without the extra
null indicator. AGLM monotone constraints are initially rejected because bounds
on individual basis coefficients do not constrain the combined curve correctly.

## Public controls and persistence

`EasyGLM.fit` and `DesignSpec.from_data` gain `numeric_kinds`, `breaks`,
`n_breaks` and `extrapolation`. A per-variable `VariableDesign` gains `breaks`,
`n_breaks` and `extrapolation`; the workbench offers the same settings. Existing
JSON can omit these additions. Frozen legacy stack recipes remain valid without
being rewritten or changing their identity.

The interactive pricing API gains a single numeric-method setter. Setting a
method or grid affects subsequent fits and leaves earlier named fits intact.
The workbench explains the grid and tails beside their controls.

AGLM fitted curves become exact piecewise-linear rating tables. Persist the tail
policy explicitly; old tables default to flat. Scoring, edits, JSON, save/reopen,
Excel and Python exports must retain the same curve, including outside the grid.
Scoring remains chunked and does not construct a design matrix. Aggregation keys
include actual numeric values, rather than only their interval codes.

## Verification and reviews

1. Verify both bases and automatic grids against R, including uneven grids,
   tied/skewed data, break boundaries and both tail policies.
2. Verify weighted fits with offsets across Poisson, Gaussian and binomial;
   include lasso and elastic net. Compare predictions and the common objective
   rather than requiring identical coefficients in a redundant basis.
3. Verify dense/sparse fitting, exact rating-table scoring, persistence, edits,
   Excel and exported Python on values inside and outside the grid.
4. Exercise Project/stack and PricingSession with training-only grids, independent
   named models, A/E, offsets and interaction tables. Run the worked example
   exactly as written.
5. Run independent software, adversarial and actuarial reviews. Inspect the
   rendered workbench at wide and narrow widths. Close findings before reporting
   completion.
6. Run relevant existing regressions, the full Python suite, frontend checks,
   Ruff, Black and mypy. Build a wheel and execute the public example in a fresh
   environment. No release or publication is part of this work.

The core builder owns encoders, fitting, rating tables and engine persistence.
The frontend builder owns the variable editor and its validation. The primary
agent owns workflow/pricing/export integration and this plan. A separate agent
owns the R fixtures and parity checks; independent agents review the result.

## Completed checks — 11 October 2026

- The R fixture checks both bases, automatic grids and tails against `aglm`
  0.4.1. Six weighted, offset fits cover Poisson, Gaussian and binomial, with
  lasso and elastic net. Comparisons use a common penalised objective and explicit
  convergence controls. Default glmnet stopping can still give different fits.
- The full Python suite passed 1,750 tests, with one skipped and the separate
  scale test deselected. After the final missing-value correction, all 105
  focused AGLM, integration and legacy linear/design tests passed again.
- Black, Ruff and mypy passed. Svelte checking reported no errors or warnings;
  all 32 frontend unit tests and all 11 interaction browser cases passed.
- Independent browser review exercised AGLM settings, coarser pair-only bands,
  grid previews, fit/keep and unchanged main curves. Regression testing also
  exposed and repaired an existing Continue-to-next-model navigation defect:
  configured models now open fitting; incomplete models open variable setup.
- A fresh installed wheel matches all 116 source Python/static files. All ten
  literal Python blocks in `examples/numeric_curves.md` executed against it;
  export/reopen preserved predictions to `1e-12`, and the disabled-null policy
  was checked separately. The final bundled JavaScript is `index-BMIWjr-T.js`.
- Independent software/adversarial and actuarial reviews accepted the result.

These are local checks. Windows, remote CI and the 5M-row scale test were not
run. No release, merge or publication is included.

## Example settings

For an interactive worked example of bands, an AGLM curve, an interaction and
rate edits, use [Numeric curves](../../examples/numeric_curves.md).

### Alternative AGLM basis

`aglm_l` uses absolute-distance columns rather than the capped interval ramps
of `aglm`. Both can represent a continuous piecewise-linear log-relativity, but
regularisation acts on different columns, so their fitted curves can differ.
To compare this method after running the numeric-curve walkthrough, use the
same age points and inspect the resulting curve and A/E:

```python
work.numeric(
    "DrivAge", kind="aglm_l", breaks=[18, 25, 35, 50, 80], extrapolation="flat"
)
l_curve = fixed.refit("Alternative driver-age curve")
l_curve.ae("DrivAge").show()
l_curve.compare(fixed)
```

### Project variable definition

When editing a Project recipe, assign a `VariableDesign` to
`project.design.variables["DrivAge"]` before fitting. This definition supplies
the method, complete age grid and extrapolation policy. Use curve `breaks`
instead of the step-band `knots`/`n_bins` controls:

```python
from easy_glm.workflow import VariableDesign

driver_age = VariableDesign(
    kind="aglm", breaks=[18, 25, 35, 50, 80], extrapolation="extend"
)
```
