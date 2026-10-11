# Modelling conventions

These rules explain how fitted models become rating tables. The public worked
example is [Building a pricing model](../examples/pricing_walkthrough.md).

- Numeric variables use step bands by default. Piecewise-linear and continuous
  effects require an explicit choice. Automatic boundaries use training data.
- Piecewise-linear effects are clamped outside their fitted range. Their fitted
  coefficients are band slopes; regularisation can make individual bands flat.
  Monotonic constraints bound the signs of those slopes.
- A table's reference point must be a band edge. Most-exposure basing chooses the
  most-exposed band. A continuous term has only two clamp edges: it uses the lower
  edge unless the exposure-weighted median lies beyond 60% of the fitted range.
  Rebasing changes the base rate and relativities together, preserving predictions.
- Main effects stay fixed when an interaction is added. Legacy GLM interactions
  fit cell adjustments against the main-effects offset. Each CatBoost interaction
  uses the GLM and every preceding deployed interaction table as its offset.
- Gaussian interactions require a log link. The Gaussian family default is log;
  explicit non-log links cannot be used with interactions. Observed values may
  be negative or zero, but fitted means and table multipliers remain positive.
  Gaussian interaction selection uses weighted squared error on the original
  response scale, including when comparing a proposed table with no interaction.
- Sparse or unseen interaction cells use the neutral adjustment. Support and
  minimum-weight rules determine which cells can receive an estimated adjustment.
- Frequency A/E uses claim counts and exposure consistently. A rate-change model
  uses the logarithm of current premium as an offset. Its base rate is the change
  for the base risk, which need not equal the overall portfolio change.
- Binomial tables contain odds relativities. Scoring converts the combined odds
  to probabilities; it does not multiply probabilities by exposure.
- Model comparisons align numeric factors on their combined band edges and match
  categorical levels by label. Base-rate changes are reported separately.
- Rate amendments show their effect on expected claims. Smoothing may preserve
  the exposure-weighted mean of log relativities without preserving total expected
  claims. Rebalancing the base rate is a separate, explicit action.

## Optional AGLM numeric curves

`aglm` and `aglm_l` are opt-in numeric designs. They are separate from the
default step bands and from EasyGLM's existing `linear`/`continuous` designs.
For an AGLM curve, `n_breaks` and `breaks` control a full grid including both
endpoints. They do not reuse `n_bins` or `knots`, which remain the controls for
step bands and piecewise-linear designs.

- `aglm` is the AGLM O/C basis: raw `x` plus one capped, normalised ramp for
  every adjacent pair in the grid.
- `aglm_l` is the AGLM L basis: raw `x` plus `abs(x - break)` for each interior
  grid point.
- An automatic grid uses the training values only, with R type-7 quantiles.
  The default requests 100 points including the endpoints; ties can leave fewer
  distinct points. A stated `breaks` list replaces that automatic grid.
- `extend` is the default tail behaviour: it leaves the raw value outside the
  grid and so continues the fitted outer slope. `flat` clips first and holds
  the curve level beyond the endpoints.
- Missing values have all finite-basis columns set to zero and use the usual
  optional null indicator. Infinite values are rejected. AGLM curves do not
  support monotone constraints.

The columns deliberately remain the literal, redundant AGLM basis. Unlike the
existing piecewise-linear band-slope rule, AGLM columns use the standard
uniform `P1=1` before a variable's explicit `penalty_weight` multiplier is
applied. The band-rise penalty rule is therefore not silently applied to an
AGLM curve. With `scale_predictors=True`, fitting standardises those columns as
glmnet does; this changes the coefficient scale used for regularisation but
not the retained basis or the scorer.

The R reference is generated from CRAN `aglm` 0.4.1 and glmnet by
[`scripts/generate_aglm_reference.R`](../scripts/generate_aglm_reference.R).
[`tests/fixtures/aglm_0_4_1_reference.json`](../tests/fixtures/aglm_0_4_1_reference.json)
records the matrices, weighted standardisation and fixed-penalty fits. The
tests compare a common penalised objective and use documented prediction
tolerances after strict solver controls. They do **not** claim bit-for-bit
agreement with glmnet's default stopping rule: on the intentionally collinear
O/C basis, that rule can stop at a different point. See
[the numeric-curves example](../examples/numeric_curves.md) for the actuarial
workflow.

### Alternative AGLM basis

`aglm_l` uses absolute-distance columns rather than the capped interval ramps
of `aglm`. Both can represent a continuous piecewise-linear log-relativity,
but regularisation acts on different columns, so their fitted curves can differ.
After running the numeric-curves walkthrough, use the same age points and
inspect the resulting curve and A/E:

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
instead of the step-band `knots` or `n_bins` controls:

```python
from easy_glm.workflow import VariableDesign

driver_age = VariableDesign(
    kind="aglm", breaks=[18, 25, 35, 50, 80], extrapolation="extend"
)
```
