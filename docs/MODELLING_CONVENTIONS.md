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

Detailed encoder, scoring and cache invariants are maintained in
[AGENTS.md](../AGENTS.md) and enforced by the regression tests.
