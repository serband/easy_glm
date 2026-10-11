# Choosing model families with real motorcycle experience

This page uses the built-in Swedish motorcycle loader, which downloads once and
then uses its local cache. It shows three distinct outcomes, each in a separate
`PricingSession`: total claim cost per policy-year, claim severity conditional on
a paid claim, and claim occurrence probability. Do not compare their A/E or
deviance values as if they measured the same outcome. Fit and inspection below
use training data first; holdout is requested explicitly afterwards. See
[workflows.md](workflows.md) for the workflow hub.

## Load and inspect the portfolio

`ClaimAmount` is total payment, `ClaimNb` is claim count and `Exposure` is
policy-years. Some rows have zero exposure, so remove them before modelling a
policy-year outcome.

```python
import polars as pl

from easy_glm import PricingSession, load_swedish_motorcycle_data

motorcycle = load_swedish_motorcycle_data()
positive_exposure = motorcycle.filter(pl.col("Exposure") > 0)
positive_exposure.select("Exposure", "ClaimNb", "ClaimAmount").head()
```

## Tweedie pure premium

Tweedie is appropriate for a non-negative aggregate cost with a mass at zero and
a positive right tail. The target is total policy payment (`ClaimAmount`), and
`Exposure` is the weight. `divide_target_by_weight=True` tells EasyGLM to fit a
per-policy-year response, while expected values convert back to total payment.
`ClaimNb` is ignored so count is not used as a predictor of its own aggregate cost.

```python
pure_premium_session = PricingSession(
    positive_exposure,
    family="tweedie",
    tweedie_power=1.5,
    target="ClaimAmount",
    weight="Exposure",
    divide_target_by_weight=True,
    train_fraction=0.70,
    seed=17,
    ignored=["ClaimNb"],
)
pure_premium = pure_premium_session.fit_glm(
    "Motorcycle pure premium",
    factors=["OwnerAge", "Gender", "Area", "RiskClass", "VehAge", "BonusClass"],
    cv=3,
    n_alphas=8,
)
pure_premium.summary()
pure_premium.ae("RiskClass").show()
```

The training A/E uses total observed and expected payments. It assesses aggregate
calibration; review deviance and factor support alongside it.

```python
pure_premium.validate_holdout()
pure_premium.ae("RiskClass", subset="holdout").show()
pure_premium.to_excel("swedish_tweedie_pure_premium.xlsx")
pure_premium.save("swedish_tweedie_pure_premium.easyglm")
```

## Gamma severity, conditional on a claim

Severity uses a different population: rows with `ClaimNb > 0` and a positive
payment. This is a conditional-paid-claim example, so claim-free and zero-payment
claim rows are excluded. The target remains total payment, but `ClaimNb` is now
the weight and the fitted unit is payment **per claim**. Exposure is irrelevant
to this conditional outcome and is ignored. This is not a pure-premium model.

```python
severity_data = positive_exposure.filter(
    (pl.col("ClaimNb") > 0) & (pl.col("ClaimAmount") > 0)
)
severity_session = PricingSession(
    severity_data,
    family="gamma",
    target="ClaimAmount",
    weight="ClaimNb",
    divide_target_by_weight=True,
    train_fraction=0.70,
    seed=17,
    ignored=["Exposure"],
)
severity = severity_session.fit_glm(
    "Motorcycle claim severity",
    factors=["OwnerAge", "Gender", "Area", "RiskClass", "VehAge", "BonusClass"],
    cv=3,
    n_alphas=8,
)
severity.summary()
severity.ae("RiskClass").show()
```

Here A/E is total observed payment divided by total expected payment on claim
records, with claim count as the weight. It does not include claim-free policies.

```python
severity.validate_holdout()
severity.ae("RiskClass", subset="holdout").show()
severity.to_excel("swedish_gamma_severity.xlsx")
severity.save("swedish_gamma_severity.easyglm")
```

## Binomial claim occurrence

For occurrence, create an explicit 0/1 target. There is no exposure multiplication:
the fitted quantity is a probability per observed policy record. Ignore the count,
aggregate amount and exposure so none of those outcome measures can leak into the
predictors.

```python
occurrence_data = positive_exposure.with_columns(
    (pl.col("ClaimNb") > 0).cast(pl.Float64).alias("AnyClaim")
)
occurrence_session = PricingSession(
    occurrence_data,
    family="binomial",
    target="AnyClaim",
    train_fraction=0.70,
    seed=17,
    ignored=["ClaimNb", "ClaimAmount", "Exposure"],
)
occurrence = occurrence_session.fit_glm(
    "Motorcycle claim occurrence",
    factors=["OwnerAge", "Gender", "Area", "RiskClass", "VehAge", "BonusClass"],
    cv=3,
    n_alphas=8,
)
occurrence.summary()
occurrence.ae("RiskClass").show()
```

For binomial work, inspect calibration and deviance as well as the A/E ratio.
The expected values are expected numbers of records with at least one claim in
the selected rows. They are not claim counts, payment amounts, or annualised
frequency.

```python
occurrence.validate_holdout()
occurrence.ae("RiskClass", subset="holdout").show()
occurrence.to_excel("swedish_binomial_occurrence.xlsx")
occurrence.save("swedish_binomial_occurrence.easyglm")
```
