# Choosing model families with real motorcycle experience

This page uses the built-in Swedish motorcycle data. It builds three different
models from the same source: pure premium per policy-year, severity per paid
claim, and the probability of at least one claim. They have different targets
and populations, so their A/E and deviance figures are not competing measures.
Each model is fitted on training rows first, then checked on holdout.

## Load and inspect the portfolio

`ClaimAmount` is total paid claims, `ClaimNb` is the claim count, and `Exposure`
is policy-years. A per-policy-year model cannot use a row with zero exposure, so
remove those rows first.

```python
import polars as pl

from easy_glm import PricingSession, load_swedish_motorcycle_data

motorcycle = load_swedish_motorcycle_data()
positive_exposure = motorcycle.filter(pl.col("Exposure") > 0)
positive_exposure.select("Exposure", "ClaimNb", "ClaimAmount").head()
```

## Tweedie pure premium

Use Tweedie for total claim cost when many policies have no cost and the positive
amounts are skewed. The target is total payment (`ClaimAmount`).
`divide_target_by_weight=True` tells EasyGLM to fit cost per policy-year using
`Exposure`, then turn predictions back into total expected cost for A/E. We mark
`ClaimNb` as ignored so the claim count cannot sneak into the predictors.

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

The A/E shown here is total observed payment divided by total expected payment on
the training rows. Look at the supporting exposure and deviance as well before
changing a factor.

```python
pure_premium.validate_holdout()
pure_premium.ae("RiskClass", subset="holdout").show()
pure_premium.to_excel("swedish_tweedie_pure_premium.xlsx")
pure_premium.save("swedish_tweedie_pure_premium.easyglm")
```

## Gamma severity, conditional on a claim

Severity answers a different question: how large is payment once there has been
a paid claim? Keep only rows with a positive claim count and payment. The target
is still total payment, but `ClaimNb` is now the weight, making the fitted unit
payment per claim. Exposure is not part of this conditional model.

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

Here A/E uses paid-claim rows only and weights them by claim count. It says
nothing about the claim-free policies left out of the severity population.

```python
severity.validate_holdout()
severity.ae("RiskClass", subset="holdout").show()
severity.to_excel("swedish_gamma_severity.xlsx")
severity.save("swedish_gamma_severity.easyglm")
```

## Binomial claim occurrence

For occurrence, turn the count into a 0/1 question: did this record have at least
one claim? The model returns a probability for a policy record, so it does not use
exposure multiplication. We explicitly ignore count, payment and exposure to keep
outcome measures out of the predictor list.

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

For binomial work, read calibration and deviance alongside A/E. Expected values
here mean expected records with at least one claim. They are not claim counts,
payment amounts or annualised frequency.

```python
occurrence.validate_holdout()
occurrence.ae("RiskClass", subset="holdout").show()
occurrence.to_excel("swedish_binomial_occurrence.xlsx")
occurrence.save("swedish_binomial_occurrence.easyglm")
```
