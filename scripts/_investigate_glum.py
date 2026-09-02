"""One-off: understand GLUM feature_names_ & scale_predictors for coefficient mapping."""
import numpy as np
import pandas as pd
import polars as pl
from glum import GeneralizedLinearRegressor
from easy_glm.core.prepare import prepare_data

# --- Small test: one numeric (o-matrix), one categorical (lumped) ---
n = 100
rng = np.random.default_rng(42)
df = pl.DataFrame({
    'VehAge': rng.integers(0, 10, n).astype(float),
    'Region': rng.choice(['North', 'South', 'Urban'], n),
    'ClaimNb': rng.poisson(0.5, n).astype(float),
    'Exposure': np.ones(n),
    'traintest': np.ones(n, dtype=np.int8),
})

blueprint = {'VehAge': [2.0, 5.0, 8.0], 'Region': ['North', 'South', 'Urban']}

prepped = prepare_data(
    df=df, modelling_variables=['VehAge', 'Region'],
    additional_columns=['ClaimNb', 'Exposure', 'traintest'],
    formats=blueprint, table_name='test', traintest_column='traintest',
)

print('=== Prepared columns ===')
print(prepped.columns)
print()

pdf = prepped.to_pandas()
text_cols = [c for c in pdf.columns if pdf[c].dtype == 'object']
for c in text_cols:
    pdf[c] = pdf[c].astype('category')

features = [c for c in pdf.columns if c not in ('ClaimNb', 'Exposure', 'traintest')]
X = pdf[features]
y = pdf['ClaimNb'].values

print('=== With scale_predictors=True ===')
m1 = GeneralizedLinearRegressor(
    family='poisson', fit_intercept=True,
    scale_predictors=True, alpha_search=True,
)
m1.fit(X, y, sample_weight=pdf['Exposure'].values)
print(f'intercept_: {m1.intercept_:.6f}')
for name, coef in zip(m1.feature_names_, m1.coef_):
    print(f'  {name:35s}  {coef:+.8f}')
print(f'categorical_levels_: {m1.categorical_levels_}')
print()

print('=== With scale_predictors=False ===')
m2 = GeneralizedLinearRegressor(
    family='poisson', fit_intercept=True,
    scale_predictors=False, alpha_search=True,
)
m2.fit(X, y, sample_weight=pdf['Exposure'].values)
print(f'intercept_: {m2.intercept_:.6f}')
for name, coef in zip(m2.feature_names_, m2.coef_):
    print(f'  {name:35s}  {coef:+.8f}')
print(f'categorical_levels_: {m2.categorical_levels_}')
print()

# --- Verify: predictions match between model.predict and manual computation ---
print('=== Manual prediction check ===')
row = X.iloc[0:1].copy()
print('Feature values:', row.to_dict('records')[0])

pred_model = m1.predict(row)[0]
print(f'model.predict:       {pred_model:.8f}')

# Manual: linear_predictor = intercept + sum(coef_i * x_i)
lp = m1.intercept_
for name, coef in zip(m1.feature_names_, m1.coef_):
    val = row[name].values[0]
    lp += coef * float(val)
    print(f'  {name} = {float(val):.4f} * {coef:.6f} = {coef * float(val):.8f}')
pred_manual = np.exp(lp)
print(f'Manual exp(lp):      {pred_manual:.8f}')
print(f'Match: {np.isclose(pred_model, pred_manual)}')
