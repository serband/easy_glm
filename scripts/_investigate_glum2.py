"""Investigate why GLUM/tabmat ignores the categorical column."""
import numpy as np
import pandas as pd
import polars as pl
from glum import GeneralizedLinearRegressor
from easy_glm.core.prepare import prepare_data

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

pdf = prepped.to_pandas()
print('=== Dtypes before conversion ===')
print(pdf.dtypes)
print()

# The Region_lumped column
col = pdf['Region_lumped']
print(f'Region_lumped dtype: {col.dtype}')
print(f'Region_lumped values: {col.unique()}')
print()

# Try different conversion approaches
print('=== Approach 1: astype("category") ===')
pdf1 = pdf.copy()
pdf1['Region_lumped'] = pdf1['Region_lumped'].astype('category')
print(f'dtype: {pdf1["Region_lumped"].dtype}')
print(f'cat categories: {pdf1["Region_lumped"].cat.categories.tolist()}')
features = [c for c in pdf1.columns if c not in ('ClaimNb', 'Exposure', 'traintest')]
X1 = pdf1[features]
try:
    m = GeneralizedLinearRegressor(family='poisson', fit_intercept=True, alpha_search=True)
    m.fit(X1, pdf1['ClaimNb'].values)
    print(f'feature_names_: {m.feature_names_}')
    print(f'coef_: {m.coef_}')
    print(f'categorical_levels_: {m.categorical_levels_}')
except Exception as e:
    print(f'ERROR: {e}')

print()
print('=== Approach 2: pd.Categorical directly ===')
pdf2 = pdf.copy()
pdf2['Region_lumped'] = pd.Categorical(pdf2['Region_lumped'])
print(f'dtype: {pdf2["Region_lumped"].dtype}')
features2 = [c for c in pdf2.columns if c not in ('ClaimNb', 'Exposure', 'traintest')]
X2 = pdf2[features2]
try:
    m2 = GeneralizedLinearRegressor(family='poisson', fit_intercept=True, alpha_search=True)
    m2.fit(X2, pdf2['ClaimNb'].values)
    print(f'feature_names_: {m2.feature_names_}')
    print(f'coef_: {m2.coef_}')
    print(f'categorical_levels_: {m2.categorical_levels_}')
except Exception as e:
    print(f'ERROR: {e}')

print()
print('=== Approach 3: Object dtype (string) ===')
pdf3 = pdf.copy()
print(f'dtype: {pdf3["Region_lumped"].dtype}')
features3 = [c for c in pdf3.columns if c not in ('ClaimNb', 'Exposure', 'traintest')]
X3 = pdf3[features3]
try:
    m3 = GeneralizedLinearRegressor(family='poisson', fit_intercept=True, alpha_search=True)
    m3.fit(X3, pdf3['ClaimNb'].values)
    print(f'feature_names_: {m3.feature_names_}')
    print(f'coef_: {m3.coef_}')
    print(f'categorical_levels_: {m3.categorical_levels_}')
except Exception as e:
    print(f'ERROR: {e}')
