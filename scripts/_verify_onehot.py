"""Verify: manual matrix multiply (intercept + X@coef) == GLUM model.predict(X)."""
import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor
from easy_glm.core.prepare import prepare_data
from easy_glm.core.blueprint import generate_blueprint

rng = np.random.default_rng(42)
n = 500

df = pl.DataFrame({
    'VehAge': rng.integers(0, 15, n).astype(float),
    'DrivAge': rng.integers(18, 80, n).astype(float),
    'Region': rng.choice(['North', 'South', 'Urban', 'Rural'], n,
                         p=[0.35, 0.35, 0.20, 0.10]),
    'ClaimNb': rng.poisson(0.5, n).astype(float),
    'Exposure': np.ones(n),
    'traintest': np.ones(n, dtype=np.int8),
})

predictors = ['VehAge', 'DrivAge', 'Region']
blueprint = generate_blueprint(df)
print('Blueprint:')
for k, v in blueprint.items():
    print(f'  {k}: {v}')

prepped = prepare_data(
    df=df, modelling_variables=predictors,
    additional_columns=['ClaimNb', 'Exposure', 'traintest'],
    formats=blueprint, table_name='test',
)
print(f'\nPrepared columns ({len(prepped.columns)}):')
print(prepped.columns)

# Fit GLUM
pdf = prepped.to_pandas()
features = [c for c in pdf.columns if c not in ('ClaimNb', 'Exposure', 'traintest')]
X = pdf[features].values.astype(float)
y = pdf['ClaimNb'].values
w = pdf['Exposure'].values

model = GeneralizedLinearRegressor(
    family='poisson', fit_intercept=True, alpha_search=True,
)
model.fit(pdf[features], y, sample_weight=w)

print(f'\nIntercept: {model.intercept_:.6f}')
print(f'Coefficients ({len(model.coef_)}):')
for name, coef in zip(model.feature_names_, model.coef_):
    print(f'  {name:30s}  {coef:+.8f}')

# --- Test 1: model.predict vs manual ---
pred_glum = model.predict(pdf[features])
lp_manual = model.intercept_ + X @ model.coef_
pred_manual = np.exp(lp_manual)

print(f'\n=== Test 1: model.predict vs exp(intercept + X@coef) ===')
print(f'Max absolute difference: {np.max(np.abs(pred_glum - pred_manual)):.2e}')
match = np.allclose(pred_glum, pred_manual)
print(f'Match: {match}')
if not match:
    raise AssertionError('Manual predictions do not match GLUM!')

# --- Test 2: Reference level has all zeros ---
print(f'\n=== Test 2: Reference level check ===')
ref_level = blueprint['Region'][0]
print(f'Reference level (first in blueprint): {ref_level}')
region_col = df['Region']
ref_mask = region_col == ref_level
ref_indices = np.where(ref_mask.to_numpy())[0]
if len(ref_indices) > 0:
    region_onehot_cols = [c for c in features if c.startswith('Region_')]
    ref_row = pdf.iloc[ref_indices[0]][region_onehot_cols]
    print(f'  One-hot row for Region={ref_level}: {ref_row.to_dict()}')
    assert (ref_row == 0).all(), f'Reference level should have all zeros'
    print('  ✓ Reference level correctly has all zeros')

# --- Test 3: Non-reference level fires correct column ---
print(f'\n=== Test 3: Non-reference level check ===')
for level in blueprint['Region'][1:]:
    mask = region_col == level
    indices = np.where(mask.to_numpy())[0]
    if len(indices) > 0:
        region_onehot_cols = [c for c in features if c.startswith('Region_')]
        row = pdf.iloc[indices[0]][region_onehot_cols]
        expected_col = f'Region_{level}'
        if expected_col in row.index:
            assert row[expected_col] == 1, f'Expected {expected_col}=1'
            others = [c for c in row.index if c != expected_col]
            assert (row[others] == 0).all(), f'Other columns should be 0'
            print(f'  Region={level}: ✓ {expected_col}=1, others=0')

# --- Test 4: Other column fires for unseen level ---
print(f'\n=== Test 4: Other catch-all for unseen level ===')
test_df = pl.DataFrame({
    'VehAge': [5.0], 'DrivAge': [35.0], 'Region': ['Mars'],
})
test_prepped = prepare_data(
    df=test_df, modelling_variables=predictors,
    formats=blueprint, table_name='test2',
)
test_pdf = test_prepped.to_pandas()
region_cols_test = [c for c in test_pdf.columns if c.startswith('Region_')]
print(f'  One-hot for Region=Mars: {test_pdf[region_cols_test].iloc[0].to_dict()}')
other_col = [c for c in region_cols_test if c.endswith('_Other')]
assert len(other_col) == 1
assert test_pdf[other_col[0]].iloc[0] == 1
known_cols = [c for c in region_cols_test if c != other_col[0]]
assert (test_pdf[known_cols].iloc[0] == 0).all()
print('  ✓ Other column fires for unseen level')

# --- Test 5: Other column fires for NULL ---
print(f'\n=== Test 5: Other catch-all for NULL ===')
null_df = pl.DataFrame({
    'VehAge': [5.0], 'DrivAge': [35.0], 'Region': [None],
})
null_prepped = prepare_data(
    df=null_df, modelling_variables=predictors,
    formats=blueprint, table_name='test3',
)
null_pdf = null_prepped.to_pandas()
print(f'  One-hot for Region=None: {null_pdf[region_cols_test].iloc[0].to_dict()}')
assert null_pdf[other_col[0]].iloc[0] == 1
print('  ✓ Other column fires for NULL')

print('\n✓ All verification tests passed!')
