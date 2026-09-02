import polars as pl
from easy_glm.core.prepare import prepare_data
from easy_glm.core.blueprint import generate_blueprint

data = {'numeric_col': [1.0, None, 3.0, None, None], 'categorical_col': ['A', None, 'C', None, None]}
df = pl.DataFrame(data)
bp = generate_blueprint(df)
print('Blueprint:', bp)
prepped = prepare_data(modelling_variables=['numeric_col', 'categorical_col'], df=df, formats=bp)
print('Columns:', prepped.columns)
print('Shape:', prepped.shape)
print(prepped)
