"""Code to test the function"""


import easy_glm 
import polars as pl
import numpy as np

df = easy_glm.load_external_dataframe()

# create a traintest column in the dataframe
df = (df.with_columns(
    pl.when(pl.lit(np.random.rand(df.height) < 0.7))
    .then(1)
    .otherwise(0)
    .alias("traintest")))



# generate the blueprint for the cars dataset 
d = easy_glm.generate_blueprint(df)

# Predictor variables
predictor_variables = ['VehAge','Region','VehGas','DrivAge','BonusMalus','Density']


# Prepped the dataset for modelling
prepped = easy_glm.prepare_data(
    df = df, 
    modelling_variables=predictor_variables, 
    additional_columns=['Exposure','ClaimNb'], 
    formats=d, 
    traintest_column='traintest', 
    table_name='cars')

# Fit model 
model = easy_glm.fit_lasso_glm(
    dataframe = prepped, 
    target = "ClaimNb", 
    model_type = "Poisson", 
    weight_col="Exposure", 
    train_test_col = "traintest",
    DivideTargetByWeight=True
    )






tbl = easy_glm.ratetable(
    model=model,
    dataset=df,               # your Polars df
    col_name="VehAge",
    levels=d['VehAge'],
    prepare=lambda df: easy_glm.prepare_data(
        df=df,
        modelling_variables=predictor_variables,
        formats=d,
        table_name="line_prepped",
    ),
    random_seed=42,
)


# Assuming 'model', 'df', 'predictor_variables', and 'd' are pre-defined
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictor_variables,
    blueprint=d
)
print(all_tables['VehAge'])  # Access the rate table for 'VehAge'


# Plot all ratetables
easy_glm.plot_all_ratetables(all_tables, d)



tbl.write_csv("/mnt/extra/ratetable2.csv")

# Predict with model
predictions = easy_glm.predict_with_model(model, new_data=prepped)

from typing import List, Optional, Union
import duckdb
import polars as pl
import pandas as pd
import numpy as np
import random
import pyarrow
from glum import GeneralizedLinearRegressorCV
from glum import GeneralizedLinearRegressor
from glum import TweedieDistribution
import polars as pl






def o_matrix(col_name: str, brks) -> List[str]:
    """
    Create SQL CASE statements for generating indicator columns based on threshold values.
    
    This function generates SQL expressions that create binary indicator columns for each
    threshold value in brks. For each threshold, the resulting column will be 1 if the
    original value is less than the threshold, and 0 otherwise.
    
    Args:
        col_name (str): The name of the column to create indicators for.
        brks (List[Union[int, float]] or np.ndarray): A list or array of numeric threshold 
            values to create indicator columns for. Each value will generate one indicator column.
    
    Returns:
        List[str]: A list of SQL CASE statement strings, one for each threshold value.
            Each string creates an indicator column named '{col_name}{threshold_value}'.
    
    Raises:
        TypeError: If col_name is not a string or brks is not a list/array of numbers.
        ValueError: If col_name is empty, brks is empty, or contains non-numeric values.
        
    Example:
        >>> col_name = "driver_age"
        >>> thresholds = [20, 30, 40]
        >>> sql_statements = o_matrix(col_name, thresholds)
        >>> print(sql_statements[0])
        CASE WHEN driver_age IS NULL THEN CASE WHEN AVG(driver_age) OVER () < 20 THEN 1 ELSE 0 END ELSE CASE WHEN driver_age < 20 THEN 1 ELSE 0 END END AS 'driver_age20'
        
        For a driver aged 25, this would create indicator columns:
        - driver_age20: 0 (25 >= 20)
        - driver_age30: 1 (25 < 30) 
        - driver_age40: 1 (25 < 40)
    
    Note:
        - NULL values are handled by comparing the column's average to the threshold
        - The generated column names follow the pattern: '{col_name}{threshold_value}'
        - Each indicator column represents: 1 if value < threshold, 0 otherwise
    """
    # Input validation
    if not isinstance(col_name, str):
        raise TypeError(f"col_name must be a string, got {type(col_name)}")
    
    if not col_name.strip():
        raise ValueError("col_name cannot be empty or whitespace only")
    
    # Convert numpy array to list if needed
    if isinstance(brks, np.ndarray):
        brks = brks.tolist()
    
    if not isinstance(brks, list):
        raise TypeError(f"brks must be a list or numpy array, got {type(brks)}")
    
    if len(brks) == 0:
        raise ValueError("brks cannot be empty")
    
    # Validate that all break values are numeric
    for i, val in enumerate(brks):
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"All values in brks must be numeric. Value at index {i} is {type(val)}: {val}")
        
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Break values cannot be NaN or infinite. Value at index {i}: {val}")
    
    # Clean column name for SQL (remove special characters that could cause issues)
    clean_col_name = col_name.strip()
    
    # Generate SQL CASE statements
    sql_statements = []
    for val in brks:
        sql_statement = (
            f"CASE WHEN {clean_col_name} IS NULL "
            f"THEN CASE WHEN AVG({clean_col_name}) OVER () < {val} THEN 1 ELSE 0 END "
            f"ELSE CASE WHEN {clean_col_name} < {val} THEN 1 ELSE 0 END END "
            f"AS '{clean_col_name}{val}'"
        )
        sql_statements.append(sql_statement)
    
    return sql_statements

def lump_fun(col_name: str, levels: List, other_category: str = 'Other') -> str:
    """
    Create a SQL CASE statement to group categorical levels, lumping rare/unseen levels into 'Other'.
    
    This function generates a SQL expression that keeps only the specified levels and groups
    all other levels into a catch-all category. This makes models robust to new data by
    ensuring unseen categorical levels don't break predictions.
    
    Args:
        col_name (str): The name of the categorical column to process.
        levels (List): A list of categorical levels to keep as-is. All other levels
            will be grouped into the other_category.
        other_category (str, optional): The name for the catch-all category. 
            Defaults to 'Other'.
    
    Returns:
        str: A SQL CASE statement string that creates a new column named '{col_name}_lumped'
            with the specified levels preserved and all others grouped.
    
    Raises:
        TypeError: If col_name is not a string or levels is not a list/array.
        ValueError: If col_name is empty, levels is empty, or other_category is empty.
        
    Example:
        >>> col_name = "vehicle_brand"
        >>> keep_levels = ["Toyota", "Honda", "Ford"]
        >>> sql_statement = lump_fun(col_name, keep_levels)
        >>> print(sql_statement)
        CASE WHEN CAST(vehicle_brand AS VARCHAR) IN ('Toyota', 'Honda', 'Ford') 
        THEN CAST(vehicle_brand AS VARCHAR) ELSE 'Other' END AS vehicle_brand_lumped
        
        This will:
        - Keep "Toyota", "Honda", "Ford" as-is
        - Group "BMW", "Mercedes", etc. into "Other"
        
    Note:
        - The output column is named '{col_name}_lumped'
        - All values are cast to VARCHAR for consistency
        - NULL values will be converted to the other_category
        - Levels are automatically quoted and escaped for SQL safety
    """
    # Input validation
    if not isinstance(col_name, str):
        raise TypeError(f"col_name must be a string, got {type(col_name)}")
    
    if not col_name.strip():
        raise ValueError("col_name cannot be empty or whitespace only")
    
    # Convert numpy array to list if needed
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    
    if not isinstance(levels, list):
        raise TypeError(f"levels must be a list or numpy array, got {type(levels)}")
    
    if len(levels) == 0:
        raise ValueError("levels cannot be empty - must specify at least one level to keep")
    
    if not isinstance(other_category, str):
        raise TypeError(f"other_category must be a string, got {type(other_category)}")
    
    if not other_category.strip():
        raise ValueError("other_category cannot be empty or whitespace only")
    
    # Clean inputs
    clean_col_name = col_name.strip()
    clean_other_category = other_category.strip()
    
    # Convert all levels to strings and escape single quotes for SQL safety
    clean_levels = []
    for i, level in enumerate(levels):
        if level is None:
            raise ValueError(f"Level at index {i} cannot be None/null")
        
        # Convert to string and escape single quotes
        level_str = str(level).replace("'", "''")
        clean_levels.append(level_str)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_levels = []
    for level in clean_levels:
        if level not in seen:
            seen.add(level)
            unique_levels.append(level)
    
    # Create the SQL IN clause with properly quoted levels
    levels_str = ", ".join(f"'{level}'" for level in unique_levels)
    
    # Generate the SQL CASE statement
    sql_statement = (
        f"CASE WHEN CAST({clean_col_name} AS VARCHAR) IN ({levels_str}) "
        f"THEN CAST({clean_col_name} AS VARCHAR) "
        f"ELSE '{clean_other_category}' END AS {clean_col_name}_lumped"
    )
    
    return sql_statement



# function used to prepare a dataset for model building
# uses duckdb as the backend to do the heavy lifting
# if df is provided, it will be used to create a new table in the database
# if con is provided, it will be used to connect to an existing database
# if con is not provided, a new in-memory database will be created
# the function will check if the specified table exists in the database
# and will raise an error if it does not
# the function will then combine the additional_columns and traintest_column while ensuring no duplicates
# and will then check if each variable in modelling_variables exists in the database
# and will raise a warning if it does not
# the function will then create a list of expressions to be used in the model building process
# and will then combine all expressions into a single SQL query
# the function will then execute the query using duckdb and return the result   




def lump_rare_levels_pl(column_series, total_count=None, threshold=0.001, fill_value='Unknown'):
    """
    Lumps rare factor levels into an 'Other' category if they are below the threshold. Handles NoneType values by filling them.
    
    Args:
    column_series (pl.Series): The column to process as a Polars Series.
    total_count (int, optional): The total number of rows in the DataFrame. If not provided, 
                                 it will be inferred from the length of the series.
    threshold (float): The minimum proportion for a level to not be lumped into 'Other'.
    fill_value (str): The value to replace NoneType values with.
    
    Returns:
    pl.Series: A series with the rare levels lumped.
    """
    # Fill NoneType values with the placeholder
    column_series = column_series.fill_null(fill_value)
    
    # Calculate total count if not provided
    if total_count is None:
        total_count = column_series.len()
    
    # Calculate the frequency of each level
    level_counts = column_series.to_frame().group_by(column_series.name).agg(pl.len().alias('counts'))
    
    # Identify levels that are below the threshold and collect them into a list
    rare_levels = level_counts.filter(pl.col('counts') / total_count < threshold)[column_series.name].to_list()
    
    # Define a custom expression to replace rare levels with 'Other'
    lump_expression = pl.when(pl.col(column_series.name).is_in(rare_levels)).then(pl.lit('Other')).otherwise(pl.col(column_series.name))
    
    # Apply the expression to the series and return the modified series
    return column_series.to_frame().with_columns(lump_expression.alias(column_series.name))[column_series.name]




def prepare_data(modelling_variables: List[str], additional_columns: Optional[List[str]] = None, traintest_column: Optional[str] = None, df: Optional[pl.DataFrame] = None, table_name: str = "dataset", formats: dict = {}, con: Optional[duckdb.DuckDBPyConnection] = None):
    # If no connection is provided, create one
    if con is None:
        if df is not None:
            con = duckdb.connect(":memory:")
            # Write the dataframe to the in-memory database
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            con = duckdb.connect()
    else:
        # Check if the provided connection is a duckdb connection
        if not isinstance(con, duckdb.DuckDBPyConnection):
            print("Warning: The provided connection is not a duckdb connection. The function will attempt to process the task on the connection provided but it has not been tested on other database engines.")

    # Check if the specified table exists
    tables = con.execute("SHOW TABLES").fetchall()
    if table_name not in [t[0] for t in tables]:
        raise ValueError(f"The specified table '{table_name}' does not exist in the database. Available tables are: {', '.join([t[0] for t in tables])}")

    # Initialize the expressions list
    expressions = []

    # Combine additional_columns and traintest_column while ensuring no duplicates
    if additional_columns is None:
        additional_columns = []
    if traintest_column and traintest_column not in additional_columns:
        additional_columns.append(traintest_column)

    for var in modelling_variables:
        if var not in con.execute(f"PRAGMA table_info({table_name})").df()['name'].tolist():
            print(f"Warning: Column '{var}' not found in the table. Skipping.")
            continue

        if var in formats:
            dict_values = formats[var]
            if all(isinstance(x, (int, float)) for x in dict_values):
                # Numeric variable: apply o_matrix
                expressions.extend(o_matrix(var, dict_values))
            else:
                # Categorical variable: apply lump_fun
                expressions.append(lump_fun(var, dict_values))
        else:
            # Variable not in dictionary: keep as is
            expressions.append(f"'{var}'")

    # Add additional columns if they exist in the dataset
    for col in additional_columns:
        if col in con.execute(f"PRAGMA table_info({table_name})").df()['name'].tolist():
            expressions.append(col)
        else:
            print(f"Warning: Additional column '{col}' not found in the table. Skipping.")

    # Combine all expressions into a single SQL query
    query = f"SELECT {', '.join(expressions)} FROM {table_name}"

    # Execute the query using duckdb
    result = con.execute(query).df()

    # Close the connection if it was created in this function
    if df is not None and con is not None:
        con.close()

    result = pl.DataFrame(result)

    return result



# Example usage:
# array is your numpy array representing the column you want to process.




"""
Function creates a 'blueprint' of how to process a datasetset
"""
def generate_blueprint(dataframe, threshold=0.0025):
    blueprint = {}
    for column in dataframe.columns:
        try:
            col_data = dataframe[column]
            dtype = col_data.dtype
            
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                # For numeric columns, calculate 5th percentile quantiles and get unique values
                quantiles = np.arange(0.05, 1.05, 0.05).tolist()
                breaks = [col_data.quantile(q) for q in quantiles]
                unique_breaks = list(set(breaks))  # Get unique breaks
                unique_breaks.sort()  # Sort the unique breaks
                blueprint[column] = unique_breaks
            else:
                # For non-numeric columns, lump rare levels using the lump_rare_levels_np function
                # and remove 'Other' as a level
                # col_data_np = col_data.to_numpy()
                lumped_levels = lump_rare_levels_pl(dataframe[column], threshold=threshold)
                levels = np.unique(lumped_levels).tolist()
                if 'Other' in levels:
                    levels.remove('Other')
                blueprint[column] = levels
        
        except Exception as e:
            print(f"Error processing column '{column}': {str(e)}")
            blueprint[column] = f"Error: Unable to process this column. Error message: {str(e)}"
    
    return blueprint









# test that the function works


from glum import GeneralizedLinearRegressorCV
import polars as pl

def fit_lasso_glm(
    dataframe: pl.DataFrame,
    target: str,
    train_test_col: str,
    model_type: str,
    weight_col: str = None,
    DivideTargetByWeight: bool = False
):
    """
    Fits a LASSO-regularized GLM (Generalized Linear Model) to a given dataset using glum's GeneralizedLinearRegressorCV.

    Parameters:
        dataframe (pl.DataFrame): The input data as a Polars DataFrame.
        target (str): The name of the target (response) column.
        train_test_col (str): The name of the column indicating train (1) vs. test (0).
        model_type (str): The type of GLM to fit. Must be either "Poisson" or "Gamma".
        weight_col (str, optional): The name of the weight or exposure column. Defaults to None.
        DivideTargetByWeight (bool, optional): Whether the target needs to be divided by the weights column transformed. If True and weight_col is provided,
                                      the target will be divided by the weight column. Defaults to False.

    Returns:
        GeneralizedLinearRegressorCV: The fitted LASSO-regularized GLM model.

    Raises:
        ValueError: If the DataFrame is empty, required columns are missing, model_type is invalid,
                    or the target/weight columns contain invalid values.
    """
    # Create a copy of the input DataFrame
    dataframe = dataframe.clone()

    # -----------------------------
    # 1. Basic Checks
    # -----------------------------
    if dataframe.height == 0:
        raise ValueError("The input DataFrame is empty. It must contain at least one row.")
    
    if len(dataframe.columns) != len(set(dataframe.columns)):
        raise ValueError("The DataFrame contains duplicate column names.")
    
    required_cols = [target, train_test_col]
    if weight_col:
        required_cols.append(weight_col)
    for col in required_cols:
        if col not in dataframe.columns:
            raise ValueError(f"Missing column '{col}' in dataframe.")
    
    if model_type.lower() not in ["poisson", "gamma"]:
        raise ValueError("model_type must be either 'Poisson' or 'Gamma'.")

    def invalid_target_exists(df: pl.DataFrame, col: str, allow_zero: bool) -> bool:
        if allow_zero:
            invalid_ct = df.filter(
                (pl.col(col) < 0) | pl.col(col).is_infinite() | pl.col(col).is_nan()
            ).height
        else:
            invalid_ct = df.filter(
                (pl.col(col) <= 0) | pl.col(col).is_infinite() | pl.col(col).is_nan()
            ).height
        return invalid_ct > 0
    
    if model_type.lower() == "poisson":
        if invalid_target_exists(dataframe, target, allow_zero=True):
            raise ValueError("Invalid Poisson target values (<0, inf, or NaN).")
    else:  # Gamma
        if invalid_target_exists(dataframe, target, allow_zero=False):
            raise ValueError("Invalid Gamma target values (<=0, inf, or NaN).")
    
    if weight_col:
        w_invalid_ct = dataframe.filter(
            (pl.col(weight_col) <= 0)
            | pl.col(weight_col).is_infinite()
            | pl.col(weight_col).is_nan()
            | pl.col(weight_col).is_null()
        ).height
        if w_invalid_ct > 0:
            raise ValueError("Weight column has invalid values (<=0, inf, NaN, or null).")
    
    # -----------------------------
    # 2. Encode UTF-8 columns
    # -----------------------------
    str_cols = [c for c, dt in zip(dataframe.columns, dataframe.dtypes) if dt == pl.Utf8]
    for sc in str_cols:
        dataframe = dataframe.with_columns(
            pl.col(sc).cast(pl.Categorical).to_physical().alias(sc)
        )
    
    # -----------------------------
    # 3. Transform target if needed
    # -----------------------------
    if (not DivideTargetByWeight) and (weight_col is not None):
        dataframe = dataframe.with_columns(
            [(pl.col(target) / pl.col(weight_col)).alias(target)]
        )
    
    # -----------------------------
    # 4. Split into train/test
    # -----------------------------
    train_df = dataframe.filter(pl.col(train_test_col) == 1)
    if train_df.height == 0:
        raise ValueError("The training subset is empty. Ensure 'train_test_col' contains at least one row with value 1.")
    
    # -----------------------------
    # 5. Prepare X, y, and weights
    # -----------------------------
    X_train = train_df.select(pl.all().exclude([target, train_test_col,weight_col])).to_numpy()
    y_train = train_df.select(pl.col(target)).to_numpy().ravel()
    
    sample_weight = None
    if weight_col:
        sample_weight = train_df.select(pl.col(weight_col)).to_numpy().ravel()
    
    # -----------------------------
    # 6. Fit LASSO GLM
    # -----------------------------
    model = GeneralizedLinearRegressorCV(
        family=model_type.lower(),  # "poisson" or "gamma"
        l1_ratio=1,
        fit_intercept=True,
        cv=5,
        scale_predictors=False
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    return model




from typing import List, Optional
import polars as pl
import duckdb
from sklearn.preprocessing import LabelEncoder
import pandas as pd


df.columns





sum(model) # 125152.054008569
model.alpha_

model.l1_ratio_






import matplotlib.pyplot as plt 

# Calculate average deviance
if model.deviance_path_.ndim == 2:
    avg_deviance = model.deviance_path_.mean(axis=0)
elif model.deviance_path_.ndim == 3:
    avg_deviance = model.deviance_path_[:, 0, :].mean(axis=0)  # Assuming l1_ratio is a single value
else:
    raise ValueError("Unexpected shape for deviance_path_")

print("Shape of avg_deviance:", avg_deviance.shape)

# Plot deviance vs alpha
plt.plot(model.alphas_, avg_deviance, '-o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Average Deviance')
plt.title('Deviance vs Alpha')
plt.show()


# Get the best alpha
best_alpha_index = np.argmin(avg_deviance)
best_alpha = model.alphas_[best_alpha_index]
print(f"Manually found best alpha: {best_alpha}")


# Make predictions with the model 
# Export the model as a rate table for all the variables in the model
# 
predictions  = model.predict(prepped.select(pl.all().exclude(['Exposure', 'traintest','ClaimNb'])).to_pandas())

# Ideally also build a model with the relativitries custom imported 


np.unique(predictions)


help(model.predict)

print(f"Chosen alpha:    {model.alpha_}")
print(f"Chosen l1 ratio: {model.l1_ratio_}")


max(model.alphas_)

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = prepped.select(pl.all().exclude(['Exposure', 'traintest','ClaimNb'])).to_pandas().corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()



# function that will create a ratetable for a given variable. function will accept a dataframe, a model and a variable name and return a ratetable for that variable


def generate_ratetable(var, predictors, dataframe, model, formats=None):
    
    var = 'VehPower'
    predictors=predictor_variables
    dataframe = cars
    model = model
    formats= d


    # if formats is not provided, use the blueprint to generate the formats
    if formats is None:
        formats = generate_blueprint(dataframe)
    
    # select a random line from the dataframe
    df = dataframe.sample(1)

    # prepare the dataframe
    df_prepped = prepare_data(
        modelling_variables=predictors,
        df=df,
        table_name='dataset',
        formats=formats
    )

    # make predictions with the prepared dataframe
    baseline_prediction = model.predict(df_prepped)


    # now we need to make a new dataframe with the unique values of the variable
    unique_values = formats[var]
    length_unique = len(unique_values)
    
    # duplicate df the length of the unique values using polars 
    df_repeated = df.select([pl.col("*").repeat_by(length_unique)]).explode("*")
    
    # replace the variable with the unique values
    df_repeated = df_repeated.with_columns(pl.Series(var, unique_values))

    # prepare the new dataframe 
    df_prepped = prepare_data(
        modelling_variables=[var],
        df=df_repeated,
        table_name='dataset',
        formats=formats
    )

    # make predictions with the prepared dataframe
    new_predictions = model.predict(df_prepped)

    # scaled predictions 
    new_predictions = new_predictions / baseline_prediction
    
    return new_predictions

cars.columns


generate_ratetable(var = 'VehPower', predictors=predictor_variables, dataframe = cars, model = model, formats= d)



d['veh_age']

# Commented out orphaned code fragment:
# This appears to be an incomplete function that was causing linter errors
# 
# def some_function():
#     unique_values = dataframe[var].unique()
#     # select a random line from the dataframe
#     df = dataframe.sample(1)
#     # create a dataframe to store the results
#     results = pd.DataFrame()
#     # loop through the unique values
#     for value in unique_values:
#         # create a dataframe with the value
#         df = pd.DataFrame({var: [value]})
#         # get the prediction for the value
#         prediction = model.predict(df)
#         # create a dataframe with the value and the prediction
#         df['prediction'] = prediction
#         # append the dataframe to the results dataframe
#         results = results.append(df)
#     # return the results dataframe
#     return results

# prepare_data(df = cars, modelling_variables=['veh_age'], additional_columns=['ex','nu_cl'], formats=d, traintest_column='traintest', table_name='dataset')






    # X_train_p = glm_categorizer.fit_transform(df[indep_vars])
    # y_train_p = df[target]
    # if sample_weights is not None:
    #     w_train_p = df[sample_weights]
    # else:
    #     w_train_p = None
    # f_glm1 = GeneralizedLinearRegressor(family=glm_type, alpha_search=True, l1_ratio=1, fit_intercept=True)
    # f_glm1.fit(X_train_p, y_train_p, sample_weight=w_train_p)


z = df['ClaimNb'].values
weight = df['Exposure'].values
y = z / weight # claims frequency

ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train, test = next(ss.split(y))

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
predictors = categoricals + ["BonusMalus", "Density"]
# glm_categorizer = Categorizer(columns=categoricals)  # Commented out - use sklearn preprocessing instead

X_train_p = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_p = glm_categorizer.transform(df[predictors].iloc[test])
y_train_p, y_test_p = y[train], y[test]
w_train_p, w_test_p = weight[train], weight[test]
z_train_p, z_test_p = z[train], z[test]

f_glm1 = GeneralizedLinearRegressor(family='poisson', alpha_search=True, l1_ratio=1, fit_intercept=True)

f_glm1.fit(
    X_train_p,
    y_train_p,
    sample_weight=w_train_p
);

pd.DataFrame({'coefficient': np.concatenate(([f_glm1.intercept_], f_glm1.coef_))},
             index=['intercept'] + f_glm1.feature_names_).T

# Generate all ratetables
all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=df,
    predictor_variables=predictor_variables,
    blueprint=d,
    random_seed=42,
)

# you can now access each ratetable by its variable name
# for example:
# all_tables['VehAge']
# all_tables['Region']

