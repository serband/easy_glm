import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def plot_all_ratetables(all_tables: Dict[str, pl.DataFrame], blueprint: Dict[str, Any]):
    """
    Plots relativity curves for all rate tables.

    This function iterates through a dictionary of rate tables and generates a plot
    for each one. It creates line plots for numeric variables and bar plots for
    categorical variables.

    Args:
        all_tables (Dict[str, pl.DataFrame]): A dictionary where keys are variable
            names and values are the corresponding Polars DataFrame rate tables.
        blueprint (Dict[str, Any]): The blueprint used to generate the rate tables,
            which helps in determining whether a variable is numeric or categorical.
    """
    for var_name, table in all_tables.items():
        if not isinstance(table, pl.DataFrame) or table.is_empty():
            print(f"Skipping '{var_name}' as it's not a valid DataFrame.")
            continue

        plt.figure(figsize=(10, 6))
        
        # Determine if the variable is numeric or categorical from the blueprint
        is_numeric = all(isinstance(x, (int, float)) for x in blueprint.get(var_name, []))

        if is_numeric:
            # Line plot for numeric variables
            sorted_table = table.sort(var_name)
            sns.lineplot(data=sorted_table.to_pandas(), x=var_name, y='relativity', marker='o')
            plt.title(f'Relativity for {var_name}')
            plt.xlabel(var_name)
            plt.ylabel('Relativity')
        else:
            # Bar plot for categorical variables
            sns.barplot(data=table.to_pandas(), x=var_name, y='relativity', color='skyblue')
            plt.title(f'Relativity for {var_name}')
            plt.xlabel(var_name)
            plt.ylabel('Relativity')
            plt.xticks(rotation=45, ha='right')

        plt.grid(True)
        plt.tight_layout()
        plt.show()
