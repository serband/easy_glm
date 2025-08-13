from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def plot_all_ratetables(
    all_tables: dict[str, pl.DataFrame], blueprint: dict[str, Any]
):  # pragma: no cover - visual
    """Plot relativity curves for each rate table (line for numeric, bar for categorical)."""
    for var_name, table in all_tables.items():
        if not isinstance(table, pl.DataFrame) or table.is_empty():
            print(f"Skipping '{var_name}' as it's not a valid DataFrame.")
            continue
        plt.figure(figsize=(10, 6))
        is_numeric = all(
            isinstance(x, int | float) for x in blueprint.get(var_name, [])
        )
        if is_numeric:
            sorted_table = table.sort(var_name)
            sns.lineplot(
                data=sorted_table.to_pandas(), x=var_name, y="relativity", marker="o"
            )
        else:
            sns.barplot(
                data=table.to_pandas(), x=var_name, y="relativity", color="skyblue"
            )
            plt.xticks(rotation=45, ha="right")
        plt.title(f"Relativity for {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Relativity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
