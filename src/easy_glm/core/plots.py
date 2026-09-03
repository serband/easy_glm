import polars as pl


def plot_all_ratetables(
    all_tables: dict[str, pl.DataFrame],
):  # pragma: no cover - visual
    """Plot numeric relativities as curves and text relativities as bars.

    The tables come from :func:`~easy_glm.rate_tables`. Numeric bands use a
    stepped curve (or a line for a piecewise-linear factor), while categorical
    levels use bars.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plot_all_ratetables needs matplotlib and seaborn; install the "
            "standard easy-glm package to include them"
        ) from exc

    for var_name, table in all_tables.items():
        if not isinstance(table, pl.DataFrame) or table.is_empty():
            print(f"Skipping '{var_name}' as it's not a valid DataFrame.")
            continue
        if "label" not in table.columns or "relativity" not in table.columns:
            raise ValueError(
                f"Table for {var_name!r} needs 'label' and 'relativity' columns "
                "(as produced by easy_glm.rate_tables)"
            )
        plt.figure(figsize=(10, 6))
        from_values = table["from"].drop_nulls()
        is_numeric = from_values.dtype.is_numeric()
        if is_numeric:
            numeric = table.filter(
                pl.col("from").is_not_null() | pl.col("to").is_not_null()
            ).sort("from", nulls_last=False)
            x = [
                row["from"] if row["from"] is not None else row["to"]
                for row in numeric.select("from", "to").to_dicts()
            ]
            y = numeric["relativity"].to_list()
            if "slope" in numeric.columns:
                plt.plot(x, y, marker="o", color="steelblue")
            else:
                plt.step(x, y, where="post", color="steelblue", linewidth=2)
                plt.scatter(x, y, color="steelblue", zorder=3)
            plt.xlabel(var_name)
        else:
            sns.barplot(
                data=table.to_pandas(), x="label", y="relativity", color="skyblue"
            )
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(var_name)
        plt.title(f"Relativity for {var_name}")
        plt.ylabel("Relativity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
