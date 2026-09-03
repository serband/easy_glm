import polars as pl


def plot_all_ratetables(
    all_tables: dict[str, pl.DataFrame],
):  # pragma: no cover - visual
    """Bar chart of relativities per band for each table from
    :func:`~easy_glm.rate_tables` (needs ``pip install "easy_glm[viz]"``)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plot_all_ratetables needs matplotlib and seaborn: "
            'pip install "easy_glm[viz]"'
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
        sns.barplot(data=table.to_pandas(), x="label", y="relativity", color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Relativity for {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Relativity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
