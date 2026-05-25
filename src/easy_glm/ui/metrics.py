from __future__ import annotations

import polars as pl

from easy_glm.engine import RateModel

FORMULAS: dict[str, str] = {
    "sum_weighted": "sum(target × weight) / sum(weight)",
    "sum_unweighted": "sum(target) / count",
    "sum_over_weight": "sum(target) / sum(weight)",
}


def compute_actual_expected(
    rm: RateModel,
    data: pl.DataFrame,
    variable: str,
    formula: str = "sum_weighted",
) -> dict:
    target = rm.metadata.target
    weight_col = rm.metadata.weight_col
    train_test_col = rm.metadata.train_test_col

    if target is None:
        raise ValueError("Model metadata missing 'target' column")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    predictions = rm.predict(data)
    data = data.with_columns(pred=pl.Series("pred", predictions))

    subsets = {"all": data}
    if train_test_col and train_test_col in data.columns:
        subsets["train"] = data.filter(pl.col(train_test_col) == 1)
        subsets["test"] = data.filter(pl.col(train_test_col) == 0)

    config = rm.variables[variable]
    rows = config.table
    level_edges = _level_labels(rows)

    results: dict[str, list[dict]] = {}
    for subset_name, subset in subsets.items():
        results[subset_name] = []
        for i, row in enumerate(rows):
            mask = _mask_for_row(subset, variable, row)
            matched = subset.filter(mask)
            if matched.is_empty():
                results[subset_name].append(
                    {
                        "level": level_edges[i],
                        "actual": 0.0,
                        "expected": 0.0,
                        "exposure": 0.0,
                    }
                )
                continue

            actual = _compute_actual(matched, target, weight_col, formula)
            expected = _compute_actual(matched, "pred", weight_col, formula)
            exposure = (
                float(matched[weight_col].sum())
                if weight_col and weight_col in matched.columns
                else float(len(matched))
            )
            results[subset_name].append(
                {
                    "level": level_edges[i],
                    "actual": actual,
                    "expected": expected,
                    "exposure": exposure,
                }
            )

    return {"subsets": results, "variable": variable}


def _level_labels(rows) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if row.from_ is None and row.to_ is None:
            labels.append("Other / Unknown")
        elif row.from_ is None:
            labels.append(f"< {row.to_}")
        elif row.to_ is None:
            labels.append(f"≥ {row.from_}")
        elif row.from_ == row.to_:
            labels.append(str(row.from_))
        else:
            labels.append(f"[{row.from_}, {row.to_})")
    return labels


def _mask_for_row(data: pl.DataFrame, variable: str, row) -> pl.Series:
    col = data[variable]
    if row.from_ is None and row.to_ is None:
        return pl.lit(True)
    if row.from_ is None:
        return col < float(row.to_)
    if row.to_ is None:
        return col >= float(row.from_)
    if row.from_ == row.to_:
        return col == row.from_
    return (col >= float(row.from_)) & (col < float(row.to_))


def _compute_actual(
    df: pl.DataFrame, value_col: str, weight_col: str | None, formula: str
) -> float:
    values = df[value_col]
    if formula == "sum_weighted":
        if weight_col and weight_col in df.columns:
            weights = df[weight_col]
            return float((values * weights).sum() / weights.sum())
        return float(values.mean())
    elif formula == "sum_over_weight":
        if weight_col and weight_col in df.columns:
            return float(values.sum() / df[weight_col].sum())
        return float(values.sum())
    else:
        return float(values.mean())
