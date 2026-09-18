from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm.core.excel import pair_table_frames, pair_tables_from_xlsx
from easy_glm.engine.models import (
    FromToRow,
    ModelMetadata,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.engine.rate_model import RateModel


def numeric_axis() -> VariableConfig:
    return VariableConfig(
        "numeric",
        [FromToRow(None, 10, 1), FromToRow(10, None, 1), FromToRow(None, None, 1)],
    )


def categorical_axis() -> VariableConfig:
    return VariableConfig(
        "categorical",
        [FromToRow("R", "R", 1), FromToRow(None, None, 1)],
    )


def model() -> RateModel:
    main = VariableConfig(
        "categorical", [FromToRow("K", "K", 2), FromToRow(None, None, 1)]
    )
    RateModel._precompute_variables({"main": main})
    first = PairTableConfig(
        "first",
        ("age", "region"),
        (numeric_axis(), categorical_axis()),
        [PairCellRow(0, 0, 3, 2, 2.0, 0.5), PairCellRow(2, 1, 4, 1, 1.0, 0.25)],
    )
    second = PairTableConfig(
        "second",
        ("region", "main"),
        (
            categorical_axis(),
            VariableConfig(
                "categorical",
                [FromToRow("K", "K", 1), FromToRow(None, None, 1)],
            ),
        ),
        [PairCellRow(1, 1, 5)],
    )
    return RateModel(
        0.1,
        {"main": main},
        metadata=ModelMetadata(offset_col="offset", exposure_col="exposure"),
        pair_tables=[first, second],
    )


def test_pair_tables_score_in_order_with_pair_only_parents_and_offset_once() -> None:
    rm = model()
    data = pl.DataFrame(
        {
            "main": ["K", "K", "other", "K"],
            "age": [5.0, 15.0, None, 5.0],
            "region": ["R", "R", "new", None],
            "offset": [0.2] * 4,
            "exposure": [2.0] * 4,
        }
    )
    expected_rate = np.array([0.1 * 2 * 3, 0.1 * 2, 0.1 * 4 * 5, 0.1 * 2]) * np.exp(0.2)
    np.testing.assert_allclose(rm.predict(data), expected_rate * 2, rtol=1e-14)
    np.testing.assert_allclose(
        np.exp(rm.linear_predictor(data)), expected_rate, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.exp(rm.linear_predictor(data, include_offset=False)),
        expected_rate / np.exp(0.2),
        rtol=1e-14,
    )


def test_pair_roundtrip_clone_and_snapshot_keep_independent_axes(tmp_path) -> None:
    rm = model()
    frame = pl.DataFrame(
        {"main": ["K"], "age": [5.0], "region": ["R"], "offset": [0.0]}
    )
    assert rm.to_dict()["format_version"] == 3
    rm.create_snapshot("fitted")
    clone = rm.clone()
    clone.update_pair_cell("first", 0, 0, 7)
    clone.create_snapshot("edited")
    assert clone.predict(frame, exposure_col=None)[0] == pytest.approx(1.4)
    assert rm.predict(frame, exposure_col=None)[0] == pytest.approx(0.6)
    clone.switch_to(1)
    assert clone.predict(frame, exposure_col=None)[0] == pytest.approx(0.6)
    path = tmp_path / "pairs.easyglm"
    rm.to_json(path)
    loaded = RateModel.from_json(path)
    assert loaded.get_pair_table("first").parents == ("age", "region")
    np.testing.assert_allclose(
        loaded.predict(frame, exposure_col=None), rm.predict(frame, exposure_col=None)
    )
    assert RateModel(0.1, {}).to_dict()["format_version"] == 2


def test_pair_excel_has_ordered_axes_and_cell_support(tmp_path) -> None:
    rm = model()
    path = rm.to_excel(tmp_path / "pairs.xlsx")
    manifest = pl.read_excel(path, sheet_name="Pair stages")
    assert manifest["stage_id"].to_list() == ["first", "second"]
    assert manifest["parent_a"].to_list() == ["age", "region"]
    first = pair_table_frames(rm)["first"]
    assert (
        first.filter((pl.col("axis_a_row") == 0) & (pl.col("axis_b_row") == 0))[
            "relativity"
        ][0]
        == 3
    )
    cells_sheet = manifest["cells_sheet"][0]
    saved = pl.read_excel(path, sheet_name=cells_sheet)
    assert saved.height == 6
    assert saved["fitting_weight"][0] == 2.0
    restored = RateModel(
        rm.base_rate,
        rm.variables,
        metadata=rm.metadata,
        pair_tables=pair_tables_from_xlsx(path),
    )
    data = pl.DataFrame(
        {
            "main": ["K", "other"],
            "age": [5.0, None],
            "region": ["R", None],
            "offset": [0.0, 0.4],
        }
    )
    np.testing.assert_allclose(
        restored.predict(data, exposure_col=None),
        rm.predict(data, exposure_col=None),
        rtol=1e-12,
    )


def test_pair_grid_limit_checked_before_matrix_allocation() -> None:
    axis = VariableConfig(
        "categorical",
        [*(FromToRow(str(i), str(i), 1) for i in range(100)), FromToRow(None, None, 1)],
    )
    with pytest.raises(ValueError, match="maximum is 10,000"):
        RateModel(
            1.0,
            {},
            pair_tables=[PairTableConfig("large", ("a", "b"), (axis, axis), [])],
        )
