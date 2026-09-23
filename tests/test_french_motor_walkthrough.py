"""Developer verification for the interactive French motor Python lesson."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "docs" / "examples" / "french_motor_walkthrough.py"
GUIDE = ROOT / "examples" / "pricing_walkthrough.md"
CELL_MARKER = re.compile(r"(?m)^# %% (\d+) — (.+)$")
PYTHON_FENCE = re.compile(r"(?ms)^```python[^\n]*\n(.*?)^```[ \t]*$")


def lesson_cells(source: str) -> list[tuple[int, str, str]]:
    """Return numbered cell source without treating the file as one-shot code."""
    matches = list(CELL_MARKER.finditer(source))
    return [
        (
            int(match.group(1)),
            match.group(2),
            (
                source[match.end() : matches[index + 1].start()]
                if index + 1 < len(matches)
                else source[match.end() :]
            ),
        )
        for index, match in enumerate(matches)
    ]


def markdown_python_blocks(source: str) -> list[str]:
    """Extract the guide's executable Python without reading its companion."""
    return [match.group(1) for match in PYTHON_FENCE.finditer(source)]


def test_optional_script_contains_the_same_cells_as_the_walkthrough():
    """The convenient script must not teach a second, incompatible workflow."""
    cells = lesson_cells(LESSON.read_text(encoding="utf-8"))
    blocks = markdown_python_blocks(GUIDE.read_text(encoding="utf-8"))
    assert len(cells) == len(blocks)
    assert [ast.dump(ast.parse(body)) for _number, _title, body in cells] == [
        ast.dump(ast.parse(block)) for block in blocks
    ]


def test_markdown_walkthrough_executes_independently_on_full_fixture(
    tmp_path, monkeypatch
):
    """Run the public page verbatim, including both pair stages and edit choices."""
    import json
    import xml.etree.ElementTree as ET
    from zipfile import ZipFile

    import polars as pl

    import easy_glm
    from easy_glm import PricingSession
    from easy_glm.core.excel import pair_tables_from_xlsx, rate_model_tables
    from easy_glm.engine import RateModel
    from easy_glm.pricing.review import ReviewPreview
    from easy_glm.pricing.views import DisplayResult
    from easy_glm.workflow.prep import train_holdout

    fixture = pl.read_parquet(ROOT / "tests" / "fixtures" / "french_motor_50k.parquet")
    monkeypatch.setattr(easy_glm, "load_external_dataframe", lambda: fixture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DisplayResult, "show", lambda self: self)
    monkeypatch.setattr(ReviewPreview, "show", lambda self: self)
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)

    source = GUIDE.read_text(encoding="utf-8")
    blocks = markdown_python_blocks(source)
    assert blocks
    assert "def plot_" not in source
    assert "def easyglm_design_kwargs" not in source
    assert "from easy_glm.workflow import" not in source

    namespace = {"__name__": "__main__"}
    for index, block in enumerate(blocks, start=1):
        exec(compile(block, f"{GUIDE.name}:block-{index}", "exec"), namespace)

    basic = namespace["basic"]
    main = namespace["main"]
    first = namespace["first"]
    second = namespace["second"]
    adjusted = namespace["adjusted"]
    refitted = namespace["refitted"]
    reopened = namespace["reopened"]
    work = namespace["work"]
    data = namespace["data"]
    assert data.height == 50_000

    # Roles drive search eligibility.  Inspect the saved candidacy table rather
    # than repeating the relatively expensive residual searches from the guide.
    ignored = work.summary().filter(pl.col("column") == "VehGas").row(0, named=True)
    assert ignored == {
        "column": "VehGas",
        "role": "ignore",
        "factor_candidate": False,
        "pair_candidate": False,
    }
    assert all("VehGas" not in model.summary()["factors"] for model in (basic, main))
    assert all(
        "VehGas" not in pair.parents for pair in second._run.rate_model.pair_tables
    )

    # The portable settings file restores the exact fixed split and the design
    # choices made before fitting, without fitting another model.
    settings_path = tmp_path / "motor_settings.json"
    saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    restored_settings = PricingSession.from_settings(data, settings_path)
    for section in ("data", "design"):
        assert (
            restored_settings.settings()[section] == saved_settings["project"][section]
        )
    assert saved_settings["project"]["data"]["roles"]["VehGas"] == "ignore"
    assert saved_settings["project"]["design"]["defaults"]["n_bins"] == 8
    saved_designs = saved_settings["project"]["design"]["variables"]
    assert saved_designs["DrivAge"]["knots"] == [25, 35, 45, 55, 65, 75]
    assert saved_designs["VehAge"]["knots"] == [1, 3, 6, 10, 15]
    assert saved_designs["BonusMalus"]["knots"] == [50, 60, 75, 100, 125]
    assert saved_designs["Density"]["knots"] == [50, 200, 1000, 5000, 10000]
    assert saved_designs["Region"]["kind"] == "categorical"

    assert basic.summary()["factors"] == ["DrivAge", "VehAge"]
    assert main.summary()["factors"] == ["DrivAge", "VehAge", "BonusMalus", "Density"]
    assert "Region" not in second.summary()["factors"]
    assert len(first._run.rate_model.pair_tables) == 1
    assert len(second._run.rate_model.pair_tables) == 2
    assert (
        main._run.rate_model.to_dict()["variables"]
        == first._run.rate_model.to_dict()["variables"]
    )
    assert (
        first._run.rate_model.to_dict()["variables"]
        == second._run.rate_model.to_dict()["variables"]
    )
    assert (
        first._run.rate_model.to_dict()["pair_tables"][0]
        == second._run.rate_model.to_dict()["pair_tables"][0]
    )
    for original, frozen in zip(
        second._run.rate_model.to_dict()["pair_tables"],
        adjusted._run.rate_model.to_dict()["pair_tables"],
        strict=True,
    ):
        # Pricing edits relabel old CV evidence as historical, but keep rates.
        for key in ("stage_id", "parents", "axes", "cells"):
            assert frozen[key] == original[key]

    for candidate in (adjusted, refitted):
        edited = candidate.relativities("DrivAge").table
        row = edited.filter((pl.col("from") == 25) & (pl.col("to") == 35))
        assert row["relativity"].item() == 0.95
        assert set(candidate._run.metrics) == {"train"}
    assert second._run.config.adjustments == []
    assert len(refitted._run.rate_model.pair_tables) == 2
    assert namespace["accepted"] is refitted

    # Every checkpoint retains the session's one policy-level split.  Reviews
    # create new scorers and leave the fitted source checkpoint unchanged.
    checkpoints = (
        basic,
        main,
        first,
        second,
        adjusted,
        refitted,
        namespace["cell_candidate"],
        reopened,
    )
    train_ids = set(basic._frame("train")["IDpol"].to_list())
    holdout_ids = set(basic._frame("holdout")["IDpol"].to_list())
    restored_train, restored_holdout = train_holdout(
        restored_settings._data, restored_settings._project.data.split
    )
    assert train_ids
    assert holdout_ids
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == set(data["IDpol"].to_list())
    assert set(restored_train["IDpol"].to_list()) == train_ids
    assert set(restored_holdout["IDpol"].to_list()) == holdout_ids
    for checkpoint in checkpoints:
        assert set(checkpoint._frame("train")["IDpol"].to_list()) == train_ids
        assert set(checkpoint._frame("holdout")["IDpol"].to_list()) == holdout_ids
        prediction = checkpoint.predict(data)
        assert np.isfinite(prediction).all()
        assert (prediction > 0).all()
    assert (
        second._run.rate_model.to_dict()["variables"]
        == main._run.rate_model.to_dict()["variables"]
    )
    assert second._run.config.adjustments == []
    assert adjusted._run.rate_model.to_dict() != second._run.rate_model.to_dict()

    np.testing.assert_allclose(
        reopened.predict(data), refitted.predict(data), rtol=1e-12
    )
    np.testing.assert_allclose(
        namespace["expected_claims"],
        namespace["predicted_frequency"] * data["Exposure"].to_numpy(),
        rtol=1e-12,
    )
    assert settings_path.is_file()
    assert (tmp_path / "motor_pricing_tables.xlsx").is_file()
    saved = json.loads((tmp_path / "motor_pricing_model.easyglm").read_text())
    assert saved
    assert "IDpol" not in saved.get("data", {})

    # Rebuild the deployed scorer from the actual workbook cells.  This checks
    # both amended main rates and ordered pair tables, rather than only checking
    # that an .xlsx file was created.
    workbook = tmp_path / "motor_pricing_tables.xlsx"
    accepted_scorer = refitted._run.rate_model
    # Read the typed worksheet cells directly.  A mixed string/numeric column
    # read through fastexcel is coerced to text and rounds the base rate.
    with ZipFile(workbook) as archive:
        namespace_xml = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = ["".join(item.itertext()) for item in shared_root]
        summary_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    def workbook_cell(cell):
        value = cell.find(f"{namespace_xml}v")
        assert value is not None and value.text is not None
        if cell.get("t") == "s":
            return shared_strings[int(value.text)]
        if cell.get("t") == "b":
            return value.text == "1"
        return float(value.text)

    workbook_summary = {}
    for row in summary_root.iter(f"{namespace_xml}row"):
        cells = row.findall(f"{namespace_xml}c")
        assert len(cells) in (1, 2)
        workbook_summary[str(workbook_cell(cells[0]))] = (
            workbook_cell(cells[1]) if len(cells) == 2 else None
        )
    assert workbook_summary["family"] == refitted._run.config.family
    assert workbook_summary["target"] == refitted._run.config.target
    assert workbook_summary["weight"] == refitted._run.config.weight
    assert workbook_summary["link"] == accepted_scorer.metadata.link
    workbook_divides_target = workbook_summary["target divided by weight"]
    assert isinstance(workbook_divides_target, bool)
    assert workbook_divides_target is accepted_scorer.metadata.divide_target_by_weight
    expected_tables = rate_model_tables(accepted_scorer)
    workbook_tables = {
        variable: pl.read_excel(workbook, sheet_name=variable)
        for variable in accepted_scorer.variables
    }
    for variable, expected in expected_tables.items():
        np.testing.assert_allclose(
            workbook_tables[variable]["relativity"].to_numpy(),
            expected["relativity"].to_numpy(),
            rtol=1e-12,
        )
    workbook_mains = RateModel.from_rate_tables(
        workbook_tables,
        float(workbook_summary["base_rate"]),
        model_type=str(workbook_summary["family"]),
        target=str(workbook_summary["target"]),
        weight_col=str(workbook_summary["weight"]),
        exposure_col=str(workbook_summary["weight"]),
        link=str(workbook_summary["link"]),
        divide_target_by_weight=workbook_divides_target,
        predictor_variables=list(accepted_scorer.variables),
    )
    workbook_scorer = RateModel(
        workbook_mains.base_rate,
        workbook_mains.variables,
        metadata=workbook_mains.metadata,
        pair_tables=pair_tables_from_xlsx(workbook),
    )
    np.testing.assert_allclose(
        workbook_scorer.predict(data, exposure_col=None),
        accepted_scorer.predict(data, exposure_col=None),
        rtol=1e-12,
    )

    # The preview is also independently usable as a chart/table, with no notebook.
    ae = refitted.ae("DrivAge")
    assert len(ae.figure.data) >= 3
    assert ae.table["actual"].sum() == refitted._frame("train")["ClaimNb"].sum()
    pair = refitted.relativities("VehAge", "Region")
    assert pair.figure is not None
    assert pair.table.height > 0
