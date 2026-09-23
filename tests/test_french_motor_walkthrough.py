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

    import polars as pl

    import easy_glm
    from easy_glm.pricing.review import ReviewPreview
    from easy_glm.pricing.views import DisplayResult

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
    data = namespace["data"]
    assert data.height == 50_000
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

    np.testing.assert_allclose(
        namespace["reopened"].predict(data), refitted.predict(data), rtol=1e-12
    )
    np.testing.assert_allclose(
        namespace["expected_claims"],
        namespace["predicted_frequency"] * data["Exposure"].to_numpy(),
        rtol=1e-12,
    )
    assert (tmp_path / "motor_settings.json").is_file()
    assert (tmp_path / "motor_pricing_tables.xlsx").is_file()
    saved = json.loads((tmp_path / "motor_pricing_model.easyglm").read_text())
    assert saved
    assert "IDpol" not in saved.get("data", {})

    # The preview is also independently usable as a chart/table, with no notebook.
    ae = refitted.ae("DrivAge")
    assert len(ae.figure.data) >= 3
    assert ae.table["actual"].sum() == refitted._frame("train")["ClaimNb"].sum()
    pair = refitted.relativities("VehAge", "Region")
    assert pair.figure is not None
    assert pair.table.height > 0
