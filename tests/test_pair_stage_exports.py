"""Frozen pair outputs retain every stage and score without CatBoost."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from easy_glm.engine import RateModel
from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.workflow import Project, VariableDesign, prepare, run_model
from easy_glm.workflow.export import to_scoring_script, to_script
from easy_glm.workflow.project import PairCandidateConfig, PairStageConfig
from easy_glm.workflow.reduction import reduced_challenger
from easy_glm.workflow.report import to_report_html


def _staged_run(tmp_path: Path):
    rng = np.random.default_rng(612)
    n = 140
    x = rng.normal(size=n)
    region = np.where(np.arange(n) % 2, "north", "south")
    segment = np.where(np.arange(n) % 3, "retail", "fleet")
    target = rng.poisson(np.exp(0.2 + 0.35 * (x >= 0) + 0.2 * (region == "north")))
    raw = pl.DataFrame(
        {
            "x": x,
            "region": region,
            "segment": segment,
            "target": target,
            "traintest": (np.arange(n) < 110).astype(int),
        }
    )
    source = tmp_path / "book.parquet"
    raw.write_parquet(source)
    project = Project(name="Pair output proof")
    project.data.source.path = str(source)
    project.data.roles = {
        "x": "predictor",
        "region": "predictor",
        "segment": "predictor",
        "target": "target",
        "traintest": "split",
    }
    project.design.variables["x"] = VariableDesign(knots=[0.0])
    cfg = project.new_model("priced", predictors=["x"])
    cfg.penalty.alpha = 0.01
    prepared = prepare(project)
    run = run_model(project, prepared, "priced")

    axis_x = VariableConfig(
        type="numeric",
        table=[
            FromToRow(None, 0.0, 1.0),
            FromToRow(0.0, None, 1.0),
            FromToRow(None, None, 1.0),
        ],
    )
    axis_region = VariableConfig(
        type="categorical",
        table=[
            FromToRow("north", "north", 1.0),
            FromToRow("south", "south", 1.0),
            FromToRow(None, None, 1.0),
        ],
    )
    table = PairTableConfig(
        stage_id="pair-1",
        parents=("x", "region"),
        axes=(axis_x, axis_region),
        cells=[
            PairCellRow(i, j, 1.2 if (i, j) == (1, 0) else 1.0, 12, 12.0, 0.1)
            for i in range(3)
            for j in range(3)
        ],
    )
    run.rate_model.add_pair_table(table)
    candidate = PairCandidateConfig(2, 5, 0.1, 3.0)
    cfg.pair_stages = [PairStageConfig("pair-1", "x", "region", candidates=[candidate])]
    run.pair_stages = [
        SimpleNamespace(
            stage_id="pair-1",
            chosen_candidate=None,
            prefix_cv_loss=1.3,
            table_cv_loss=1.2,
            teacher_cv_loss=1.1,
            approximation_loss=0.1,
        )
    ]
    return project, run, prepared


def test_frozen_python_scorer_matches_saved_model_in_fresh_process(tmp_path):
    _, run, prepared = _staged_run(tmp_path)
    script = tmp_path / "frozen.py"
    script.write_text(to_scoring_script(run, output_prefix="frozen"))
    script_source = script.read_text()
    assert "CatBoost" not in script_source.split("import json", 1)[1]
    child = r"""
import builtins
import json
import runpy
import sys
import polars as pl
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'catboost' or name.startswith('catboost.'):
        raise AssertionError('scoring imported CatBoost')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
scope = runpy.run_path(sys.argv[1])
frame = pl.read_parquet(sys.argv[2])
print(json.dumps(scope['predict'](frame, exposure_col=None).tolist()))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    project_path = run.project_snapshot["data"]["source"]["path"]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(script),
            project_path,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert project_path
    np.testing.assert_allclose(
        json.loads(process.stdout), run.predict(prepared), rtol=1e-14
    )


def test_pair_training_export_replays_workflow_and_saved_screen(tmp_path):
    project, run, _ = _staged_run(tmp_path)
    screen = project.to_dict()
    screen["models"] = {}
    screen["champion"] = None
    screen["exploration"] = {}
    project.exploration["feature_selection"] = {
        "version": 1,
        "project": screen,
        "options": {"permutations": 1},
    }
    source = to_script(project, "priced", run=run, output_prefix="staged")
    compile(source, "staged.py", "exec")
    assert "RUN_FEATURE_SELECTION = True" in source
    assert "prepared = prepare(project, df)" in source
    assert (
        "run = run_model(project, prepared, 'priced', replay_pair_adjustments=True)"
        in source
    )
    assert "pair-1" in source
    assert "VariableDesign" not in source


def _independent_excel_scores(workbook: Path, new_data: Path) -> list[float]:
    """Read XLSX XML in a fresh process, without easy_glm or Excel helpers."""
    child = r"""
import json
import re
import sys
import zipfile
from xml.etree import ElementTree as ET
import polars as pl

book_path, data_path = sys.argv[1:]
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(book_path) as book:
    workbook = ET.fromstring(book.read('xl/workbook.xml'))
    relationships = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
    targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships}
    sheet_paths = {}
    for sheet in workbook.findall('m:sheets/m:sheet', ns):
        target = targets[sheet.attrib['{' + ns['r'] + '}id']]
        sheet_paths[sheet.attrib['name']] = target.lstrip('/') if target.startswith('/') else 'xl/' + target
    strings = []
    if 'xl/sharedStrings.xml' in book.namelist():
        shared = ET.fromstring(book.read('xl/sharedStrings.xml'))
        for item in shared.findall('m:si', ns):
            strings.append(''.join(node.text or '' for node in item.findall('.//m:t', ns)))
    def rows(name):
        root = ET.fromstring(book.read(sheet_paths[name]))
        result = []
        for row in root.findall('m:sheetData/m:row', ns):
            cells = {}
            for cell in row.findall('m:c', ns):
                letters = re.match(r'[A-Z]+', cell.attrib['r']).group()
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter) - ord('A') + 1
                index -= 1
                value = cell.find('m:v', ns)
                if cell.attrib.get('t') == 'inlineStr':
                    inline = cell.find('m:is', ns)
                    parsed = ''.join(node.text or '' for node in inline.findall('.//m:t', ns))
                elif value is None:
                    parsed = None
                elif cell.attrib.get('t') == 's':
                    parsed = strings[int(value.text)]
                elif cell.attrib.get('t') in ('str', 'e'):
                    parsed = value.text
                else:
                    parsed = float(value.text)
                cells[index] = parsed
            result.append([cells.get(i) for i in range(max(cells, default=-1) + 1)])
        return result
    def table(name):
        raw = rows(name)
        names = raw[0]
        return [dict(zip(names, row + [None] * (len(names) - len(row)))) for row in raw[1:]]

    summary = {row[0]: row[1] for row in rows('Summary') if len(row) >= 2}
    main = table('x')
    stages = table('Pair stages')
    def index(value, axis, kind):
        for i, row in enumerate(axis):
            low, high = row['from'], row['to']
            if kind == 'categorical':
                if value is not None and low == high == value:
                    return i
            elif value is None:
                if low is None and high is None:
                    return i
            elif not (low is None and high is None):
                if (low is None or value >= low) and (high is None or value < high):
                    return i
        if kind == 'categorical':
            return len(axis) - 1
        raise AssertionError('numeric value outside tiled axis')
    compiled = []
    for stage in sorted(stages, key=lambda row: row['order']):
        axis_a = table(stage['axis_a_sheet'])
        axis_b = table(stage['axis_b_sheet'])
        cells = {(int(row['axis_a_row']), int(row['axis_b_row'])): row['relativity']
                 for row in table(stage['cells_sheet'])}
        compiled.append((stage, axis_a, axis_b, cells))
    predictions = []
    for row in pl.read_parquet(data_path).to_dicts():
        result = summary['base_rate'] * main[index(row['x'], main, 'numeric')]['relativity']
        for stage, axis_a, axis_b, cells in compiled:
            ia = index(row[stage['parent_a']], axis_a, stage['axis_a_type'])
            ib = index(row[stage['parent_b']], axis_b, stage['axis_b_type'])
            result *= cells.get((ia, ib), 1.0)
        predictions.append(result)
    print(json.dumps(predictions))
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(workbook), str(new_data)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_excel_reconstruction_independently_matches_frozen_scorer(tmp_path):
    _, run, _ = _staged_run(tmp_path)
    workbook = tmp_path / "stage_tables.xlsx"
    run.rate_model.to_excel(workbook)
    new_rows = pl.DataFrame(
        {
            "x": [-100.0, -0.1, 0.0, 3.0, None],
            "region": ["north", "south", "north", "unseen", "north"],
        }
    )
    new_data = tmp_path / "new.parquet"
    new_rows.write_parquet(new_data)
    np.testing.assert_allclose(
        _independent_excel_scores(workbook, new_data),
        run.rate_model.predict(new_rows, exposure_col=None),
        rtol=1e-13,
    )


def test_training_script_replays_two_sequential_pairs_end_to_end(tmp_path):
    project, _, prepared = _staged_run(tmp_path)
    candidate = PairCandidateConfig(2, 5, 0.1, 3.0)
    project.models["priced"].pair_stages = [
        PairStageConfig("first", "x", "region", candidates=[candidate]),
        PairStageConfig("second", "x", "segment", candidates=[candidate]),
    ]
    direct = run_model(project, prepared, "priced")
    screening_project = project.to_dict()
    screening_project["models"] = {}
    screening_project["champion"] = None
    screening_project["exploration"] = {}
    project.exploration["feature_selection"] = {
        "version": 1,
        "project": screening_project,
        "options": {
            "family": "poisson",
            "n_alphas": 2,
            "repeats": 1,
            "seed": 11,
            "include_unassigned": False,
        },
    }
    assert [stage.stage_id for stage in direct.pair_stages] == ["first", "second"]
    assert [table.stage_id for table in direct.rate_model.pair_tables] == [
        "first",
        "second",
    ]
    script = tmp_path / "retrain.py"
    script.write_text(to_script(project, "priced", run=direct, output_prefix="replay"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    assert "pair stages: 2" in process.stdout
    assert (tmp_path / "replay_feature_selection.json").is_file()
    replayed = RateModel.from_json(tmp_path / "replay.easyglm")
    assert [table.stage_id for table in replayed.pair_tables] == ["first", "second"]
    np.testing.assert_allclose(
        replayed.predict(prepared, exposure_col=None),
        direct.predict(prepared),
        rtol=1e-8,
    )
    workbook = tmp_path / "replay_rate_tables.xlsx"
    assert workbook.is_file()
    fresh = pl.DataFrame(
        {
            "x": [-10.0, 0.0, 10.0, None],
            "region": ["north", "south", "new", "north"],
            "segment": ["retail", "fleet", "new", "retail"],
        }
    )
    fresh_path = tmp_path / "fresh.parquet"
    fresh.write_parquet(fresh_path)
    np.testing.assert_allclose(
        _independent_excel_scores(workbook, fresh_path),
        replayed.predict(fresh, exposure_col=None),
        rtol=1e-13,
    )
    frozen = tmp_path / "frozen_two_pairs.py"
    frozen.write_text(to_scoring_script(direct))
    child = r"""
import builtins
import json
import runpy
import sys
import polars as pl
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'catboost' or name.startswith('catboost.'):
        raise AssertionError('frozen scoring imported CatBoost')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
scope = runpy.run_path(sys.argv[1])
print(json.dumps(scope['predict'](pl.read_parquet(sys.argv[2]), exposure_col=None).tolist()))
"""
    frozen_scores = subprocess.run(
        [sys.executable, "-c", child, str(frozen), str(fresh_path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(
        json.loads(frozen_scores.stdout),
        direct.rate_model.predict(fresh, exposure_col=None),
        rtol=1e-14,
    )
    report = to_report_html(project, {"priced": direct}, prepared, champion="priced")
    assert "Stage 2: x × region" in report
    assert "Stage 3: x × segment" in report


def test_report_shows_own_pair_axes_support_and_complete_importance(tmp_path):
    project, run, prepared = _staged_run(tmp_path)
    report = to_report_html(project, {"priced": run}, prepared, champion="priced")
    assert "Stage 2: x × region" in report
    assert "pair-1" in report
    assert "All deployed cells and support" in report
    assert "north" in report and "south" in report
    assert "teacher-to-table approximation loss" in report
    assert "CV deviance change: table minus CatBoost" in report
    assert "Complete deployed scorer" in report
    assert "Main-effects GLM coefficients" in report


def test_smaller_challenger_drops_dependent_pair_and_keeps_pair_only_parent(tmp_path):
    project, _, _ = _staged_run(tmp_path)
    project.data.roles["aux"] = "predictor"
    project.models["priced"].predictors = ["x", "aux"]
    project.models["priced"].pair_stages.append(
        PairStageConfig(
            "pair-only",
            "region",
            "aux",
            candidates=[PairCandidateConfig(2, 5, 0.1, 3.0)],
        )
    )
    reduced = reduced_challenger(project, "priced", "smaller", ["x"])
    assert [s.stage_id for s in reduced.models["smaller"].pair_stages] == ["pair-1"]
    assert "pair-only" in reduced.models["smaller"].notes
