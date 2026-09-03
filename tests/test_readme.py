"""The release gate: every code block on the README actually runs (piece R11).

The owner's rule, verbatim: "everything shown on the README HAS TO WORK". In
0.3 the README examples did not all run and that cost trust, so this test is
what makes it a *build failure*, not a documentation nit, when a README block
stops working.

Two things are checked:

1. Every fenced ```python block in ``README.md``, extracted in order and run
   in **one shared namespace** (so the README can read like a tutorial —
   later blocks use ``df``, ``spec``, ``fit``, ``rm``, ``project`` etc. from
   earlier ones — exactly as a reader copy-pasting block after block would).
   A block fenced ```python skip-test`` needs a browser or the workbench
   server and is not run here; at most a handful of those are allowed, so a
   block cannot be quietly exempted from the gate by mislabelling it.
2. The actuarial lessons run in curriculum order. Later lessons use the saved
   model, rate tables and project produced by earlier lessons.

Both run from a temporary working directory with ``tests/fixtures`` linked in.
The tests replace the public-data loader with that local fixture, so no lesson
downloads during the build; every artifact lands in a directory pytest cleans
up.

Runtime budget: the whole module — README blocks plus every example — is
comfortably under the ~3 minute budget on the checked-in 50k-row fixture and
the small synthetic books the "large books" material uses; nothing here
downloads anything.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import pytest

import easy_glm

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BASIC_EXAMPLE = ROOT / "examples" / "basic_usage.py"
CURRICULUM = (
    ROOT / "examples" / "advanced_pipeline.py",
    ROOT / "examples" / "export_rate_tables.py",
    ROOT / "examples" / "score_new_data.py",
    ROOT / "examples" / "easy_glm_demo.py",
)

#: ```python fences, with an optional " skip-test" flag right after the
#: language tag (never inside the code, so this can't be spoofed by a comment).
_BLOCK_RE = re.compile(r"```python( skip-test)?\n(.*?)```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^]]*\]\((docs/images/[^)]+)\)")

#: How many ```python skip-test blocks the README is allowed to have. Small on
#: purpose: a block should only be exempted from running because it needs a
#: browser or a long-lived server, never because it was inconvenient to test.
MAX_SKIPPED_BLOCKS = 3


def _extract_blocks() -> list[tuple[bool, str]]:
    """``(skip, code)`` for every ```python fence in README.md, in order."""
    text = README.read_text(encoding="utf-8")
    return [(bool(flag), code) for flag, code in _BLOCK_RE.findall(text)]


def _link_fixtures(into: Path) -> None:
    """Make ``tests/fixtures`` resolve under ``into`` exactly as it does at
    the repository root, without copying the fixture data."""
    fixtures = into / "tests" / "fixtures"
    fixtures.parent.mkdir(parents=True, exist_ok=True)
    fixtures.symlink_to(ROOT / "tests" / "fixtures")


@pytest.fixture(scope="module")
def readme_blocks() -> list[tuple[bool, str]]:
    blocks = _extract_blocks()
    assert blocks, "found no ```python blocks in README.md — did the fences change?"
    return blocks


def test_readme_has_at_most_a_few_skip_test_blocks(readme_blocks):
    skipped = [code for skip, code in readme_blocks if skip]
    assert len(skipped) <= MAX_SKIPPED_BLOCKS, (
        f"{len(skipped)} README blocks are marked skip-test (max "
        f"{MAX_SKIPPED_BLOCKS}) — only a block that needs a browser or the "
        "workbench server should be exempted from actually running"
    )


def test_readme_embedded_images_exist():
    """GitHub-rendered first-lesson charts must be checked in with the README."""
    images = _IMAGE_RE.findall(README.read_text(encoding="utf-8"))
    assert images, "README has no embedded first-lesson chart assets"
    missing = [image for image in images if not (ROOT / image).is_file()]
    assert not missing, f"README references missing image assets: {missing}"


def test_readme_code_blocks_all_run(readme_blocks, tmp_path, monkeypatch):
    """Every non-skip block runs, in order, in one namespace, exactly as a
    reader copy-pasting the page top to bottom would run it."""
    _link_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        easy_glm,
        "load_external_dataframe",
        lambda: pl.read_parquet(ROOT / "tests/fixtures/french_motor_50k.parquet"),
    )
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(go.Figure, "show", lambda self: None)

    namespace: dict = {}
    t0 = time.perf_counter()
    run_count = 0
    for i, (skip, code) in enumerate(readme_blocks):
        if skip:
            continue
        run_count += 1
        try:
            exec(compile(code, f"<README.md block {i}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - re-raised with the source
            raise AssertionError(
                f"README.md python block {i} raised {exc.__class__.__name__}: "
                f"{exc}\n\n--- block {i} source ---\n{code}"
            ) from exc
    elapsed = time.perf_counter() - t0
    plt.close("all")
    print(f"\n{run_count} README blocks ran in {elapsed:.1f}s")
    assert elapsed < 120, f"README blocks took {elapsed:.1f}s (budget: 120s)"


def _run_lesson(script: Path, tmp_path: Path, *, use_local_public_data: bool) -> None:
    """Run one real curriculum script, with non-interactive local test data."""
    setup = (
        "import matplotlib.pyplot as plt\n"
        "import plotly.graph_objects as go\n"
        "plt.show = lambda: None\n"
        "go.Figure.show = lambda self: None\n"
    )
    if use_local_public_data:
        setup += (
            "import polars as pl\n"
            "import easy_glm\n"
            f"easy_glm.load_external_dataframe = lambda: pl.read_parquet({str(ROOT / 'tests/fixtures/french_motor_50k.parquet')!r})\n"
        )
    setup += f"import runpy\nrunpy.run_path({str(script)!r}, run_name='__main__')\n"
    result = subprocess.run(
        [sys.executable, "-c", setup],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )


def test_basic_example_runs_with_local_public_data(tmp_path):
    """The beginner script stays linear; the test supplies its cached dataset."""
    code = (
        "import runpy\n"
        "import matplotlib.pyplot as plt\n"
        "import polars as pl\n"
        "import plotly.graph_objects as go\n"
        "import easy_glm\n"
        "plt.show = lambda: None\n"
        "go.Figure.show = lambda self: None\n"
        f"easy_glm.load_external_dataframe = lambda: pl.read_parquet({str(ROOT / 'tests/fixtures/french_motor_50k.parquet')!r})\n"
        f"runpy.run_path({str(BASIC_EXAMPLE)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"basic_usage.py exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )


def test_actuarial_examples_run_in_curriculum_order(tmp_path):
    """Each lesson consumes the real artifacts produced by the earlier ones."""
    advanced, export, score, demo = CURRICULUM
    _run_lesson(advanced, tmp_path, use_local_public_data=True)

    fitted = tmp_path / "french_motor_model"
    assert fitted.is_dir()
    assert (fitted / "config.json").is_file()
    assert (fitted / "spec.json").is_file()
    assert (fitted / "glm_model.joblib").is_file()
    assert (fitted / "rate_model.json").is_file()
    assert (fitted / "rate_tables").is_dir()
    assert (tmp_path / "french_motor.easyglm").is_file()

    _run_lesson(export, tmp_path, use_local_public_data=False)
    assert (tmp_path / "french_motor_rate_tables.xlsx").is_file()

    _run_lesson(score, tmp_path, use_local_public_data=False)

    _run_lesson(demo, tmp_path, use_local_public_data=True)
    project_path = tmp_path / "french_motor_project.json"
    assert project_path.is_file()
    from easy_glm.workflow import Project

    assert Project.from_json(project_path).validate("Frequency") == []
