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
2. Every ``examples/*.py`` runs standalone, via ``subprocess``, exit code 0.

Both run from a temporary working directory with ``tests/fixtures`` linked in,
so the README's own ``DATA = "tests/fixtures/french_motor_50k.parquet"`` line
resolves without touching the repository checkout, and every file a block or
an example writes (a project JSON, an exported script, a `.easyglm`, an
`.xlsx`) lands in a directory pytest cleans up.

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

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))

#: ```python fences, with an optional " skip-test" flag right after the
#: language tag (never inside the code, so this can't be spoofed by a comment).
_BLOCK_RE = re.compile(r"```python( skip-test)?\n(.*?)```", re.DOTALL)

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


def test_readme_code_blocks_all_run(readme_blocks, tmp_path, monkeypatch):
    """Every non-skip block runs, in order, in one namespace, exactly as a
    reader copy-pasting the page top to bottom would run it."""
    _link_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)

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
    print(f"\n{run_count} README blocks ran in {elapsed:.1f}s")
    assert elapsed < 120, f"README blocks took {elapsed:.1f}s (budget: 120s)"


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(example, tmp_path):
    """Every examples/*.py runs standalone (subprocess, exit code 0). Each
    example resolves its own data file from ``__file__``, so it needs no
    fixtures linked into the working directory — only a clean place to write
    the files it produces (a saved model, a `.easyglm`, a spec JSON)."""
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{example.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )


def test_post_fit_examples_use_the_saved_scorer(tmp_path):
    """The review and scoring lessons consume the first lesson's artefact."""
    basic = ROOT / "examples" / "basic_usage.py"
    review = ROOT / "examples" / "exploring_fit.py"
    scoring = ROOT / "examples" / "scoring_editor.py"

    for example, args in (
        (basic, []),
        (review, ["my_model.easyglm"]),
        (scoring, ["my_model.easyglm"]),
    ):
        result = subprocess.run(
            [sys.executable, str(example), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"{example.name} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )

    assert (tmp_path / "review_copy.easyglm").exists()
