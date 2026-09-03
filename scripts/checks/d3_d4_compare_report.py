"""Actuarial check for D3 / D4 — comparing two models and the HTML report.

Starts the workbench on the French-motor fixture with two frequency models
(``freq_v1``: plain step terms; ``freq_v2``: the same plus a DrivAge × VehPower
interaction and Density as a piecewise-linear term), fits both, photographs the
Compare page and the downloaded HTML report with Playwright, and writes a
plain-language page explaining what to look at when comparing two models.

Usage: python scripts/checks/d3_d4_compare_report.py [--write]
  --write regenerates docs/checks/d3-d4-compare-report.md and
  docs/checks/img/d3_*.png / d4_*.png; otherwise the document is printed.
  Screenshots need Playwright: either importable here, or an interpreter with
  it in EASY_GLM_PLAYWRIGHT_PYTHON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"
DOC = ROOT / "docs" / "checks" / "d3-d4-compare-report.md"
IMG = ROOT / "docs" / "checks" / "img"
DRIVER = ROOT / "scripts" / "checks" / "_d3_screens.py"
PREDICTORS = ["DrivAge", "VehAge", "BonusMalus", "Density", "VehPower", "Region"]
#: pictures must stay small enough to live in the repository
MAX_IMAGE_KB = 300


def _model(interaction: bool) -> dict:
    cfg = {
        "family": "poisson",
        "target": "ClaimNb",
        "weight": "Exposure",
        "divide_target_by_weight": True,
        "predictors": PREDICTORS,
        "penalty": {"alpha": 0.0005, "cv": None},
    }
    if interaction:
        cfg["interactions"] = [
            {"a": "DrivAge", "b": "VehPower", "min_cell_exposure": 0.005}
        ]
    return cfg


def _project(folder: Path) -> Path:
    project = {
        "name": "d3check",
        "version": 2,
        "data": {
            "source": {"type": "parquet", "path": str(FIXTURE), "options": {}},
            "roles": {
                "IDpol": "id",
                "ClaimNb": "target",
                "Exposure": "weight",
                **dict.fromkeys(PREDICTORS, "predictor"),
            },
            "split": {
                "mode": "random",
                "column": "traintest",
                "fraction": 0.7,
                "seed": 7,
            },
        },
        "design": {"variables": {"Density": {"kind": "linear"}}},
        "models": {"freq_v1": _model(False), "freq_v2": _model(True)},
        "champion": "freq_v1",
    }
    path = folder / "d3check.easyglm-project.json"
    path.write_text(json.dumps(project, indent=2))
    return path


def _playwright_python() -> str | None:
    try:
        import playwright  # noqa: F401

        return sys.executable
    except ImportError:
        cand = os.environ.get("EASY_GLM_PLAYWRIGHT_PYTHON")
        if cand and Path(cand).exists():
            return cand
    return None


def _screens(out: Path) -> tuple[bool, str]:
    py = _playwright_python()
    if py is None:
        return (
            False,
            "Playwright not available (set EASY_GLM_PLAYWRIGHT_PYTHON); text only.",
        )
    folder = Path(tempfile.mkdtemp(prefix="d3check_"))
    project = _project(folder)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "src/easy_glm/app/main.py"),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--",
            f"--project={project}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{port}/_stcore/health", timeout=2
                ) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        res = subprocess.run(
            [py, str(DRIVER), f"http://localhost:{port}", str(out)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        ok = res.returncode == 0 and "DONE" in res.stdout
        return ok, (res.stdout + res.stderr)[-2000:]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(folder, ignore_errors=True)


DOC_TEXT = """# D3 / D4 — comparing two models, and the report you can send

*French-motor fixture (50,000 policies, 70/30 random split). Two frequency
models on the same six factors: `freq_v1` uses plain banded (step) terms;
`freq_v2` adds a DrivAge × VehPower interaction and treats Density as a
straight line in log space; the report shown here is `freq_v2`'s, with `freq_v1`
as its challenger. Screenshots regenerated by
`scripts/checks/d3_d4_compare_report.py --write`.*

## Why this exists

Until now the workbench could show you one model at a time. Deciding whether a
change is an improvement meant writing numbers on paper and flipping between
screens. There are now three things instead:

1. a **"Compare with"** box in the sidebar — pick a challenger once and the
   Diagnostics page, the Rate tables page and the Compare page all use it;
2. a **Compare** page — the two models' numbers next to each other, the same
   charts with both models drawn on them, and a table of *which relativities
   actually differ*;
3. a **Download HTML report** button on the Export page — one file with the
   whole model written up, which you can email or attach to a filing.

## Compare — the numbers side by side

![metrics](img/d3_compare_metrics.png)

What you see: one column per model, split into the training rows and the
holdout rows, and then the facts about each model (its penalty `alpha`, how
many terms survived it, its interactions, its linear terms, how many manual
adjustments it carries and its base rate).

What to look at, in this order:

* **Holdout first.** Both models saw the training rows while they were being
  fitted, so a model that looks better there may simply have memorised noise.
  The holdout rows are the honest comparison.
* **A/E** should sit at 1.00 on the training rows for both (the fit makes it
  so). On the holdout, a number away from 1.00 is a level problem: the model
  charges too much or too little *overall*.
* **Gini** says how well the model *orders* the risks — nothing about the
  level. Higher is better; the scale is normalised so 1.00 would be a perfect
  ordering of these rows. Differences below about 0.005 on 15,000 holdout rows
  are inside the noise; treat them as a tie.
* **Deviance explained** and **mean deviance** are the statistical measures of
  fit. They usually move with Gini; when they disagree, believe the one that
  matches the decision you are making (Gini for segmentation, deviance for
  overall accuracy).
* **Non-zero terms** is how complicated the model is. A challenger that buys
  0.002 of Gini with 40 more terms is usually not worth filing.

## Compare — the same picture with both models on it

![A/E with both models](img/d3_compare_ae.png)

What you see: for any variable — in either model or in neither — the observed
rate (blue), the champion's expected rate (orange) and the challenger's
(green, dashed), with exposure as bars behind them.

What to look at: the bands where the two model lines separate **and** there is
real exposure. That is where the two models would actually charge different
premiums. Where the challenger's line sits closer to the blue line across
several neighbouring bands, it is genuinely better on that factor; a single
band that fits better with little exposure behind it is noise.

The **Lift** tab shows each model against the observed rate over ten
equal-exposure bins, and the **Double lift** tab sorts the book by how much
cheaper the champion is than the challenger — the model whose A/E stays closer
to 1.00 across those bins is the one getting the disputed policies right.

## Compare — which relativities actually differ

![relativity diff](img/d3_compare_diff.png)

This is the table to take to a rate meeting. Every row is **one band of one
factor whose relativity moved**, plus the tolerance box that decides what
counts as a move.

* `log_diff` is log(challenger ÷ champion). It is on the log scale because
  relativities multiply: **+0.10 means the challenger charges about 10 % more**
  for that band, −0.10 about 10 % less. The table is sorted by the size of the
  move, so the top row is the biggest change in the whole rate structure.
* The tolerance (default 0.01, i.e. 1 %) is what a band has to move by before
  it is listed. Two models that were fitted separately always differ in the
  sixth decimal; 1 % is the level at which a difference is worth a sentence.
  Raise it to 0.05 to see only the changes that would move a premium
  noticeably.
* **Bands are matched by their label** — the same `[28.0, 30.0)` you see on the
  rate table. If the challenger's knots moved, its bands do not line up with
  the champion's, and the table says *band only in freq_v1* / *band only in
  freq_v2* rather than inventing a comparison. That is the honest answer: the
  two models cut the factor differently and their relativities are not
  comparable band by band.
* A factor one model has and the other does not (here Density is linear in one
  and banded in the other, and the interaction exists only in `freq_v2`) is
  listed once as *only in …*.
* **`(base rate)`** is compared too. Two models can have identical relativities
  and still charge different premiums because their base rate differs; that row
  is the overall level.

Two identical models give an **empty** table — that is the test, and it is what
you should see if you compare a model with itself.

The **Make … champion** buttons at the top promote either model; the champion
is what the rest of the workbench (and the exported script) defaults to.

## The HTML report

![report summary](img/d4_report_summary.png)

The Export page has a **Download HTML report** button. It produces *one file* —
no folder of images, nothing fetched from the internet when it is opened, which
means it works on a machine with no network, from a shared drive, or in five
years' time. Open it by double-clicking; it prints sensibly too.

What is in it:

* **Summary** — the data file, how many rows survived the filters, the split,
  and the model's family, target, weight, offset, penalty and base rate;
  then the metrics table (both models when a challenger is chosen).
* **One block per rating factor** — the fitted relativities, actual against
  expected on the training rows *and* on the holdout, and the rate table
  itself.

![report factor](img/d4_report_variable.png)

* **Interactions** — the cell adjustments as a heatmap (white is 1.00, i.e. no
  adjustment; grey is a cell with nothing in it), then actual over expected in
  the same cells on train and on holdout, and a list of just the cells that
  carry an adjustment.

![report interaction](img/d4_report_interaction.png)

* **Lift and Gini** on both sets of rows.
* **The comparison section**, when a challenger was chosen: the double lift and
  the same relativity-difference table.

![report comparison](img/d4_report_compare.png)

* **Appendix** — every fitted coefficient, and the Python script that rebuilds
  the model from scratch. That script is the audit trail: anyone with the data
  can re-run it and get this model back.

The charts in the report are drawn as plain pictures rather than as an
interactive charting library, so the file stays a few hundred kilobytes instead
of five megabytes. Hovering a band or a heatmap cell still shows its numbers.

## What was checked

* Two identical models produce an empty difference table; a single edited
  relativity produces exactly one row, with the right size of change.
* A model with a factor the other does not have lists that factor once.
* The report contains no link to anything outside itself, opens in a browser
  with no errors, has one section per rating factor, and only carries the
  comparison section when a challenger was chosen.

## Questions for you

1. **Tolerance.** The difference table defaults to 1 % (|log ratio| > 0.01).
   Is that the level at which you would want a band listed, or would you rather
   start at 5 % and drill down?
2. **Base rate row.** We list the base rate as a row of the difference table so
   a pure level change cannot hide. Would you rather see it as a separate
   headline ("overall level: +2.4 %")?
3. **Report audience.** The report currently leads with the data and the model
   configuration and puts the coefficients in an appendix. For a filing, would
   you want a one-page summary at the front (headline metrics, the rate tables,
   nothing else) with everything else after it?
4. **Moved knots.** When the challenger's bands do not line up with the
   champion's we refuse to compare them and say so. The alternative is to
   compare the two curves at a set of common points (say every band edge of
   either model). Which would you rather have?
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write:
        IMG.mkdir(parents=True, exist_ok=True)
        ok, log = _screens(IMG)
        print(log)
        if not ok:
            print(
                "screenshots failed or unavailable; document written without new images"
            )
        DOC.write_text(DOC_TEXT)
        for f in sorted([*IMG.glob("d3_*.png"), *IMG.glob("d4_*.png")]):
            size = f.stat().st_size // 1024
            flag = "  <-- OVER BUDGET" if size > MAX_IMAGE_KB else ""
            print(f"{f.name}: {size} KB{flag}")
        print(f"wrote {DOC}")
    else:
        print(DOC_TEXT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
