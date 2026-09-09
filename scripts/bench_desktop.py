"""Fresh-process readiness measurements for the local Svelte experiment.

Run from an installed checkout: python scripts/bench_desktop.py --runs 5.
OS disk caches are not flushed; this is not a first-install measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from easy_glm.desktop import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--project")
    args = parser.parse_args()
    elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        process = launch(args.project, open_browser=False)
        elapsed.append(time.perf_counter() - start)
        process.terminate()
        process.wait(timeout=10)
    print(
        json.dumps(
            {"ready_seconds": elapsed, "median": statistics.median(elapsed)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
