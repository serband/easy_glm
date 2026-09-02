"""``python -m easy_glm.app [project.json] [--port N] [--headless]``"""

from __future__ import annotations

import argparse

from easy_glm.app import launch


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m easy_glm.app", description="easy_glm workbench"
    )
    parser.add_argument("project", nargs="?", help="project JSON to open")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--headless", action="store_true", help="do not open a browser tab"
    )
    args = parser.parse_args()
    launch(args.project, port=args.port, block=True, headless=args.headless)


if __name__ == "__main__":
    main()
