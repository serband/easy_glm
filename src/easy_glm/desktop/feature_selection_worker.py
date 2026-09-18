"""Isolated feature selection worker using pipes, without temporary files."""

from __future__ import annotations

import os
import sys
from typing import Any

from easy_glm.desktop.feature_selection_ipc import read_request, write_message


def main() -> None:
    # Reserve the original stdout pipe for messages. Redirect both Python and
    # native-library console output so it cannot corrupt the protocol.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def progress(value: dict[str, Any]) -> None:
            write_message(output, {"type": "progress", "value": value})

        try:
            from easy_glm.workflow.feature_selection import select_variables

            project, raw, options = read_request(sys.stdin.buffer)
            result = select_variables(project, raw, **options, progress=progress)
        except Exception as exc:  # Worker boundary: expose only an actionable error.
            result = {"error": str(exc)}
        write_message(output, {"type": "result", "value": result})


if __name__ == "__main__":
    main()
