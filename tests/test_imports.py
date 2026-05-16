import os
import subprocess
import sys


def test_import_easy_glm_does_not_import_matplotlib():
    code = (
        "import sys; "
        "import easy_glm; "
        "raise SystemExit(1 if 'matplotlib' in sys.modules else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0
