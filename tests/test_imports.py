import os
import subprocess
import sys

import pytest

# Modules that `import easy_glm` must NOT pull in (optional extras / lazy imports).
LAZY = ["matplotlib", "seaborn", "duckdb", "rdata", "streamlit", "plotly"]


@pytest.mark.parametrize("module", LAZY)
def test_import_easy_glm_is_lazy_about(module):
    code = (
        "import sys; "
        "import easy_glm; "
        f"raise SystemExit(1 if {module!r} in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, f"import easy_glm imported {module}"
