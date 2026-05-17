## AGENTS.md

Purpose: Provide build/test commands and code style guidelines for AI agents operating in this repo.

Build, lint, and tests
- Single test: `pytest tests/test_blueprint.py -k test_blueprint_specific --maxfail=1 -q`
  - Replace with exact test name or use `pytest -k <name>` for a quick run.
- Full suite: `pytest -q`
- Lint: `ruff check .`
- Format: `black .`
- Type checks (if pyproject includes mypy): `mypy .`  # optional if configured
- Run all quality steps: `black . && ruff check . && pytest -q`

Installation
- `python setup_dev.py` — handles editable install + symlink fallback for Python 3.14
- If `import easy_glm` fails: `ln -sf $(pwd)/src/easy_glm .venv/lib/python*/site-packages/easy_glm`
- After source changes in engine/ or core/: `pip install .` (non-editable) or use the symlink approach
- `PYTHONPATH=src` also works as a quick workaround

Code style and conventions
- Imports: standard library first, third-party second, local imports last; group with blank lines.
- Formatting: adhere to Black; line length 88; trailing commas where helpful.
- Types: use type hints everywhere; `from __future__ import annotations` where possible.
- Naming: descriptive variable/function/class names; avoid abbreviations; class names in PascalCase; functions in snake_case.
- Error handling: raise specific exceptions; avoid bare `except:`; include meaningful messages.
- Tests: small, fast unit tests; use `pytest`; follow existing test style in `tests/`.
- Documentation: docstrings for public API; comments sparing but clear.

Module layout
- `src/easy_glm/core/` — Blueprint, prepare, model fitting, rate tables, EasyGLM pipeline
- `src/easy_glm/engine/` — RateModel (From/To/Relativity tables, versioning, .easyglm JSON export, scoring with exposure). Key files: `rate_model.py` (predict, snapshots, serialisation), `_scoring.py` (np.searchsorted fast path), `models.py` (dataclasses)
- `src/easy_glm/ui/` — Streamlit relativity editor (optional, requires `ui` extras)
- `tests/test_engine.py` — RateModel tests (prediction, snapshots, JSON roundtrip, exposure)
- `tests/test_scoring.py` — Isolated scoring tests (numeric searchsorted, categorical lookups)
- `examples/basic_usage.py` — Full pipeline demo including .easyglm export, scoring, A/E plotting

Cursor and Copilot rules
- Cursor rules: see `.cursor/rules/` or `.cursorrules` for guidance and include them here if present.
- Copilot rules: include any guidelines from `.github/copilot-instructions.md` if present.

Notes
- This AGENTS.md is lightweight and focused on quick, repeatable steps for agents.
