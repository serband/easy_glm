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

Code style and conventions
- Imports: standard library first, third-party second, local imports last; group with blank lines.
- Formatting: adhere to Black; line length 88; trailing commas where helpful.
- Types: use type hints everywhere; `from __future__ import annotations` where possible.
- Naming: descriptive variable/function/class names; avoid abbreviations; class names in PascalCase; functions in snake_case.
- Error handling: raise specific exceptions; avoid bare `except:`; include meaningful messages.
- Tests: small, fast unit tests; use `pytest`; follow existing test style in `tests/`.
- Documentation: docstrings for public API; comments sparing but clear.

Cursor and Copilot rules
- Cursor rules: see `.cursor/rules/` or `.cursorrules` for guidance and include them here if present.
- Copilot rules: include any guidelines from `.github/copilot-instructions.md` if present.

Notes
- This AGENTS.md is lightweight and focused on quick, repeatable steps for agents.
