# Contributing to easy_glm

Thanks for taking the time to contribute!

## Quick Start
1. Fork the repo & clone your fork
2. Create a virtual environment & install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Create a feature branch: `git checkout -b feat/my-feature`
4. Run tests & linters before committing:
   ```bash
   ruff check .
   black .
   pytest
   ```
5. Commit with a clear message and open a Pull Request (PR)

## Development Guidelines
* Keep functions small & focused
* Add / update tests for any behavioral change
* Maintain type hints where practical
* Prefer Polars operations over Pandas for performance-critical paths
* Avoid adding heavy dependencies without discussion first

## Testing
Run the full test suite:
```bash
pytest -q
```
Generate coverage:
```bash
pytest --cov=easy_glm --cov-report=term-missing
```

## Commit Messages
Use short (<72 char) imperative subject line. Optionally follow Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).

## Pull Requests
Checklist before opening:
- [ ] Tests pass locally
- [ ] New / changed logic is covered by tests
- [ ] Docs / README updated if API changes
- [ ] No stray debug prints
- [ ] No local handovers, agent reviews or one-off verification reports; keep them in gitignored `.internal/`

## Releasing
Releases require the owner's explicit authorization. Every pushed `v*` tag
automatically publishes to PyPI through GitHub Actions Trusted Publishing.

1. Verify the remote base and latest public version, finish review and CI, and
   update `pyproject.toml` and `CHANGELOG.md`.
2. Build a wheel, install it in a fresh environment, and run the public examples
   against that installation.
3. After authorization, merge the reviewed change into `main`, create its
   release tag, and push that specific tag. Confirm the publish workflow succeeds.

## Code Style
Configured via Black & Ruff (see `pyproject.toml`). If in doubt, run the formatters.

## Questions?
Open an issue with the label `question`.

Happy hacking!
