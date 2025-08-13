# Contributing to easy_glm

Thanks for taking the time to contribute!

## Quick Start
1. Fork the repo & clone your fork
2. Create a virtual environment & install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
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

## Releasing
1. Bump version in `pyproject.toml`
2. Update `CHANGELOG.md` (create if missing)
3. Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Push tags: `git push --tags`
5. Publish (e.g. `pipx run build && pipx run twine upload dist/*`)

## Code Style
Configured via Black & Ruff (see `pyproject.toml`). If in doubt, run the formatters.

## Questions?
Open an issue with the label `question`.

Happy hacking!
