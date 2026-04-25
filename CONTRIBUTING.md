# Contributing

trade-learn 2.0 development follows the project documents in `docs/`.

## Source of Truth

- Implement only behavior that is traceable to `docs/specs/`.
- Keep Stage 0 limited to foundation and golden baseline scaffolding.
- Record acceptable behavioral differences in `docs/specs/MIGRATION.md`.

## Local Checks

```bash
python -m pytest
python -m compileall tradelearn scripts tests
```

CI also runs ruff, doctests, and interrogate when the corresponding dev tools are installed.

## Golden Data

Do not edit `tests/golden/expected/` to make failing tests pass. Golden changes require a documented reason in `docs/specs/MIGRATION.md`.
