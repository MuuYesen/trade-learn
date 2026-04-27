# Stage 0 Oracle Readiness Design

Date: 2026-04-25

Source of truth:

- `docs/PROGRESS.md`
- `docs/specs/CONSISTENCY.md`
- `docs/PROJECT.md`

## Scope

Prepare the frozen `reference/tradelearn_1x` oracle for real Stage 0 golden generation.

This work verifies import readiness and gives clear diagnostics before any remote data fetch or expected JSON generation is attempted. It does not edit frozen reference source code and does not generate fake golden data.

## Diagnostic Script

Add `scripts/check_oracle.py` with a read-only CLI:

- Confirms `reference/tradelearn_1x` exists.
- Confirms `reference/tradelearn_1x/query/query.py` exists.
- Imports `tradelearn_1x.query.Query` through an isolated `sys.path`.
- Reports optional provider availability for the 1.x engines without fetching data.
- Exits non-zero with a clear message when the oracle cannot be imported.

## Builder Hardening

Move shared oracle loading behavior into `scripts/build_golden.py` helper functions so the builder and diagnostic script use the same path assumptions.

The builder must prefer `reference/` over the active `tradelearn` package when loading 1.x oracle modules. If 1.x imports fail because optional provider dependencies are missing, the error must identify the missing module.

## Tests

Add `tests/golden/test_oracle.py`:

- Validates the reference layout.
- Exercises the diagnostic CLI.
- Verifies `build_golden.py --dry-run` stays read-only.
- Verifies failed default generation writes no fake expected JSON.

## Progress Update

Update `docs/PROGRESS.md` to split remaining Stage 0 goldens into:

- oracle import readiness
- external provider/data availability
- expected generation adapter

Mark only the readiness item complete if local diagnostics pass.

## Verification

Run:

- `uv run pytest tests/golden/test_oracle.py tests/golden/test_manifest.py`
- `uv run pytest`
- `uv run ruff check tradelearn/core tests/unit/core tests/golden tests/consistency scripts`
- `uv run interrogate tradelearn/core --fail-under 90`
- `python scripts/check_oracle.py`
