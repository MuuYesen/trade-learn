# Stage 0 Provider Readiness Design

Date: 2026-04-25

Source of truth:

- `docs/PROGRESS.md`
- `docs/VISION.md`
- `docs/specs/CONSISTENCY.md`

## Scope

Wire the oracle provider dependencies needed for frozen 1.x oracle checks and golden dataset generation.

This work does not put oracle providers back into the default trade-learn 2.0 dependency surface. They are only available through a development dependency group used by oracle readiness and golden generation.

## Dependency Group

Add a uv dependency group named `oracle`:

- `opentdx`
- `tvdatafeed @ git+https://github.com/rongardF/tvdatafeed.git`

The group is installed with:

```bash
uv sync --group oracle --extra dev
```

## Diagnostic Behavior

When `scripts/check_oracle.py` finds missing providers, it should keep returning success if the frozen Query can be loaded, but print an explicit install hint:

```text
hint=uv sync --group oracle --extra dev
```

This keeps oracle import readiness distinct from external provider availability.

## CI

Add an oracle readiness CI step that syncs the oracle group and runs:

```bash
uv run python scripts/check_oracle.py
```

The check remains read-only and does not fetch remote market data.

## Progress Update

Update `docs/PROGRESS.md` so provider readiness is no longer a vague local failure. It should state that the oracle dependency group is wired and that real dataset generation still requires provider imports plus live provider access.

## Verification

Run:

- `uv run pytest tests/golden/test_oracle.py`
- `uv run pytest`
- `uv run ruff check tradelearn/core tests/unit/core tests/golden tests/consistency scripts`
- `uv run interrogate tradelearn/core --fail-under 90`
- `uv sync --group oracle --extra dev`
- `uv run python scripts/check_oracle.py`
