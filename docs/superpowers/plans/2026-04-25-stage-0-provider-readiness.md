# Stage 0 Provider Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire oracle provider dependencies for frozen 1.x oracle checks without changing the default 2.0 install surface.

**Architecture:** Provider dependencies live in a uv dependency group named `oracle`. `scripts/check_oracle.py` remains read-only and reports missing providers plus an install hint. CI verifies the group can be synced and the oracle diagnostic can run.

**Tech Stack:** Python argparse/import diagnostics, uv dependency groups, GitHub Actions, pytest.

---

## Tasks

### Task 1: Tests

**Files:**
- Modify: `tests/golden/test_oracle.py`

- [ ] **Step 1: Write failing tests**

Add tests that read `pyproject.toml` and assert `[dependency-groups].oracle` includes `opentdx` and `tvdatafeed`. Add a CLI test that expects `hint=uv sync --group oracle --extra dev` when providers are missing.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/golden/test_oracle.py -q`

Expected: FAIL because the dependency group and hint are not implemented yet.

### Task 2: Implementation

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/check_oracle.py`

- [ ] **Step 1: Add dependency group**

Add `[dependency-groups] oracle = [...]` with oracle provider packages.

- [ ] **Step 2: Add diagnostic hint**

Print the uv sync command when any provider is missing.

- [ ] **Step 3: Add CI oracle check**

Add a read-only CI job or step that runs `uv sync --group oracle --extra dev` and `uv run python scripts/check_oracle.py`.

### Task 3: Progress, Verify, Commit

**Files:**
- Modify: `docs/PROGRESS.md`

- [ ] **Step 1: Update progress**

Record that oracle dependency group wiring is complete and real dataset generation still requires provider import/live access verification.

- [ ] **Step 2: Run checks**

Run:

```bash
uv run pytest
uv run ruff check tradelearn/core tests/unit/core tests/golden tests/consistency scripts
uv run interrogate tradelearn/core --fail-under 90
uv sync --group oracle --extra dev
uv run python scripts/check_oracle.py
```

- [ ] **Step 3: Commit scoped changes**

Commit only provider readiness code/test/config files; keep `docs/PROGRESS.md` local and unstaged.
