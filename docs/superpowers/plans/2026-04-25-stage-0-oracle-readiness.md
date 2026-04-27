# Stage 0 Oracle Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and harden the frozen 1.x oracle path required for Stage 0 golden generation.

**Architecture:** Keep oracle diagnostics in `scripts/check_oracle.py`; keep shared path/import helpers in `scripts/build_golden.py`. Tests stay in `tests/golden/` because oracle readiness is part of the golden baseline.

**Tech Stack:** Python argparse/importlib, pytest, uv, ruff.

---

## Tasks

### Task 1: Oracle Readiness Tests

**Files:**
- Create: `tests/golden/test_oracle.py`

- [ ] **Step 1: Write failing tests**

Add tests that require `scripts/check_oracle.py`, check reference layout, run diagnostic CLI, and assert failed generation does not write JSON.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/golden/test_oracle.py -q`

Expected: FAIL because `scripts/check_oracle.py` does not exist.

### Task 2: Diagnostic CLI and Shared Import Helpers

**Files:**
- Create: `scripts/check_oracle.py`
- Modify: `scripts/build_golden.py`

- [ ] **Step 1: Implement import helpers**

Add reference path insertion and missing-module diagnostics in `build_golden.py`.

- [ ] **Step 2: Implement diagnostic CLI**

Add `check_oracle.py` using the builder helpers. It prints `oracle=ok` and provider statuses when import succeeds.

- [ ] **Step 3: Verify green**

Run: `uv run pytest tests/golden/test_oracle.py tests/golden/test_manifest.py -q`

Expected: PASS.

### Task 3: Progress and Verification

**Files:**
- Modify: `docs/PROGRESS.md`

- [ ] **Step 1: Update progress**

Split remaining Stage 0 blockers into oracle readiness, external providers, and expected adapter.

- [ ] **Step 2: Run all checks**

Run:

```bash
uv run pytest
uv run ruff check tradelearn/core tests/unit/core tests/golden tests/consistency scripts
uv run interrogate tradelearn/core --fail-under 90
python scripts/check_oracle.py
```

- [ ] **Step 3: Commit scoped changes**

Commit only oracle readiness files and `docs/PROGRESS.md`.
