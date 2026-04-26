#!/usr/bin/env python
"""Run the Stage 9 release golden gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_golden_readiness import (  # noqa: E402
    DATASETS_DIR,
    EXPECTED_DIR,
)
from scripts.check_golden_readiness import (  # noqa: E402
    build_report as build_readiness_report,
)
from scripts.compare_golden import compare as compare_golden  # noqa: E402


class PytestResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


def run_pytest(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "ARROW_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def build_report(
    *,
    engine: str = "tv",
    datasets_root: Path = DATASETS_DIR,
    expected_root: Path = EXPECTED_DIR,
    rtol: float = 1e-4,
    skip_pytest: bool = False,
    pytest_args: list[str] | None = None,
    pytest_runner: Any = run_pytest,
) -> dict[str, Any]:
    """Build the release golden gate report from readiness, compare, and pytest."""
    readiness = build_readiness_report(
        datasets_root=datasets_root,
        expected_root=expected_root,
        engine=engine,
    )
    comparison = compare_golden(
        engine=engine,
        datasets_root=datasets_root,
        expected_root=expected_root,
        rtol=rtol,
    )
    if skip_pytest:
        pytest_payload: dict[str, Any] = {"skipped": True, "returncode": 0}
    else:
        args = pytest_args or ["tests/golden", "-q"]
        result: PytestResult = pytest_runner(args)
        pytest_payload = {
            "skipped": False,
            "args": args,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return {
        "ok": bool(
            readiness["ok"]
            and comparison["ok"]
            and pytest_payload["returncode"] == 0
        ),
        "engine": engine,
        "readiness": readiness,
        "comparison": comparison,
        "pytest": pytest_payload,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the release golden gate.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--engine", choices=["tv"], default="tv")
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_DIR)
    parser.add_argument("--expected-root", type=Path, default=EXPECTED_DIR)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--skip-pytest", action="store_true")
    return parser.parse_args(argv)


def print_text(payload: dict[str, Any]) -> None:
    readiness = payload["readiness"]["summary"]
    comparison = payload["comparison"]["summary"]
    pytest_result = payload["pytest"]
    print(
        "release-golden:"
        f"ok={payload['ok']}"
        f" engine={payload['engine']}"
        f" datasets={readiness['datasets_ready']}/{readiness['datasets_total']}"
        f" expected={readiness['expected_ready']}/{readiness['expected_total']}"
        f" compared={comparison['compared']}"
        f" failed={comparison['failed']}"
        f" pytest={pytest_result['returncode']}"
    )
    for failure in payload["comparison"].get("failures", [])[:5]:
        print(f"failure:{failure.get('reason', 'unknown')}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_report(
        engine=args.engine,
        datasets_root=args.datasets_root,
        expected_root=args.expected_root,
        rtol=args.rtol,
        skip_pytest=args.skip_pytest,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_text(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
