#!/usr/bin/env python
"""Compare current Stage 3 golden runs against expected v1.0 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_golden import (  # noqa: E402
    filter_manifest_by_engine,
    load_manifest,
    planned_jobs,
    run_expected_job,
)
from scripts.check_golden_readiness import (  # noqa: E402
    DATASETS_DIR,
    EXPECTED_DIR,
    build_report,
    expected_path,
)

COMPARABLE_SUMMARY_KEYS = {
    "annual_return",
    "avg_trade_pct",
    "bars",
    "expectancy",
    "final_cash",
    "final_realized_pnl",
    "final_value",
    "max_drawdown",
    "peak_value",
    "profit_factor",
    "return_pct",
    "sharpe",
    "sqn",
    "total_fills",
    "total_orders",
    "total_trades",
    "win_rate_pct",
}


def _load_expected(strategy: str, dataset: str, expected_root: Path) -> dict[str, Any]:
    path = expected_path(strategy, dataset, expected_root)
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_value(payload: dict[str, Any], key: str) -> float:
    value = payload.get("summary", {}).get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"summary.{key} must be numeric")
    return float(value)


def _numeric_summary(payload: dict[str, Any]) -> dict[str, float]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric[key] = float(value)
    return numeric


def _trade_signature(payload: dict[str, Any]) -> list[tuple[str, float, float, bool, bool]]:
    return [
        (
            str(trade["datetime"]),
            float(trade["size"]),
            float(trade["price"]),
            bool(trade["isopen"]),
            bool(trade["isclosed"]),
        )
        for trade in payload.get("trades", [])
    ]


def _equity_signature(payload: dict[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(row["datetime"]), float(row["value"]))
        for row in payload.get("equity", [])
    ]


def _equity_differs(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float,
) -> bool:
    actual_equity = _equity_signature(actual)
    expected_equity = _equity_signature(expected)
    if len(actual_equity) != len(expected_equity):
        return True
    for (actual_dt, actual_value), (expected_dt, expected_value) in zip(
        actual_equity,
        expected_equity,
        strict=True,
    ):
        if actual_dt != expected_dt:
            return True
        if not math.isclose(actual_value, expected_value, rel_tol=rtol, abs_tol=rtol):
            return True
    return False


def _compare_job(
    *,
    strategy: str,
    dataset: dict[str, str],
    datasets_root: Path,
    expected_root: Path,
    rtol: float,
) -> dict[str, Any] | None:
    dataset_symbol = dataset["symbol"]
    expected = _load_expected(strategy, dataset_symbol, expected_root)
    actual = run_expected_job(strategy, dataset, datasets_root)
    try:
        trades_differ = _trade_signature(actual) != _trade_signature(expected)
    except (KeyError, TypeError, ValueError):
        trades_differ = True
    if trades_differ:
        return {
            "strategy": strategy,
            "dataset": dataset_symbol,
            "reason": "trades differ",
        }
    try:
        equity_differs = _equity_differs(actual, expected, rtol=1e-6)
    except (KeyError, TypeError, ValueError):
        equity_differs = True
    if equity_differs:
        return {
            "strategy": strategy,
            "dataset": dataset_symbol,
            "reason": "equity differs",
        }
    actual_summary = _numeric_summary(actual)
    expected_summary = _numeric_summary(expected)
    summary_keys = sorted(
        set(actual_summary) & set(expected_summary) & COMPARABLE_SUMMARY_KEYS
    )
    for key in summary_keys:
        actual_value = actual_summary[key]
        expected_value = expected_summary[key]
        if math.isnan(actual_value) and math.isnan(expected_value):
            continue
        if not math.isclose(actual_value, expected_value, rel_tol=rtol, abs_tol=rtol):
            return {
                "strategy": strategy,
                "dataset": dataset_symbol,
                "reason": f"summary.{key} differs",
                "actual": actual_value,
                "expected": expected_value,
            }
    return None


def compare(
    *,
    engine: str,
    datasets_root: Path,
    expected_root: Path,
    rtol: float,
) -> dict[str, Any]:
    readiness = build_report(
        datasets_root=datasets_root,
        expected_root=expected_root,
        engine=engine,
    )
    if not readiness["ok"]:
        return {
            "ok": False,
            "summary": {"compared": 0, "failed": 0},
            "readiness": readiness,
            "failures": [],
        }

    manifest = filter_manifest_by_engine(load_manifest(), engine)
    datasets = {dataset["symbol"]: dataset for dataset in manifest["datasets"]}
    failures = []
    for strategy, dataset_symbol in planned_jobs(manifest):
        failure = _compare_job(
            strategy=strategy,
            dataset=datasets[dataset_symbol],
            datasets_root=datasets_root,
            expected_root=expected_root,
            rtol=rtol,
        )
        if failure is not None:
            failures.append(failure)
    return {
        "ok": not failures,
        "summary": {"compared": len(planned_jobs(manifest)), "failed": len(failures)},
        "readiness": readiness["summary"],
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare golden expected results.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--engine",
        choices=["all", "tv", "tdx"],
        default="tv",
        help="provider engine subset to compare",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DATASETS_DIR,
        help="root containing golden dataset engine folders",
    )
    parser.add_argument(
        "--expected-root",
        type=Path,
        default=EXPECTED_DIR,
        help="root containing expected v1.0 result files",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare(
        engine=args.engine,
        datasets_root=args.datasets_root,
        expected_root=args.expected_root,
        rtol=args.rtol,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        summary = payload["summary"]
        print(
            "golden-compare:"
            f"ok={payload['ok']}"
            f" compared={summary['compared']}"
            f" failed={summary['failed']}"
        )
        for failure in payload["failures"][:5]:
            print(f"failure:{failure['strategy']}:{failure['dataset']}:{failure['reason']}")
    if isinstance(payload.get("readiness"), dict) and not payload["summary"]["compared"]:
        return 2
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # PyArrow can occasionally block during interpreter shutdown while tearing
    # down its global thread pool after parquet reads in subprocess-heavy tests.
    # This script has no cleanup side effects after emitting its payload.
    import os

    os._exit(code)
