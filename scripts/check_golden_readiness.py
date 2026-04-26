from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

from scripts.build_golden import dataset_path, load_manifest, planned_jobs

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIR = ROOT / "tests" / "golden" / "expected" / "v1.0"
STRATEGY_DIR = ROOT / "tests" / "golden" / "strategies"


def expected_path(strategy: str, dataset: str) -> Path:
    return EXPECTED_DIR / f"{strategy}__{dataset}.json"


def strategy_adapter_ready(strategy_name: str) -> tuple[bool, str]:
    path = STRATEGY_DIR / f"{strategy_name}.py"
    if not path.exists():
        return False, "missing strategy script"
    namespace = runpy.run_path(str(path))
    strategy_classes = [
        value
        for key, value in namespace.items()
        if key.endswith("Strategy") and isinstance(value, type)
    ]
    if len(strategy_classes) != 1:
        return False, "missing single strategy adapter class"
    strategy = strategy_classes[0]()
    run = getattr(strategy, "run", None)
    if not callable(run):
        return False, "missing runnable adapter"
    try:
        run()
    except NotImplementedError:
        return False, "adapter still blocks execution"
    except Exception as exc:
        return False, f"adapter execution failed: {type(exc).__name__}"
    return True, "ready"


def build_report() -> dict[str, Any]:
    manifest = load_manifest()
    datasets = manifest["datasets"]
    strategies = manifest["strategies"]
    jobs = planned_jobs(manifest)

    dataset_blockers = []
    for dataset in datasets:
        path = dataset_path(dataset)
        if not path.exists():
            dataset_blockers.append(
                {
                    "symbol": dataset["symbol"],
                    "engine": dataset["engine"],
                    "path": str(path.relative_to(ROOT)),
                    "reason": "missing real parquet",
                }
            )

    expected_blockers = []
    for strategy, dataset_symbol in jobs:
        path = expected_path(strategy, dataset_symbol)
        if not path.exists():
            expected_blockers.append(
                {
                    "strategy": strategy,
                    "dataset": dataset_symbol,
                    "path": str(path.relative_to(ROOT)),
                    "reason": "missing expected v1.0 result",
                }
            )

    strategy_blockers = []
    for strategy in strategies:
        name = strategy["name"]
        ready, reason = strategy_adapter_ready(name)
        if not ready:
            strategy_blockers.append({"strategy": name, "reason": reason})

    dataset_ready = len(datasets) - len(dataset_blockers)
    expected_ready = len(jobs) - len(expected_blockers)
    strategy_ready = len(strategies) - len(strategy_blockers)
    ok = not dataset_blockers and not expected_blockers and not strategy_blockers
    return {
        "ok": ok,
        "summary": {
            "datasets_total": len(datasets),
            "datasets_ready": dataset_ready,
            "expected_total": len(jobs),
            "expected_ready": expected_ready,
            "strategies_total": len(strategies),
            "strategies_ready": strategy_ready,
        },
        "blockers": {
            "datasets": dataset_blockers,
            "expected": expected_blockers,
            "strategies": strategy_blockers,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage 3 golden readiness gates.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def print_text(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "golden-readiness:"
        f"ok={report['ok']}"
        f" datasets={summary['datasets_ready']}/{summary['datasets_total']}"
        f" expected={summary['expected_ready']}/{summary['expected_total']}"
        f" strategies={summary['strategies_ready']}/{summary['strategies_total']}"
    )
    for category, blockers in report["blockers"].items():
        for blocker in blockers[:3]:
            print(f"{category}:blocked: {blocker['reason']}")


def main() -> int:
    args = parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_text(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
