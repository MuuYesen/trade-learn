from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_golden import (  # noqa: E402
    dataset_path,
    filter_manifest_by_engine,
    load_manifest,
    planned_jobs,
)

DATASETS_DIR = ROOT / "tests" / "golden" / "datasets"
EXPECTED_DIR = ROOT / "tests" / "golden" / "expected" / "v1.0"
STRATEGY_DIR = ROOT / "tests" / "golden" / "strategies"


def expected_path(strategy: str, dataset: str, expected_root: Path = EXPECTED_DIR) -> Path:
    return expected_root / f"{strategy}__{dataset}.json"


def resolved_dataset_path(dataset: dict[str, str], datasets_root: Path = DATASETS_DIR) -> Path:
    return datasets_root / dataset["engine"] / dataset_path(dataset).name


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def strategy_adapter_ready(
    strategy_name: str,
    strategy_dir: Path = STRATEGY_DIR,
) -> tuple[bool, str]:
    path = strategy_dir / f"{strategy_name}.py"
    if not path.exists():
        return False, "missing strategy script"
    namespace = runpy.run_path(str(path))
    from tradelearn.backtest import Strategy

    strategy_classes = [
        value
        for key, value in namespace.items()
        if key.endswith("Strategy")
        and isinstance(value, type)
        and value is not Strategy
    ]
    if len(strategy_classes) != 1:
        return False, "missing single strategy adapter class"
    if issubclass(strategy_classes[0], Strategy):
        return True, "ready"
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


def _subset_summary(
    datasets: list[dict[str, str]],
    dataset_blockers: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    subsets: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        engine = dataset["engine"]
        summary = subsets.setdefault(
            engine,
            {"datasets_total": 0, "datasets_ready": 0, "ok": False, "blockers": []},
        )
        summary["datasets_total"] += 1

    for blocker in dataset_blockers:
        subsets[blocker["engine"]]["blockers"].append(blocker)

    for summary in subsets.values():
        summary["datasets_ready"] = summary["datasets_total"] - len(summary["blockers"])
        summary["ok"] = summary["datasets_total"] > 0 and not summary["blockers"]
    return subsets


def build_report(
    *,
    datasets_root: Path = DATASETS_DIR,
    expected_root: Path = EXPECTED_DIR,
    strategy_dir: Path = STRATEGY_DIR,
    engine: str = "all",
) -> dict[str, Any]:
    manifest = filter_manifest_by_engine(load_manifest(), engine)
    datasets = manifest["datasets"]
    strategies = manifest["strategies"]
    jobs = planned_jobs(manifest)

    dataset_blockers = []
    for dataset in datasets:
        path = resolved_dataset_path(dataset, datasets_root)
        if not path.exists():
            dataset_blockers.append(
                {
                    "symbol": dataset["symbol"],
                    "engine": dataset["engine"],
                    "path": display_path(path),
                    "reason": "missing real parquet",
                }
            )

    expected_blockers = []
    for strategy, dataset_symbol in jobs:
        path = expected_path(strategy, dataset_symbol, expected_root)
        if not path.exists():
            expected_blockers.append(
                {
                    "strategy": strategy,
                    "dataset": dataset_symbol,
                    "path": display_path(path),
                    "reason": "missing expected v1.0 result",
                }
            )

    strategy_blockers = []
    for strategy in strategies:
        name = strategy["name"]
        ready, reason = strategy_adapter_ready(name, strategy_dir)
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
        "subsets": _subset_summary(datasets, dataset_blockers),
        "blockers": {
            "datasets": dataset_blockers,
            "expected": expected_blockers,
            "strategies": strategy_blockers,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage 3 golden readiness gates.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
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
    parser.add_argument(
        "--strategies-root",
        type=Path,
        default=STRATEGY_DIR,
        help="root containing golden strategy adapters",
    )
    parser.add_argument(
        "--require-tv-subset",
        action="store_true",
        help=(
            "return success when all TV parquet datasets are present, "
            "even if full golden is blocked"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["all", "tv", "tdx"],
        default="all",
        help="Limit readiness totals and blockers to one provider engine",
    )
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
    tv = report["subsets"].get("tv")
    if tv is not None:
        print(
            "tv-subset:"
            f"ok={tv['ok']}"
            f" datasets={tv['datasets_ready']}/{tv['datasets_total']}"
        )
    for category, blockers in report["blockers"].items():
        for blocker in blockers[:3]:
            print(f"{category}:blocked: {blocker['reason']}")


def main() -> int:
    args = parse_args()
    report = build_report(
        datasets_root=args.datasets_root,
        expected_root=args.expected_root,
        strategy_dir=args.strategies_root,
        engine=args.engine,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_text(report)
    tv_ok = report["subsets"].get("tv", {}).get("ok", False)
    if args.require_tv_subset and tv_ok:
        return 0
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
