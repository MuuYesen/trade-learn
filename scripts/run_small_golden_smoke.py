#!/usr/bin/env python
"""Run a small golden smoke over available parquet datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_golden import dataset_path, load_manifest  # noqa: E402
from tradelearn.backtest import Analyzer, Cerebro, Strategy  # noqa: E402

DATASETS_DIR = ROOT / "tests" / "golden" / "datasets"


class BuyThenClose(Strategy):
    """Minimal strategy that proves order/trade/equity plumbing."""

    def __init__(self) -> None:
        self._bar_count = 0

    def next(self) -> None:
        self._bar_count += 1
        if self._bar_count == 1:
            self.buy(size=2)
        elif self._bar_count == 3:
            self.close()


class SmokeAnalyzer(Analyzer):
    """Collect trades and mark-to-market equity for smoke output."""

    def __init__(self) -> None:
        self.trades: list[dict[str, float | bool]] = []
        self.equity: list[float] = []

    def on_trade(self, trade: Any) -> None:
        self.trades.append(
            {
                "size": trade.size,
                "price": trade.price,
                "pnl": trade.pnl,
                "isclosed": trade.isclosed,
            }
        )

    def on_bar(self, bar: Any) -> None:
        position = self.strategy.position
        cash = self.strategy.broker.getcash()
        self.equity.append(cash + position.size * bar.close)

    def get_analysis(self) -> dict[str, object]:
        return {"trades": self.trades, "equity": self.equity}


def manifest_datasets(engine: str) -> list[dict[str, str]]:
    """Return manifest datasets for an engine in documented order."""

    manifest = load_manifest()
    return [dataset for dataset in manifest["datasets"] if dataset["engine"] == engine]


def run_dataset_smoke(dataset: dict[str, str], path: Path) -> dict[str, object]:
    """Run the synthetic strategy over one parquet dataset."""

    frame = pd.read_parquet(path)
    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(frame, name=dataset["symbol"])
    cerebro.addstrategy(BuyThenClose)
    cerebro.addanalyzer(SmokeAnalyzer, name="smoke")

    [strategy] = cerebro.run()
    analysis = strategy.analyzers["smoke"].get_analysis()
    final_cash = strategy.broker.getcash()
    return {
        "dataset": dataset["symbol"],
        "path": str(path),
        "trades": analysis["trades"],
        "equity": analysis["equity"],
        "final_cash": final_cash,
        "pnl": final_cash - 100.0,
    }


def run_smoke(
    engine: str,
    datasets_root: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Run small golden smoke over every manifest dataset for an engine."""

    datasets = manifest_datasets(engine)
    results: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for dataset in datasets:
        path = dataset_path(dataset, datasets_root)
        item = {"dataset": dataset["symbol"], "path": str(path)}
        if not path.exists():
            missing.append({**item, "reason": "missing parquet"})
            continue
        try:
            results.append(run_dataset_smoke(dataset, path))
        except Exception as exc:
            failed.append({**item, "reason": f"{type(exc).__name__}: {exc}"})

    ok = bool(results) and not failed and (allow_missing or not missing)
    return {
        "ok": ok,
        "engine": engine,
        "summary": {
            "requested": len(datasets),
            "ran": len(results),
            "missing": len(missing),
            "failed": len(failed),
        },
        "results": results,
        "missing": missing,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["tv", "tdx"], default="tv")
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_DIR)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="run available datasets without requiring every manifest dataset",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_smoke(
        args.engine,
        args.datasets_root,
        allow_missing=args.allow_missing,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "small-golden-smoke:"
            f"engine={result['engine']}"
            f" ok={result['ok']}"
            f" requested={result['summary']['requested']}"
            f" ran={result['summary']['ran']}"
            f" missing={result['summary']['missing']}"
            f" failed={result['summary']['failed']}"
        )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
