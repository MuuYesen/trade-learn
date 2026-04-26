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


def find_dataset(engine: str, datasets_root: Path) -> tuple[dict[str, str], Path]:
    manifest = load_manifest()
    for dataset in manifest["datasets"]:
        if dataset["engine"] != engine:
            continue
        path = datasets_root / engine / dataset_path(dataset).name
        if path.exists():
            return dataset, path
    raise FileNotFoundError(f"no available {engine} parquet dataset under {datasets_root}")


def run_smoke(engine: str, datasets_root: Path) -> dict[str, object]:
    dataset, path = find_dataset(engine, datasets_root)
    frame = pd.read_parquet(path)
    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(frame, name=dataset["symbol"])
    cerebro.addstrategy(BuyThenClose)
    cerebro.addanalyzer(SmokeAnalyzer, name="smoke")

    [strategy] = cerebro.run()
    analysis = strategy.analyzers["smoke"].get_analysis()
    return {
        "engine": engine,
        "dataset": dataset["symbol"],
        "path": str(path),
        "trades": analysis["trades"],
        "equity": analysis["equity"],
        "final_cash": strategy.broker.getcash(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["tv", "tdx"], default="tv")
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_smoke(args.engine, args.datasets_root)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "small-golden-smoke:"
            f"engine={result['engine']}"
            f" dataset={result['dataset']}"
            f" trades={len(result['trades'])}"
            f" final_cash={result['final_cash']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
