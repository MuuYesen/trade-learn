#!/usr/bin/env python
"""Run a minimal Backtrader oracle for TV subset golden parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.core import GoldenDataError  # noqa: E402


def _clean_json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    return value


def _load_backtrader() -> Any:
    try:
        import backtrader as bt
    except ModuleNotFoundError as exc:
        raise GoldenDataError(
            "backtrader is required for --oracle backtrader; "
            "run `uv sync --group oracle --extra dev`"
        ) from exc
    return bt


def _values(line: Any, size: int) -> list[float]:
    return [float(value) for value in line.get(size=size)]


def _sma(line: Any, size: int) -> float:
    values = _values(line, size)
    return sum(values) / len(values) if values else float(line[0])


def _strategy_class(strategy_name: str) -> type[Any]:
    bt = _load_backtrader()

    class GoldenBacktraderBase(bt.Strategy):
        size = 1.0

        def next(self) -> None:
            if len(self.data) <= 3:
                return
            if not self.position and self.should_enter():
                self.buy(size=self.size)
            elif self.position and self.should_exit():
                self.close()

        def should_enter(self) -> bool:
            return False

        def should_exit(self) -> bool:
            return False

    class SmaCross(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return _sma(self.data.close, 2) > _sma(self.data.close, 3)

        def should_exit(self) -> bool:
            return _sma(self.data.close, 2) < _sma(self.data.close, 3)

    class MacdCross(GoldenBacktraderBase):
        def _macd_proxy(self) -> float:
            return _sma(self.data.close, 2) - _sma(self.data.close, 3)

        def should_enter(self) -> bool:
            return self._macd_proxy() > 0

        def should_exit(self) -> bool:
            return self._macd_proxy() < 0

    class Tdx30Kdj(GoldenBacktraderBase):
        def _k_value(self) -> float:
            highs = _values(self.data.high, 3)
            lows = _values(self.data.low, 3)
            span = max(highs) - min(lows)
            if span == 0.0:
                return 50.0
            return (float(self.data.close[0]) - min(lows)) / span * 100.0

        def should_enter(self) -> bool:
            return self._k_value() > 50.0

        def should_exit(self) -> bool:
            return self._k_value() < 50.0

    strategies = {
        "sma_cross": SmaCross,
        "macd_cross": MacdCross,
        "tdx30_kdj": Tdx30Kdj,
    }
    try:
        return strategies[strategy_name]
    except KeyError as exc:
        supported = ", ".join(sorted(strategies))
        raise GoldenDataError(
            f"unsupported Backtrader oracle strategy: {strategy_name}; supported: {supported}"
        ) from exc


class GoldenRecorder:
    """Collect Backtrader orders, trades, fills, and equity into JSON rows."""

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []
        self.fills: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.equity: list[dict[str, Any]] = []
        self._position_size = 0.0
        self._entry_price = 0.0

    def bind(self, strategy: Any) -> None:
        strategy._golden_recorder = self

    def record_order(self, strategy: Any, order: Any) -> None:
        dt = strategy.data.datetime.datetime(0)
        status = order.getstatusname()
        executed_size = float(order.executed.size)
        executed_price = float(order.executed.price)
        commission = float(order.executed.comm)
        row = {
            "datetime": dt,
            "side": "buy" if order.isbuy() else "sell",
            "status": status,
            "size": float(order.created.size),
            "price": None if order.created.price is None else float(order.created.price),
            "executed_size": executed_size,
            "executed_price": executed_price,
            "executed_value": float(order.executed.value),
            "commission": commission,
        }
        self.orders.append(row)
        if status == "Completed" and executed_size != 0.0:
            self.fills.append(
                {
                    "datetime": dt,
                    "size": executed_size,
                    "price": executed_price,
                    "value": abs(executed_size) * executed_price,
                    "commission": commission,
                }
            )
            self._record_trade_from_fill(dt, executed_size, executed_price, commission)

    def _record_trade_from_fill(
        self,
        dt: Any,
        executed_size: float,
        executed_price: float,
        commission: float,
    ) -> None:
        next_size = self._position_size + executed_size
        if self._position_size == 0.0 and next_size != 0.0:
            self._entry_price = executed_price
            self.trades.append(
                {
                    "datetime": dt,
                    "size": next_size,
                    "price": executed_price,
                    "value": abs(next_size) * executed_price,
                    "commission": commission,
                    "pnl": 0.0,
                    "pnlcomm": -commission,
                    "isopen": True,
                    "isclosed": False,
                }
            )
        elif self._position_size != 0.0 and next_size == 0.0:
            pnl = (executed_price - self._entry_price) * self._position_size
            self.trades.append(
                {
                    "datetime": dt,
                    "size": 0.0,
                    "price": executed_price,
                    "value": 0.0,
                    "commission": commission,
                    "pnl": pnl,
                    "pnlcomm": pnl - commission,
                    "isopen": False,
                    "isclosed": True,
                }
            )
            self._entry_price = 0.0
        self._position_size = next_size

    def record_trade(self, strategy: Any, trade: Any) -> None:
        return

    def record_equity(self, strategy: Any) -> None:
        self.equity.append(
            {
                "datetime": strategy.data.datetime.datetime(0),
                "value": float(strategy.broker.getvalue()),
            }
        )


def _attach_hooks(strategy_cls: type[Any], recorder: GoldenRecorder) -> type[Any]:
    class RecordingStrategy(strategy_cls):
        def start(self) -> None:
            recorder.bind(self)
            if hasattr(super(), "start"):
                super().start()

        def notify_order(self, order: Any) -> None:
            recorder.record_order(self, order)

        def notify_trade(self, trade: Any) -> None:
            recorder.record_trade(self, trade)

        def next(self) -> None:
            super().next()
            recorder.record_equity(self)

    return RecordingStrategy


def run_backtrader_oracle(
    strategy_name: str,
    parquet: Path,
    *,
    dataset: str | None = None,
    cash: float = 100_000.0,
) -> dict[str, Any]:
    """Run one Backtrader oracle strategy and return expected payload."""

    bt = _load_backtrader()
    if not parquet.exists():
        raise GoldenDataError(f"missing dataset parquet: {parquet}")
    bars = pd.read_parquet(parquet)
    if bars.index.tz is not None:
        bars = bars.copy()
        bars.index = bars.index.tz_convert("UTC").tz_localize(None)
    recorder = GoldenRecorder()

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    feed = bt.feeds.PandasData(dataname=bars)
    cerebro.adddata(feed, name=dataset or parquet.stem)
    cerebro.addstrategy(_attach_hooks(_strategy_class(strategy_name), recorder))
    [strategy] = cerebro.run()

    final_cash = float(strategy.broker.getcash())
    final_value = float(strategy.broker.getvalue())
    returns = pd.Series(
        [row["value"] for row in recorder.equity],
        index=[row["datetime"] for row in recorder.equity],
        dtype="float64",
    ).pct_change().fillna(0.0)
    summary = {
        "bars": int(len(bars)),
        "initial_cash": float(cash),
        "final_cash": final_cash,
        "final_value": final_value,
        "total_return": (final_value / cash) - 1.0 if cash else 0.0,
        "trades": len(recorder.trades),
        "orders": len(recorder.orders),
        "fills": len(recorder.fills),
        "returns": len(returns),
    }
    return {
        "version": "v1.0",
        "strategy": strategy_name,
        "dataset": dataset or parquet.stem,
        "engine": "tv",
        "source": str(parquet),
        "source_engine": "backtrader",
        "summary": _clean_json_value(summary),
        "trades": _clean_json_value(recorder.trades),
        "orders": _clean_json_value(recorder.orders),
        "fills": _clean_json_value(recorder.fills),
        "positions": [],
        "equity": _clean_json_value(recorder.equity),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["sma_cross", "macd_cross", "tdx30_kdj"],
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dataset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_backtrader_oracle(args.strategy, args.parquet, dataset=args.dataset)
    except GoldenDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if args.out is None:
        print(output)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"expected={args.strategy}:{payload['dataset']} status=ok path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
