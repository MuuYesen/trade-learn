#!/usr/bin/env python
"""Run a minimal Backtrader oracle for TV subset golden parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.core.errors import GoldenDataError  # noqa: E402

SUPPORTED_BACKTRADER_STRATEGIES = (
    "sma_cross",
    "rsi_oversold",
    "bollinger_breakout",
    "macd_cross",
    "tdx_kdj",
    "supertrend_tv",
    "pairs_trading",
    "equal_weight",
    "alpha101_ml",
    "momentum_portfolio",
)


def _clean_json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    return value


def _utc_datetime(value: Any) -> Any:
    if hasattr(value, "tzinfo") and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
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


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _metric_value(func: Any, *args: Any, scale: float = 1.0, **kwargs: Any) -> float:
    try:
        return _safe_float(func(*args, **kwargs)) * scale
    except Exception:
        return 0.0


def _closed_trade_rows(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trade for trade in trades if bool(trade.get("isclosed"))]


def _prepare_backtrader_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a single-symbol DatetimeIndex frame consumable by Backtrader."""

    prepared = bars.copy()
    if isinstance(prepared.index, pd.MultiIndex):
        if "timestamp" not in prepared.index.names:
            raise GoldenDataError(
                "Backtrader oracle requires a timestamp level in Bars MultiIndex"
            )
        symbols = (
            prepared.index.get_level_values("symbol").unique()
            if "symbol" in prepared.index.names
            else []
        )
        if len(symbols) > 1:
            raise GoldenDataError(
                "Backtrader oracle expects one symbol per parquet, "
                f"got {len(symbols)} symbols"
            )
        prepared = prepared.reset_index("symbol", drop=True)
    if not isinstance(prepared.index, pd.DatetimeIndex):
        raise GoldenDataError("Backtrader oracle requires a DatetimeIndex")
    if prepared.index.tz is not None:
        prepared.index = prepared.index.tz_convert("UTC").tz_localize(None)
    return prepared.sort_index()


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _analyzer_results(strategy: Any) -> dict[str, Any]:
    return {
        "drawdown": dict(strategy.analyzers.drawdown.get_analysis()),
        "returns": dict(strategy.analyzers.returns.get_analysis()),
        "sharpe": dict(strategy.analyzers.sharpe.get_analysis()),
        "sqn": dict(strategy.analyzers.sqn.get_analysis()),
        "trades": dict(strategy.analyzers.trades.get_analysis()),
    }


def _trade_analyzer_metrics(
    trade_analyzer: dict[str, Any],
    initial_cash: float,
) -> dict[str, float]:
    total_closed = _safe_float(_nested_value(trade_analyzer, "total", "closed"))
    won_total = _safe_float(_nested_value(trade_analyzer, "won", "total"))
    won_pnl_total = _safe_float(_nested_value(trade_analyzer, "won", "pnl", "total"))
    lost_pnl_total = _safe_float(_nested_value(trade_analyzer, "lost", "pnl", "total"))
    net_pnl_total = _safe_float(_nested_value(trade_analyzer, "pnl", "net", "total"))
    net_pnl_average = _safe_float(_nested_value(trade_analyzer, "pnl", "net", "average"))
    gross_loss = abs(lost_pnl_total)
    return {
        "win_rate_pct": won_total / total_closed * 100.0 if total_closed else 0.0,
        "profit_factor": won_pnl_total / gross_loss if gross_loss > 1e-12 else 0.0,
        "expectancy": net_pnl_average,
        "avg_trade_pct": net_pnl_average / initial_cash * 100.0 if initial_cash else 0.0,
        "final_realized_pnl": net_pnl_total,
    }


def summarize_backtrader_metrics(
    *,
    bars: int,
    initial_cash: float,
    final_cash: float,
    final_value: float,
    orders: int,
    fills: int,
    trades: list[dict[str, Any]],
    equity: pd.Series,
    analyzers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a summary from Backtrader-native results and analyzers."""

    clean_equity = equity.astype("float64") if not equity.empty else pd.Series(dtype="float64")
    analyzer_payload = analyzers or {}
    drawdown = analyzer_payload.get("drawdown", {})
    returns = analyzer_payload.get("returns", {})
    sharpe = analyzer_payload.get("sharpe", {})
    sqn = analyzer_payload.get("sqn", {})
    trade_analyzer = analyzer_payload.get("trades", {})
    closed = _closed_trade_rows(trades)

    analyzer_closed_trades = _safe_float(_nested_value(trade_analyzer, "total", "closed"))
    summary = {
        "bars": float(bars),
        "initial_cash": float(initial_cash),
        "final_cash": float(final_cash),
        "final_value": float(final_value),
        "peak_value": float(clean_equity.max()) if not clean_equity.empty else float(final_value),
        "total_return": (float(final_value) / float(initial_cash) - 1.0)
        if initial_cash
        else 0.0,
        "return_pct": (float(final_value) / float(initial_cash) - 1.0) * 100.0
        if initial_cash
        else 0.0,
        "trades": len(trades),
        "orders": float(orders),
        "fills": float(fills),
        "returns": len(clean_equity.pct_change().fillna(0.0)),
        "total_trades": float(analyzer_closed_trades or len(closed)),
        "total_orders": float(fills),
        "total_fills": float(fills),
    }
    summary.update(_trade_analyzer_metrics(trade_analyzer, initial_cash))
    optional_metrics = {
        "annual_return": _nested_value(returns, "rnorm100"),
        "log_return": _nested_value(returns, "rtot"),
        "max_drawdown": _safe_float(_nested_value(drawdown, "max", "drawdown")) / 100.0,
        "drawdown": _safe_float(_nested_value(drawdown, "drawdown")) / 100.0,
        "drawdown_len": _nested_value(drawdown, "len"),
        "max_drawdown_len": _nested_value(drawdown, "max", "len"),
        "sharpe": _nested_value(sharpe, "sharperatio"),
        "sqn": _nested_value(sqn, "sqn"),
    }
    for key, value in optional_metrics.items():
        if value is not None:
            summary[key] = _safe_float(value)
    if "sharperatio" in sharpe and "sharpe" not in summary:
        summary["sharpe"] = None
    return summary


def _values(line: Any, size: int) -> list[float]:
    return [float(value) for value in line.get(size=size)]


def _sma(line: Any, size: int) -> float:
    values = _values(line, size)
    return sum(values) / len(values) if values else float(line[0])


def _momentum(line: Any, size: int = 2) -> float:
    values = _values(line, size + 1)
    if len(values) <= size:
        return 0.0
    return values[-1] - values[0]


def _range_midpoint(data: Any, size: int = 3) -> float:
    highs = _values(data.high, size)
    lows = _values(data.low, size)
    return (max(highs) + min(lows)) / 2.0


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

    class RsiOversold(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return _momentum(self.data.close) < 0

        def should_exit(self) -> bool:
            return _momentum(self.data.close) > 0

    class BollingerBreakout(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return float(self.data.close[0]) >= max(_values(self.data.high, 3))

        def should_exit(self) -> bool:
            return float(self.data.close[0]) < _range_midpoint(self.data, 3)

    class MacdCross(GoldenBacktraderBase):
        def _macd_proxy(self) -> float:
            return _sma(self.data.close, 2) - _sma(self.data.close, 3)

        def should_enter(self) -> bool:
            return self._macd_proxy() > 0

        def should_exit(self) -> bool:
            return self._macd_proxy() < 0

    class SupertrendTv(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return float(self.data.close[0]) > _range_midpoint(self.data, 3)

        def should_exit(self) -> bool:
            return float(self.data.close[0]) < _range_midpoint(self.data, 3)

    class PairsTrading(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return float(self.data.close[0]) < _sma(self.data.close, 3)

        def should_exit(self) -> bool:
            return float(self.data.close[0]) > _sma(self.data.close, 3)

    class EqualWeight(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return True

        def should_exit(self) -> bool:
            return False

    class Alpha101Ml(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return (
                _momentum(self.data.close) > 0
                and float(self.data.close[0]) > _range_midpoint(self.data, 3)
            )

        def should_exit(self) -> bool:
            return _momentum(self.data.close) < 0

    class MomentumPortfolio(GoldenBacktraderBase):
        def should_enter(self) -> bool:
            return _momentum(self.data.close, 2) > 0

        def should_exit(self) -> bool:
            return _momentum(self.data.close, 2) < 0

    class TdxKdj(GoldenBacktraderBase):
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
        "rsi_oversold": RsiOversold,
        "bollinger_breakout": BollingerBreakout,
        "macd_cross": MacdCross,
        "tdx_kdj": TdxKdj,
        "supertrend_tv": SupertrendTv,
        "pairs_trading": PairsTrading,
        "equal_weight": EqualWeight,
        "alpha101_ml": Alpha101Ml,
        "momentum_portfolio": MomentumPortfolio,
    }
    try:
        return strategies[strategy_name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_BACKTRADER_STRATEGIES)
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
        dt = _utc_datetime(strategy.data.datetime.datetime(0))
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
                "datetime": _utc_datetime(strategy.data.datetime.datetime(0)),
                "value": float(strategy.broker.getvalue()),
                "position_size": float(strategy.position.size),
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
    bars = _prepare_backtrader_bars(pd.read_parquet(parquet))
    recorder = GoldenRecorder()

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    feed = bt.feeds.PandasData(dataname=bars)
    cerebro.adddata(feed, name=dataset or parquet.stem)
    cerebro.addstrategy(_attach_hooks(_strategy_class(strategy_name), recorder))
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.0,
        stddev_sample=True,
    )
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    [strategy] = cerebro.run()

    final_cash = float(strategy.broker.getcash())
    final_value = float(strategy.broker.getvalue())
    equity = pd.Series(
        [row["value"] for row in recorder.equity],
        index=[row["datetime"] for row in recorder.equity],
        dtype="float64",
    )
    summary = summarize_backtrader_metrics(
        bars=int(len(bars)),
        initial_cash=float(cash),
        final_cash=final_cash,
        final_value=final_value,
        orders=len(recorder.orders),
        fills=len(recorder.fills),
        trades=recorder.trades,
        equity=equity,
        analyzers=_analyzer_results(strategy),
    )
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
        choices=SUPPORTED_BACKTRADER_STRATEGIES,
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
