from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradelearn.backtest.targets import TargetWeightSnapshot, build_target_weight_intents


LOOKBACK = 20
INVESTMENT_WEIGHT = 0.95
DEFAULT_TOLERANCE = 1e-2


@dataclass(frozen=True)
class ParityResult:
    name: str
    final_value: float
    elapsed_seconds: float
    bars_per_sec: float
    order_count: int
    timings_seconds: tuple[float, ...]
    target_count: int = 0
    target_history: tuple[tuple[int, tuple[str, ...]], ...] = ()
    submitted_count: int = 0
    intent_count: int = 0
    status_counts: tuple[tuple[str, int], ...] = ()
    zero_size_orders: int = 0
    unnotified_orders: int = 0
    unnotified_sample: tuple[str, ...] = ()


def make_panel(symbols: int, bars: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2000-01-01", periods=bars, freq="B", tz="UTC")
    panel: dict[str, pd.DataFrame] = {}
    for i in range(symbols):
        returns = rng.normal(loc=0.0002 + i * 0.0000001, scale=0.01, size=bars)
        close = 100.0 * np.exp(np.cumsum(returns))
        # Keep open equal to close so this benchmark isolates target-weight
        # ordering semantics instead of mixing in execution-price timing.
        open_ = close.copy()
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        volume = rng.integers(10_000, 100_000, size=bars).astype(float)
        panel[f"S{i:04d}"] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
    return panel


def _target_weights_from_scores(scores: list[tuple[str, float]], holdings: int) -> dict[str, float]:
    selected = [
        symbol
        for symbol, _score in sorted(scores, key=lambda item: item[1], reverse=True)[:holdings]
    ]
    weights = {symbol: 0.0 for symbol, _score in scores}
    if not selected:
        return weights
    weight = INVESTMENT_WEIGHT / len(selected)
    for symbol in selected:
        weights[symbol] = weight
    return weights


def _engine_strategy_cls(bt_module, rebalance_every: int, holdings: int, total_bars: int):
    class TargetWeightParityStrategy(bt_module.Strategy):
        params = (
            ("rebalance_every", rebalance_every),
            ("holdings", holdings),
            ("lookback", LOOKBACK),
        )

        def __init__(self) -> None:
            self.completed_orders = 0
            self.submitted_targets = 0
            self.intent_count = 0
            self.status_counts = {}
            self.zero_size_orders = 0
            self.returned_order_info = {}
            self.notified_order_refs = set()
            self.target_history = []
            self.addminperiod(self.p.lookback + 1)

        def notify_order(self, order) -> None:
            self.notified_order_refs.add(getattr(order, "ref", None))
            status_name = str(getattr(order, "getstatusname", lambda: order.status)())
            self.status_counts[status_name] = self.status_counts.get(status_name, 0) + 1
            if order.status == order.Completed:
                self.completed_orders += 1

        def next(self) -> None:
            if len(self) <= int(self.p.lookback):
                return
            # Backtrader returns final-bar orders but has no subsequent
            # lifecycle step to notify or complete them. The parity benchmark
            # compares executable target rebalance events, so skip terminal
            # orders that cannot have a full lifecycle in both engines.
            if len(self) >= total_bars:
                return
            if len(self) % int(self.p.rebalance_every):
                return

            scores = []
            for data in self.datas:
                past = float(data.close[-int(self.p.lookback)])
                current = float(data.close[0])
                scores.append((data._name, current / past - 1.0 if past else 0.0))
            weights = _target_weights_from_scores(scores, int(self.p.holdings))
            self.target_history.append((len(self), dict(weights)))
            data_by_symbol = {str(data._name): data for data in self.datas}
            snapshots = {
                name: TargetWeightSnapshot(
                    price=float(data.close[0]),
                    size=float(self.getposition(data=data).size),
                )
                for name, data in data_by_symbol.items()
            }
            intents = build_target_weight_intents(
                weights,
                data_by_symbol=data_by_symbol,
                snapshots=snapshots,
                equity=float(self.broker.getvalue()),
                close_missing=True,
            )
            self.intent_count += len(intents)

            for intent in intents:
                order = self.order_target_percent(data=intent.data, target=intent.target_weight)
                if order is not None:
                    created = getattr(order, "created", None)
                    size = getattr(created, "size", getattr(order, "size", None))
                    if size is not None and abs(float(size)) < 1e-12:
                        self.zero_size_orders += 1
                    self.returned_order_info[getattr(order, "ref", len(self.returned_order_info))] = (
                        f"bar={len(self)} symbol={intent.symbol} target={intent.target_weight:.6f} "
                        f"size={float(size) if size is not None else 'NA'}"
                    )
                    self.submitted_targets += 1

    return TargetWeightParityStrategy


def _lite_strategy_cls(rebalance_every: int, holdings: int, total_bars: int):
    from tradelearn.lite import Strategy

    class TargetWeightParityLiteStrategy(Strategy):
        def init(self) -> None:
            self.completed_orders = 0
            self.intent_count = 0
            self.target_history = []
            self.start_on_bar(LOOKBACK)

        def next(self) -> None:
            if len(self.data) <= LOOKBACK:
                return
            if len(self.data) >= total_bars:
                return
            if len(self.data) % rebalance_every:
                return

            scores = []
            for symbol, feed in self._target_weight_data_map().items():
                close = feed.get_array("close")
                cursor = int(feed._cursor)
                past = float(close[cursor - LOOKBACK])
                current = float(close[cursor])
                scores.append((symbol, current / past - 1.0 if past else 0.0))
            weights = _target_weights_from_scores(scores, holdings)
            self.target_history.append((len(self.data), dict(weights)))
            orders = self.target_weights(weights, close_missing=True)
            self.intent_count += len(orders)

    return TargetWeightParityLiteStrategy


def _run_engine(
    panel: dict[str, pd.DataFrame],
    *,
    holdings: int,
    rebalance_every: int,
    cash: float,
    match_mode: str,
    trade_on_close: bool,
    repeats: int,
    warmup: int,
) -> ParityResult:
    import tradelearn.engine as bt

    timings = []
    final_value = 0.0
    order_count = 0
    target_count = 0
    target_history = ()
    submitted_count = 0
    intent_count = 0
    status_counts = ()
    zero_size_orders = 0
    unnotified_orders = 0
    unnotified_sample = ()
    total_bars = min(len(frame) for frame in panel.values())
    strategy_cls = _engine_strategy_cls(bt, rebalance_every, holdings, total_bars)
    for run_idx in range(warmup + repeats):
        cerebro = bt.Cerebro(match_mode=match_mode, trade_on_close=trade_on_close)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.0)
        for name, frame in panel.items():
            cerebro.adddata(bt.feeds.PandasData(dataname=frame, name=name))
        cerebro.addstrategy(strategy_cls)

        start = time.perf_counter()
        strategies = cerebro.run()
        elapsed = time.perf_counter() - start
        if run_idx >= warmup:
            timings.append(elapsed)
            final_value = float(cerebro.broker.getvalue())
            strategy = strategies[0]
            order_count = int(strategy.completed_orders)
            submitted_count = int(strategy.submitted_targets)
            intent_count = int(strategy.intent_count)
            status_counts = tuple(sorted(strategy.status_counts.items()))
            zero_size_orders = int(strategy.zero_size_orders)
            missing_refs = sorted(set(strategy.returned_order_info) - strategy.notified_order_refs)
            unnotified_orders = len(missing_refs)
            unnotified_sample = tuple(strategy.returned_order_info[ref] for ref in missing_refs[:5])
            target_count = len(strategy.target_history)
            target_history = tuple(
                (bar, tuple(symbol for symbol, weight in weights.items() if weight > 0))
                for bar, weights in strategy.target_history
            )

    elapsed = statistics.median(timings)
    bars_per_sec = _total_bars(panel) / elapsed if elapsed > 0 else 0.0
    return ParityResult(
        "Tradelearn Engine",
        final_value,
        elapsed,
        bars_per_sec,
        order_count,
        tuple(timings),
        target_count,
        target_history,
        submitted_count,
        intent_count,
        status_counts,
        zero_size_orders,
        unnotified_orders,
        unnotified_sample,
    )


def _run_backtrader(
    panel: dict[str, pd.DataFrame],
    *,
    holdings: int,
    rebalance_every: int,
    cash: float,
    repeats: int,
    warmup: int,
    trade_on_close: bool,
) -> ParityResult:
    import backtrader as bt

    class PandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", None),
        )

    timings = []
    final_value = 0.0
    order_count = 0
    target_count = 0
    target_history = ()
    submitted_count = 0
    intent_count = 0
    status_counts = ()
    zero_size_orders = 0
    unnotified_orders = 0
    unnotified_sample = ()
    total_bars = min(len(frame) for frame in panel.values())
    strategy_cls = _engine_strategy_cls(bt, rebalance_every, holdings, total_bars)
    for run_idx in range(warmup + repeats):
        cerebro = bt.Cerebro()
        cerebro.broker.set_coc(trade_on_close)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.0)
        for name, frame in panel.items():
            cerebro.adddata(PandasData(dataname=frame, name=name))
        cerebro.addstrategy(strategy_cls)

        start = time.perf_counter()
        strategies = cerebro.run()
        elapsed = time.perf_counter() - start
        if run_idx >= warmup:
            timings.append(elapsed)
            final_value = float(cerebro.broker.getvalue())
            strategy = strategies[0]
            order_count = int(strategy.completed_orders)
            submitted_count = int(strategy.submitted_targets)
            intent_count = int(strategy.intent_count)
            status_counts = tuple(sorted(strategy.status_counts.items()))
            zero_size_orders = int(strategy.zero_size_orders)
            missing_refs = sorted(set(strategy.returned_order_info) - strategy.notified_order_refs)
            unnotified_orders = len(missing_refs)
            unnotified_sample = tuple(strategy.returned_order_info[ref] for ref in missing_refs[:5])
            target_count = len(strategy.target_history)
            target_history = tuple(
                (bar, tuple(symbol for symbol, weight in weights.items() if weight > 0))
                for bar, weights in strategy.target_history
            )

    elapsed = statistics.median(timings)
    bars_per_sec = _total_bars(panel) / elapsed if elapsed > 0 else 0.0
    return ParityResult(
        "Backtrader",
        final_value,
        elapsed,
        bars_per_sec,
        order_count,
        tuple(timings),
        target_count,
        target_history,
        submitted_count,
        intent_count,
        status_counts,
        zero_size_orders,
        unnotified_orders,
        unnotified_sample,
    )


def _run_lite(
    panel: dict[str, pd.DataFrame],
    *,
    holdings: int,
    rebalance_every: int,
    cash: float,
    match_mode: str,
    trade_on_close: bool,
    repeats: int,
    warmup: int,
) -> ParityResult:
    from tradelearn.lite import Backtest

    timings = []
    final_value = 0.0
    order_count = 0
    target_count = 0
    target_history = ()
    submitted_count = 0
    intent_count = 0
    status_counts = ()
    zero_size_orders = 0
    unnotified_orders = 0
    unnotified_sample = ()
    total_bars = min(len(frame) for frame in panel.values())
    strategy_cls = _lite_strategy_cls(rebalance_every, holdings, total_bars)
    for run_idx in range(warmup + repeats):
        backtest = Backtest(
            panel,
            strategy_cls,
            cash=cash,
            match_mode=match_mode,
            trade_on_close=trade_on_close,
        )
        start = time.perf_counter()
        stats = backtest.run()
        elapsed = time.perf_counter() - start
        if run_idx >= warmup:
            timings.append(elapsed)
            final_value = float(stats["final_value"])
            order_count = len(stats.fills)
            submitted_count = len(stats.orders)
            intent_count = int(getattr(stats.strategy, "intent_count", 0))
            status_counts = (
                tuple(sorted(stats.orders["status"].value_counts().to_dict().items()))
                if "status" in stats.orders
                else ()
            )
            zero_size_orders = int((stats.orders["size"].abs() < 1e-12).sum()) if "size" in stats.orders else 0
            target_count = len(stats.strategy.target_history)
            target_history = tuple(
                (bar, tuple(symbol for symbol, weight in weights.items() if weight > 0))
                for bar, weights in stats.strategy.target_history
            )

    elapsed = statistics.median(timings)
    bars_per_sec = _total_bars(panel) / elapsed if elapsed > 0 else 0.0
    return ParityResult(
        "Tradelearn Lite",
        final_value,
        elapsed,
        bars_per_sec,
        order_count,
        tuple(timings),
        target_count,
        target_history,
        submitted_count,
        intent_count,
        status_counts,
        zero_size_orders,
        unnotified_orders,
        unnotified_sample,
    )


def _total_bars(panel: dict[str, pd.DataFrame]) -> int:
    return sum(len(frame) for frame in panel.values())


def run_parity_benchmark(
    *,
    symbols: int,
    bars: int,
    holdings: int,
    rebalance_every: int,
    cash: float,
    seed: int,
    match_mode: str = "exact",
    repeats: int = 1,
    warmup: int = 0,
    include_lite: bool = True,
    include_backtrader: bool = True,
    trade_on_close: bool = True,
) -> list[ParityResult]:
    if symbols <= 0 or bars <= LOOKBACK + 1 or holdings <= 0:
        raise ValueError("symbols must be > 0, bars must exceed lookback, holdings must be > 0")
    panel = make_panel(symbols=symbols, bars=bars, seed=seed)
    results = [
        _run_engine(
            panel,
            holdings=holdings,
            rebalance_every=rebalance_every,
            cash=cash,
            match_mode=match_mode,
            trade_on_close=trade_on_close,
            repeats=max(1, repeats),
            warmup=max(0, warmup),
        )
    ]
    if include_lite:
        results.append(
            _run_lite(
                panel,
                holdings=holdings,
                rebalance_every=rebalance_every,
                cash=cash,
                match_mode=match_mode,
                trade_on_close=trade_on_close,
                repeats=max(1, repeats),
                warmup=max(0, warmup),
            )
        )
    if include_backtrader:
        results.append(
            _run_backtrader(
                panel,
                holdings=holdings,
                rebalance_every=rebalance_every,
                cash=cash,
                trade_on_close=trade_on_close,
                repeats=max(1, repeats),
                warmup=max(0, warmup),
            )
        )
    return results


def _status(results: list[ParityResult], tolerance: float) -> str:
    parity_results = [
        result for result in results if result.name in {"Tradelearn Engine", "Backtrader"}
    ]
    if len(parity_results) < 2:
        return "N/A"
    values = [result.final_value for result in parity_results]
    orders = {result.order_count for result in parity_results}
    value_ok = max(values) - min(values) < tolerance
    order_ok = len(orders) == 1
    return "EXACT" if value_ok and order_ok else "DIFF"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit target-weight portfolio parity with sell-first order_target_percent "
            "semantics. Increase --symbols/--bars for the 1000-symbol pressure test."
        )
    )
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--holdings", type=int, default=10)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--match-mode", default="exact")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--no-lite", action="store_true")
    parser.add_argument("--no-backtrader", action="store_true")
    parser.add_argument(
        "--next-open",
        action="store_true",
        help="use next-open execution instead of cheat-on-close; parity gate is intended for close execution",
    )
    args = parser.parse_args()

    results = run_parity_benchmark(
        symbols=args.symbols,
        bars=args.bars,
        holdings=args.holdings,
        rebalance_every=args.rebalance_every,
        cash=args.cash,
        seed=args.seed,
        match_mode=args.match_mode,
        repeats=args.repeat,
        warmup=args.warmup,
        include_lite=not args.no_lite,
        include_backtrader=not args.no_backtrader,
        trade_on_close=not args.next_open,
    )
    total_bars = args.symbols * args.bars
    print("Target Weight Parity Benchmark")
    print(
        f"symbols={args.symbols} bars={args.bars} holdings={args.holdings} "
        f"rebalance_every={args.rebalance_every} total_data_bars={total_bars:,}"
    )
    print(
        f"{'Runner':<20} | {'Final Value':>14} | {'Time':>10} | "
        f"{'Bars/s':>12} | {'Completed':>9} | {'Submitted':>9} | "
        f"{'Intents':>8} | {'Targets':>8}"
    )
    print("-" * 104)
    for result in results:
        print(
            f"{result.name:<20} | {result.final_value:>14,.2f} | "
            f"{result.elapsed_seconds:>8.3f}s | {result.bars_per_sec:>12,.0f} | "
            f"{result.order_count:>9,} | {result.submitted_count:>9,} | "
            f"{result.intent_count:>8,} | {result.target_count:>8,}"
        )
    print("-" * 104)
    print("Order status counts:")
    for result in results:
        statuses = ", ".join(f"{name}={count}" for name, count in result.status_counts) or "-"
        print(
            f"  {result.name}: {statuses}; "
            f"zero_size_returned={result.zero_size_orders:,}; "
            f"unnotified_returned={result.unnotified_orders:,}"
        )
        if result.unnotified_sample:
            print("    sample: " + " | ".join(result.unnotified_sample))
    status = _status(results, args.tolerance)
    print(f"Status: {status} (Tradelearn Engine vs Backtrader)")
    if status == "DIFF":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
