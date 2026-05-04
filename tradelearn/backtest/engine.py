from __future__ import annotations

import math
import sys
from typing import Any
from tradelearn.utils.console import smart_tqdm as tqdm, smart_print

import numpy as np
import pandas as pd

from tradelearn.backtest.models import BarSnapshot, Stats, SummaryDict
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.backtest.strategy import Strategy as CoreStrategy


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def getbyname(self, name: str) -> Any:
        return self[name]


def _build_observer_nexts(strategy: Any) -> tuple[Any, ...]:
    return tuple(
        observer.next
        for observer in getattr(strategy, "observers", {}).values()
        if callable(getattr(observer, "next", None))
    )


def _observer_step(observer_nexts: tuple[Any, ...]) -> None:
    for observer_next in observer_nexts:
        observer_next()


def _current_bar(data: Any) -> BarSnapshot:
    idx = max(0, int(getattr(data, "_cursor", 0)))
    timestamp = data.datetime[0] if hasattr(data, "datetime") else idx
    return BarSnapshot(
        datetime=timestamp,
        open=float(data.open[0]),
        high=float(data.high[0]),
        low=float(data.low[0]),
        close=float(data.close[0]),
        volume=float(data.volume[0]),
        data=data,
    )


def _build_analyzer_bar_callbacks(strategy: Any) -> tuple[Any, ...]:
    callbacks = []
    for analyzer in getattr(strategy, "analyzers", {}).values():
        on_bar = getattr(analyzer, "on_bar", None)
        if not callable(on_bar):
            continue
        if not getattr(analyzer, "is_streaming", False) and "on_bar" not in type(analyzer).__dict__:
            continue
        callbacks.append(on_bar)
    return tuple(callbacks)


def _analyzer_bar_step(strategy: Any, analyzer_bar_callbacks: tuple[Any, ...]) -> None:
    if not analyzer_bar_callbacks or strategy.data is None:
        return
    bar = _current_bar(strategy.data)
    for on_bar in analyzer_bar_callbacks:
        on_bar(bar)


def _orders_frame(broker: Any) -> pd.DataFrame:
    rows = []
    for order in getattr(broker, "_orders", []):
        data = getattr(order, "data", None)
        rows.append(
            {
                "ref": order.ref,
                "datetime": broker._fill_datetime(data) if data is not None else None,
                "data": getattr(data, "_name", None),
                "side": "buy" if order.isbuy() else "sell",
                "exectype": order.exectype,
                "status": order.getstatusname(),
                "size": order.size,
                "executed_size": order.executed.size,
                "executed_price": order.executed.price,
            }
        )
    return pd.DataFrame(rows)


def _fills_frame(broker: Any) -> pd.DataFrame:
    fills = pd.DataFrame(getattr(broker, "_fills", []))
    if fills.empty and hasattr(broker, "fills_frame") and not hasattr(broker, "_fills"):
        fills = broker.fills_frame()
    if fills.empty:
        return pd.DataFrame(columns=["order_ref", "data", "size", "price"])
    rows = fills.rename(columns={"ref": "order_ref"}).copy()
    if "datetime" in rows.columns:
        datetimes = rows["datetime"]
        if pd.api.types.is_numeric_dtype(datetimes):
            rows["datetime"] = pd.to_datetime(datetimes, unit="s", utc=True)
        else:
            rows["datetime"] = pd.to_datetime(datetimes, utc=True)
    if "data" not in rows.columns:
        rows["data"] = None
    return rows


def _trades_frame(fills: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "datetime",
        "data",
        "size",
        "price",
        "value",
        "commission",
        "pnl",
        "pnlcomm",
        "isopen",
        "isclosed",
        "dtopen",
        "dtclose",
    ]
    if fills.empty:
        return pd.DataFrame(columns=columns)

    position_size = 0.0
    avg_price = 0.0
    open_datetime = None
    rows: list[dict[str, Any]] = []
    for fill in fills.to_dict("records"):
        signed_size = float(fill.get("size", 0.0))
        price = float(fill.get("price", 0.0))
        comm = float(fill.get("commission", 0.0))
        curr_dt = fill.get("datetime")
        old_size = position_size
        new_size = old_size + signed_size
        if old_size == 0.0 and abs(new_size) > 1e-9:
            avg_price = price
            open_datetime = curr_dt
            rows.append(
                {
                    "datetime": curr_dt,
                    "data": fill.get("data"),
                    "size": new_size,
                    "price": price,
                    "value": abs(new_size) * price,
                    "commission": comm,
                    "pnl": 0.0,
                    "pnlcomm": -comm,
                    "isopen": True,
                    "isclosed": False,
                    "dtopen": open_datetime,
                    "dtclose": None,
                }
            )
        elif old_size * new_size <= 0:
            pnl = (price - avg_price) * old_size
            rows.append(
                {
                    "datetime": curr_dt,
                    "data": fill.get("data"),
                    "size": 0.0 if abs(new_size) < 1e-9 else new_size,
                    "price": price,
                    "value": abs(new_size) * price,
                    "commission": comm,
                    "pnl": pnl,
                    "pnlcomm": pnl - comm,
                    "isopen": False,
                    "isclosed": True,
                    "dtopen": open_datetime,
                    "dtclose": curr_dt,
                }
            )
            avg_price = price if abs(new_size) > 1e-9 else 0.0
            open_datetime = curr_dt if abs(new_size) > 1e-9 else None
        elif old_size * signed_size > 0:
            total_abs = abs(old_size) + abs(signed_size)
            avg_price = (abs(old_size) * avg_price + abs(signed_size) * price) / total_abs
        position_size = 0.0 if abs(new_size) < 1e-9 else new_size
    return pd.DataFrame(rows, columns=columns)


def _positions_frame(strategy: Any, fills: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    columns = [
        "datetime",
        "data",
        "size",
        "avg_price",
        "mark_price",
        "value",
        "unrealized_pnl",
        "realized_pnl",
        "margin_used",
    ]
    if fills.empty:
        return pd.DataFrame(columns=columns)

    def is_missing_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

    position_size = 0.0
    avg_price = 0.0
    realized_pnl = 0.0
    rows = []
    data_name = getattr(getattr(strategy, "data", None), "_name", None)
    sizes = fills["size"].to_numpy(dtype=float, copy=False)
    prices = fills["price"].to_numpy(dtype=float, copy=False)
    datetimes = fills["datetime"].to_numpy(copy=False) if "datetime" in fills else None
    data_values = fills["data"].to_numpy(copy=False) if "data" in fills else None
    fallback_datetime = index[-1] if len(index) else None

    for row_idx, (signed_size, price) in enumerate(zip(sizes, prices, strict=True)):
        previous_size = position_size
        new_size = position_size + signed_size
        if previous_size == 0 or previous_size * signed_size > 0:
            total_abs = abs(previous_size) + abs(signed_size)
            avg_price = (
                (abs(previous_size) * avg_price + abs(signed_size) * price) / total_abs
                if total_abs
                else 0.0
            )
        elif previous_size * new_size <= 0:
            realized_pnl += (price - avg_price) * previous_size
            avg_price = price if new_size else 0.0
        position_size = 0.0 if abs(new_size) < 1e-9 else new_size
        mark_price = price
        value = position_size * mark_price
        fill_datetime = datetimes[row_idx] if datetimes is not None else None
        if is_missing_scalar(fill_datetime):
            fill_datetime = fallback_datetime
        fill_data = data_values[row_idx] if data_values is not None else None
        if is_missing_scalar(fill_data):
            fill_data = data_name
        rows.append(
            {
                "datetime": fill_datetime,
                "data": fill_data,
                "size": position_size,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "value": value,
                "unrealized_pnl": (mark_price - avg_price) * position_size
                if position_size
                else 0.0,
                "realized_pnl": realized_pnl,
                "margin_used": abs(value),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _resolve_close_series(strategy: Any, frame: pd.DataFrame | None, index: pd.Index) -> np.ndarray | None:
    """Return per-bar close prices aligned to ``index``, or None if unavailable."""
    if isinstance(frame, pd.DataFrame) and "close" in frame.columns and len(frame) == len(index):
        return frame["close"].to_numpy(dtype=float, copy=False)
    data = getattr(strategy, "data", None)
    close_arr = getattr(data, "_close", None)
    if close_arr is not None:
        try:
            arr = np.asarray(close_arr, dtype=float)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and len(arr) == len(index):
            return arr
    cerebro = getattr(strategy, "cerebro", None)
    datas = getattr(cerebro, "datas", None) if cerebro is not None else None
    if datas:
        primary = datas[0]
        primary_frame = getattr(primary, "_frame", None)
        if (
            isinstance(primary_frame, pd.DataFrame)
            and "close" in primary_frame.columns
            and len(primary_frame) == len(index)
        ):
            return primary_frame["close"].to_numpy(dtype=float, copy=False)
        primary_close = getattr(primary, "_close", None)
        if primary_close is not None:
            try:
                arr = np.asarray(primary_close, dtype=float)
            except (TypeError, ValueError):
                arr = None
            if arr is not None and len(arr) == len(index):
                return arr
    return None


def _build_equity_from_fills(
    strategy: Any,
    index: pd.Index,
    fills: pd.DataFrame,
    frame: pd.DataFrame | None = None,
) -> pd.Series | None:
    if fills.empty or len(index) == 0:
        return None
    closes = _resolve_close_series(strategy, frame, index)
    if closes is None:
        return None

    aligned_fills = fills.copy()
    aligned_fills["datetime"] = pd.to_datetime(aligned_fills["datetime"], utc=True)
    try:
        bar_index = pd.DatetimeIndex(index)
        if bar_index.tz is None:
            bar_index = bar_index.tz_localize("UTC")
        else:
            bar_index = bar_index.tz_convert("UTC")
    except (TypeError, ValueError):
        return None

    fills_by_time: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for fill in aligned_fills.to_dict("records"):
        try:
            ts = pd.Timestamp(fill["datetime"]).tz_convert("UTC")
        except (TypeError, ValueError, AttributeError):
            continue
        fills_by_time.setdefault(ts, []).append(fill)

    # Use the immutable initial cash field (_cash), not getcash() which returns
    # the live active-cash balance after fills have settled.
    cash = float(getattr(strategy.broker, "_cash", None) or strategy.broker.getcash())
    position_size = 0.0
    values: list[float] = []
    for ts, close in zip(bar_index, closes, strict=False):
        for fill in fills_by_time.get(pd.Timestamp(ts), ()):
            signed_size = float(fill.get("size", 0.0))
            price = float(fill.get("price", 0.0))
            commission = float(fill.get("commission", 0.0))
            cash -= signed_size * price + commission
            position_size += signed_size
        values.append(cash + position_size * float(close))
    return pd.Series(values, index=index, name="equity", dtype=float)


def _build_equity_returns(
    strategy: Any,
    index: pd.Index,
    fills: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    if fills is not None:
        fill_equity = _build_equity_from_fills(strategy, index, fills, frame)
        if fill_equity is not None:
            returns = fill_equity.pct_change().fillna(0.0)
            returns.name = "returns"
            return fill_equity, returns

    values = []
    observers_get = getattr(
        getattr(strategy, "observers", {}),
        "get",
        lambda key, default=None: None,
    )
    observer_value = observers_get("value", None)
    if observer_value is not None:
        values = list(getattr(observer_value, "values", []))
    if not values:
        values = [float(strategy.broker.getvalue())] * len(index)
    if len(values) < len(index):
        fill_value = values[-1] if values else float(strategy.broker.getvalue())
        values.extend([fill_value] * (len(index) - len(values)))
    equity = pd.Series(values[: len(index)], index=index, name="equity", dtype=float)
    returns = equity.pct_change().fillna(0.0)
    returns.name = "returns"
    return equity, returns


def _stats_config(cerebro: Any) -> dict[str, Any]:
    config = BacktestRuntimeConfig.from_owner(cerebro)
    
    # Extract strategy name from cerebro's strategy list if available.
    strategy_name = "TradeLearn Report"
    if hasattr(cerebro, "strats") and cerebro.strats:
        strategy_cls = cerebro.strats[0][0]
        if hasattr(strategy_cls, "__name__"):
            strategy_name = strategy_cls.__name__

    return {
        "strategy": strategy_name,
        "callback_batch": getattr(cerebro, "callback_batch", 1),
        "trade_on_close": config.trade_on_close,
        "exactbars": config.exactbars,
        "stdstats": config.stdstats,
        "broker": {
            "cash": config.cash,
            "commission": config.commission,
        },
    }


def _return_pct(final_value: float, broker: Any) -> float:
    start_cash = float(getattr(broker, "_cash", 0.0) or 0.0)
    if start_cash == 0.0:
        return float("nan")
    return (float(final_value) / start_cash - 1.0) * 100.0


def _trade_summary(broker: Any, trades: pd.DataFrame | None = None) -> tuple[float, float]:
    trade_summary = getattr(broker, "trade_summary", None)
    if callable(trade_summary):
        total, wins = trade_summary()
        return float(total), float(wins)
    if trades is not None and not trades.empty and "isclosed" in trades.columns:
        closed = trades[trades["isclosed"].astype(bool)]
        total = float(len(closed))
        wins = float((closed["pnl"] > 0).sum()) if total > 0 else 0.0
        return total, wins
    return 0.0, 0.0


def _win_rate_pct(total_trades: float, wins: float) -> float:
    return wins / total_trades * 100.0 if total_trades else 0.0


def _summary_pnl(positions: pd.DataFrame) -> tuple[float, float, float]:
    """Read realized/unrealized pnl and margin from the last positions row."""
    if positions is None or positions.empty:
        return 0.0, 0.0, 0.0
    last = positions.iloc[-1]

    def _f(value: Any) -> float:
        try:
            if value is None or pd.isna(value):
                return 0.0
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return (
        _f(last.get("realized_pnl", 0.0)),
        _f(last.get("unrealized_pnl", 0.0)),
        _f(last.get("margin_used", 0.0)),
    )


def _summary_max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    cummax = equity.cummax().replace(0, np.nan)
    dd = (cummax - equity) / cummax
    return float(dd.fillna(0.0).max())


def _summary_sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe using the metrics module; return 0.0 on degenerate input."""
    if returns is None or returns.empty:
        return 0.0
    if returns[returns != 0.0].shape[0] < 2:
        return 0.0
    try:
        from tradelearn.metrics.risk import sharpe as _sharpe

        value = _sharpe(returns, periods=periods)
    except Exception:
        return 0.0
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _summary_annual_return(returns: pd.Series, periods: int = 252) -> float:
    if returns is None or returns.empty or returns[returns != 0.0].shape[0] < 2:
        return 0.0
    try:
        from tradelearn.metrics.returns import annual_return as _ar
        return _safe_float(_ar(returns, periods=periods)) * 100.0
    except Exception:
        return 0.0


def _summary_volatility(returns: pd.Series, periods: int = 252) -> float:
    if returns is None or returns.empty or returns[returns != 0.0].shape[0] < 2:
        return 0.0
    try:
        from tradelearn.metrics.risk import volatility as _vol
        return _safe_float(_vol(returns, periods=periods)) * 100.0
    except Exception:
        return 0.0


def _summary_sortino(returns: pd.Series, periods: int = 252) -> float:
    if returns is None or returns.empty or returns[returns != 0.0].shape[0] < 2:
        return 0.0
    try:
        from tradelearn.metrics.risk import sortino as _sortino
        return _safe_float(_sortino(returns, periods=periods))
    except Exception:
        return 0.0


def _summary_calmar(returns: pd.Series, periods: int = 252) -> float:
    if returns is None or returns.empty or returns[returns != 0.0].shape[0] < 2:
        return 0.0
    try:
        from tradelearn.metrics.risk import calmar as _calmar
        return _safe_float(_calmar(returns, periods=periods))
    except Exception:
        return 0.0


def _summary_trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    empty = {
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "max_consec_wins": 0,
        "max_consec_losses": 0,
        "total_commission": 0.0,
    }
    if trades is None or trades.empty:
        return empty
    closed = trades[trades.get("isclosed", pd.Series(dtype=bool)).astype(bool)] if "isclosed" in trades.columns else trades
    if closed.empty:
        return empty
    pnl = closed["pnlcomm"] if "pnlcomm" in closed.columns else closed.get("pnl", pd.Series(dtype=float))
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 1e-9 else 0.0
    expectancy = float(pnl.mean()) if not pnl.empty else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    best = float(pnl.max()) if not pnl.empty else 0.0
    worst = float(pnl.min()) if not pnl.empty else 0.0
    # consecutive streaks
    signs = (pnl > 0).astype(int)
    max_w = max_l = cur_w = cur_l = 0
    for s in signs:
        if s:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    commission = float(closed["commission"].sum()) if "commission" in closed.columns else 0.0
    return {
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best,
        "worst_trade": worst,
        "max_consec_wins": max_w,
        "max_consec_losses": max_l,
        "total_commission": commission,
    }


def _summary_exposure_pct(positions: pd.DataFrame) -> float:
    if positions is None or positions.empty or "size" not in positions.columns:
        return 0.0
    exposed_rows = (positions["size"].abs() > 1e-9).sum()
    return float(exposed_rows) / len(positions) * 100.0


def _summary_drawdown_stats(equity: pd.Series) -> dict[str, Any]:
    if equity is None or equity.empty:
        return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "max_dd_duration": pd.Timedelta(0), "avg_dd_duration": pd.Timedelta(0)}
    cummax = equity.cummax()
    dd = (cummax - equity) / cummax
    dd_mask = dd > 0
    # Calculate durations
    dd_groups = (dd_mask != dd_mask.shift()).cumsum()
    durations = []
    for _name, group in dd[dd_mask].groupby(dd_groups):
        if len(group) > 0:
            durations.append(group.index[-1] - group.index[0])
    
    return {
        "max_drawdown": float(dd.max()),
        "avg_drawdown": float(dd[dd_mask].mean()) if dd_mask.any() else 0.0,
        "max_dd_duration": max(durations) if durations else pd.Timedelta(0),
        "avg_dd_duration": sum(durations, pd.Timedelta(0)) / len(durations) if durations else pd.Timedelta(0),
    }


def _summary_trade_stats(trades: pd.DataFrame, start_cash: float) -> dict[str, Any]:
    empty = {
        "best_trade_pct": 0.0, "worst_trade_pct": 0.0, "avg_trade_pct": 0.0,
        "max_trade_duration": pd.Timedelta(0), "avg_trade_duration": pd.Timedelta(0),
        "sqn": 0.0, "kelly_criterion": 0.0
    }
    if trades is None or trades.empty:
        return empty
    closed = trades[trades.get("isclosed", pd.Series(dtype=bool)).astype(bool)] if "isclosed" in trades.columns else trades
    if closed.empty:
        return empty
    
    pnl_pct = (closed["pnlcomm"] / start_cash * 100.0) if start_cash > 0 else pd.Series(dtype=float)
    durations = closed["dtclose"] - closed["dtopen"]
    
    wins = closed[closed["pnlcomm"] > 0]
    win_rate = len(wins) / len(closed)
    avg_win = wins["pnlcomm"].mean() if not wins.empty else 0.0
    avg_loss = closed[closed["pnlcomm"] < 0]["pnlcomm"].mean() if not closed[closed["pnlcomm"] < 0].empty else 1e-9
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0
    
    # Kelly: W - (1-W)/R
    kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0
    
    # SQN: sqrt(N) * mean / std
    mean_pnl = closed["pnlcomm"].mean()
    std_pnl = closed["pnlcomm"].std()
    sqn = (math.sqrt(len(closed)) * mean_pnl / std_pnl) if std_pnl > 1e-9 else 0.0

    return {
        "best_trade_pct": float(pnl_pct.max()) if not pnl_pct.empty else 0.0,
        "worst_trade_pct": float(pnl_pct.min()) if not pnl_pct.empty else 0.0,
        "avg_trade_pct": float(pnl_pct.mean()) if not pnl_pct.empty else 0.0,
        "max_trade_duration": durations.max(),
        "avg_trade_duration": durations.mean(),
        "sqn": sqn,
        "kelly_criterion": kelly,
    }


def _build_stats(cerebro: Any, strategy: Any, *, lazy_artifacts: bool = False) -> Stats:
    data = strategy.data
    frame = getattr(data, "_frame", None)
    if not isinstance(frame, pd.DataFrame):
        primary = cerebro.datas[0] if getattr(cerebro, "datas", None) else None
        frame = getattr(primary, "_frame", pd.DataFrame())
    index = frame.index

    if lazy_artifacts:
        final_cash = float(strategy.broker.getcash())
        final_value = float(strategy.broker.getvalue())
        total_orders = float(len(getattr(strategy.broker, "_orders", ())))
        total_fills = float(len(getattr(strategy.broker, "_fills", ())))

        # Build summary inputs eagerly (cheap: bounded by # of fills),
        # but expose DataFrames lazily through factories.
        eager_fills = _fills_frame(strategy.broker)
        eager_trades = _trades_frame(eager_fills)
        eager_positions = _positions_frame(strategy, eager_fills, index)
        eager_equity, eager_returns = _build_equity_returns(strategy, index, eager_fills, frame)
        total_trades, winning_trades = _trade_summary(strategy.broker, eager_trades)
        realized_pnl, unrealized_pnl, margin_used = _summary_pnl(eager_positions)

        def equity_factory() -> pd.Series:
            return eager_equity

        def returns_factory() -> pd.Series:
            return eager_returns

        def fills_factory() -> pd.DataFrame:
            return eager_fills

        def trades_factory() -> pd.DataFrame:
            return eager_trades

        def positions_factory() -> pd.DataFrame:
            return eager_positions

        trade_metrics = _summary_trade_metrics(eager_trades)
        dd_stats = _summary_drawdown_stats(eager_equity)
        initial_cash = float(strategy.broker._cash)
        ext_trade_stats = _summary_trade_stats(eager_trades, initial_cash)

        summary = SummaryDict({
            "bars": float(len(index)),
            "start": index[0] if not index.empty else None,
            "end": index[-1] if not index.empty else None,
            "duration": (index[-1] - index[0]) if not index.empty else None,
            "exposure_pct": _summary_exposure_pct(eager_positions),
            "final_cash": final_cash,
            "final_value": final_value,
            "peak_value": float(eager_equity.max()) if not eager_equity.empty else final_value,
            "return_pct": _return_pct(final_value, strategy.broker),
            "annual_return": _summary_annual_return(eager_returns),
            "volatility": _summary_volatility(eager_returns),
            "sharpe": _summary_sharpe(eager_returns),
            "sortino": _summary_sortino(eager_returns),
            "calmar": _summary_calmar(eager_returns),
            "max_drawdown": dd_stats["max_drawdown"],
            "avg_drawdown": dd_stats["avg_drawdown"],
            "max_dd_duration": dd_stats["max_dd_duration"],
            "avg_dd_duration": dd_stats["avg_dd_duration"],
            "total_trades": total_trades,
            "win_rate_pct": _win_rate_pct(total_trades, winning_trades),
            "profit_factor": trade_metrics["profit_factor"],
            "expectancy": trade_metrics["expectancy"],
            "best_trade_pct": ext_trade_stats["best_trade_pct"],
            "worst_trade_pct": ext_trade_stats["worst_trade_pct"],
            "avg_trade_pct": ext_trade_stats["avg_trade_pct"],
            "max_trade_duration": ext_trade_stats["max_trade_duration"],
            "avg_trade_duration": ext_trade_stats["avg_trade_duration"],
            "sqn": ext_trade_stats["sqn"],
            "kelly_criterion": ext_trade_stats["kelly_criterion"],
        })
        
        stats = Stats(
            returns=returns_factory,
            equity=equity_factory,
            trades=trades_factory,
            positions=positions_factory,
            orders=lambda: _orders_frame(strategy.broker),
            summary=summary,
            analyzers={},
            config=_stats_config(cerebro),
            fills=fills_factory,
        )
        return stats

    fills = _fills_frame(strategy.broker)
    equity, returns = _build_equity_returns(strategy, index, fills, frame)
    trades = _trades_frame(fills)
    orders = _orders_frame(strategy.broker)
    positions = _positions_frame(strategy, fills, index)
    final_cash = float(strategy.broker.getcash())
    final_value = float(strategy.broker.getvalue())
    total_trades, winning_trades = _trade_summary(strategy.broker, trades)
    realized_pnl, unrealized_pnl, margin_used = _summary_pnl(positions)
    trade_metrics = _summary_trade_metrics(trades)
    dd_stats = _summary_drawdown_stats(equity)
    initial_cash = float(strategy.broker._cash)
    ext_trade_stats = _summary_trade_stats(trades, initial_cash)

    summary = SummaryDict({
        "bars": float(len(index)),
        "start": index[0] if not index.empty else None,
        "end": index[-1] if not index.empty else None,
        "duration": (index[-1] - index[0]) if not index.empty else None,
        "exposure_pct": _summary_exposure_pct(positions),
        "final_cash": final_cash,
        "final_value": final_value,
        "peak_value": float(equity.max()) if not equity.empty else final_value,
        "return_pct": _return_pct(final_value, strategy.broker),
        "annual_return": _summary_annual_return(returns),
        "volatility": _summary_volatility(returns),
        "sharpe": _summary_sharpe(returns),
        "sortino": _summary_sortino(returns),
        "calmar": _summary_calmar(returns),
        "max_drawdown": dd_stats["max_drawdown"],
        "avg_drawdown": dd_stats["avg_drawdown"],
        "max_dd_duration": dd_stats["max_dd_duration"],
        "avg_dd_duration": dd_stats["avg_dd_duration"],
        "total_trades": total_trades,
        "win_rate_pct": _win_rate_pct(total_trades, winning_trades),
        "profit_factor": trade_metrics["profit_factor"],
        "expectancy": trade_metrics["expectancy"],
        "best_trade_pct": ext_trade_stats["best_trade_pct"],
        "worst_trade_pct": ext_trade_stats["worst_trade_pct"],
        "avg_trade_pct": ext_trade_stats["avg_trade_pct"],
        "max_trade_duration": ext_trade_stats["max_trade_duration"],
        "avg_trade_duration": ext_trade_stats["avg_trade_duration"],
        "sqn": ext_trade_stats["sqn"],
        "kelly_criterion": ext_trade_stats["kelly_criterion"],
        "final_realized_pnl": realized_pnl,
        "final_unrealized_pnl": unrealized_pnl,
        "final_margin_used": margin_used,
        "total_orders": float(len(orders)),
        "total_fills": float(len(fills)),
    })
    stats = Stats(
        returns=returns,
        equity=equity,
        trades=trades,
        positions=positions,
        orders=orders,
        summary=summary,
        analyzers={},
        config=_stats_config(cerebro),
        fills=fills,
    )
    smart_print(stats.summary, file=sys.stderr)
    return stats


def _build_bar_advancers(
    strategy: Any,
    datas: list[Any],
    indicators: list[Any],
    *,
    include_data: bool = True,
) -> tuple[Any, ...]:
    """Build a stable per-bar advance plan for data feeds and indicators."""
    bar_advancers = []
    seen_advancer_ids = set()
    for data in datas:
        if include_data:
            bar_advancers.append(data._advance)
        seen_advancer_ids.add(id(data))
    for indicator in indicators:
        advance = getattr(indicator, "_advance", None)
        if callable(advance) and id(indicator) not in seen_advancer_ids:
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(indicator))
    strategy_advance = getattr(strategy, "_advance", None)
    strategy_lines = getattr(strategy, "lines", None)
    strategy_has_lines = strategy_lines is None or len(strategy_lines) > 0
    if strategy_advance is not None and strategy_has_lines:
        bar_advancers.append(strategy_advance)
    for attr, val in strategy.__dict__.items():
        if attr.startswith("_"):
            continue
        if id(val) in seen_advancer_ids:
            continue
        advance = getattr(val, "_advance", None)
        if callable(advance):
            bar_advancers.append(advance)
            seen_advancer_ids.add(id(val))
    return tuple(bar_advancers)


def _build_data_advance_plan(datas: list[Any]) -> Any | None:
    """Build a Rust primary-clock cursor runner for multi-data runs."""
    if len(datas) <= 1:
        return None
    primary_timestamps = np.asarray(datas[0]._datetime, dtype=np.int64)
    if all(
        len(data._datetime) == len(primary_timestamps)
        and np.array_equal(np.asarray(data._datetime, dtype=np.int64), primary_timestamps)
        for data in datas[1:]
    ):
        return None
    try:
        from tradelearn._rust import RustBarRunner
    except (ImportError, AttributeError):
        try:
            from tradelearn._rust import RustPrimaryClockPlan as RustBarRunner
        except (ImportError, AttributeError):
            return None
    return RustBarRunner(
        primary_timestamps,
        [np.asarray(data._datetime, dtype=np.int64) for data in datas[1:]],
    )


def _build_clocked_multi_data_runner(datas: list[Any]) -> Any | None:
    """Build the Rust clocked multi-data runner when the feed schema is supported."""
    if len(datas) <= 1:
        return None
    try:
        from tradelearn._rust import RustClockedMultiDataRunner
    except (ImportError, AttributeError):
        return None

    symbols: list[str] = []
    timestamps: list[np.ndarray] = []
    opens: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    volumes: list[np.ndarray] = []
    for index, data in enumerate(datas):
        required = ("_datetime", "_open", "_high", "_low", "_close", "_volume")
        if not all(hasattr(data, attr) for attr in required):
            return None
        symbols.append(str(getattr(data, "_name", None) or f"data{index}"))
        timestamps.append(np.asarray(data._datetime, dtype=np.int64))
        opens.append(np.asarray(data._open, dtype=np.float64))
        highs.append(np.asarray(data._high, dtype=np.float64))
        lows.append(np.asarray(data._low, dtype=np.float64))
        closes.append(np.asarray(data._close, dtype=np.float64))
        volumes.append(np.asarray(data._volume, dtype=np.float64))
    return RustClockedMultiDataRunner(symbols, timestamps, opens, highs, lows, closes, volumes)


def run_backtest(cerebro: Any) -> list[Any]:
    """Unified backtest engine that runs any strategy inheriting from core.Strategy."""
    runtime_config = BacktestRuntimeConfig.from_owner(cerebro)
    cerebro.runtime_config = runtime_config
    strategy_cls, args, kwargs = cerebro.strats[0]
    bind_strategy_context = getattr(cerebro, "_bind_strategy_context", None)
    strategy = strategy_cls(*args, **kwargs)
    strategy.cerebro = cerebro
    if callable(bind_strategy_context):
        bind_strategy_context(strategy)

    # Core attributes
    strategy.datas = cerebro.datas
    if cerebro.datas:
        strategy.data = cerebro.datas[0]
    strategy.broker = cerebro.broker

    # ... (Rust Engine Initialization omitted for brevity but preserved in real file) ...
    # ---------------------------------------------------------
    # Rust Engine Initialization
    # ---------------------------------------------------------
    from .broker import RustBroker

    if isinstance(cerebro.broker, RustBroker):
        data = cerebro.datas[0]
        # Robustly handle both native containers and facade data feeds.
        if hasattr(data, "_datetime"):
            timestamps = data._datetime
            opens = data._open
            highs = data._high
            lows = data._low
            closes = data._close
            volumes = data._volume
        else:  # LineSeries fallback
            timestamps = np.array(data.datetime._values, dtype=np.int64)
            opens = np.array(data.open._values, dtype=np.float64)
            highs = np.array(data.high._values, dtype=np.float64)
            lows = np.array(data.low._values, dtype=np.float64)
            closes = np.array(data.close._values, dtype=np.float64)
            volumes = np.array(data.volume._values, dtype=np.float64)

        try:
            from tradelearn._rust import RustBacktestEngine
        except (ImportError, AttributeError):
            RustBacktestEngine = None

        if RustBacktestEngine is not None:
            rust_engine = RustBacktestEngine(
                timestamps,
                opens,
                highs,
                lows,
                closes,
                volumes,
                runtime_config.cash,
                runtime_config.commission,
                runtime_config.trade_on_close,
                False,
                False,
                0.0,
                0.0,
                False,
                False,
                False,
                float(cerebro.broker._mult),
                1.0,
                runtime_config.match_mode == "smart",
            )
            cerebro.broker.bind_engine(rust_engine)
        if hasattr(cerebro.broker, "bind_datas"):
            cerebro.broker.bind_datas(cerebro.datas)
        cerebro.broker._open_prices = opens
        cerebro.broker._high_prices = highs
        cerebro.broker._low_prices = lows
        cerebro.broker._close_prices = closes

    # 2. Initialize Sizer & Analyzers
    sizer_cls, sizer_kwargs = cerebro._sizer_spec
    strategy.setsizer(sizer_cls(**sizer_kwargs))

    # Support indicators from both facades
    indicators = getattr(strategy, "_indicators", [])
    indicators_bt = getattr(strategy, "_indicators_bt", [])

    strategy.analyzers = _AttrDict()
    for name, ana_spec in cerebro.analyzers.items():
        if len(ana_spec) == 3:
            ana_cls, ana_args, ana_kwargs = ana_spec
        else:
            ana_cls, ana_kwargs = ana_spec
            ana_args = ()
        ana_inst = ana_cls(*ana_args, **ana_kwargs)
        ana_inst.strategy = strategy
        strategy.analyzers[name] = ana_inst

    attach_observers = getattr(cerebro, "_attach_observers", None)
    if callable(attach_observers):
        attach_observers(strategy)

    # 3. Lifecycle Start
    if hasattr(strategy, "_setup"):
        strategy._setup()
    strategy.init()
    strategy.start()
    for ana in strategy.analyzers.values():
        on_start = getattr(ana, "on_start", None)
        if callable(on_start):
            on_start()
        if hasattr(ana, "start"):
            ana.start()
    for observer in getattr(strategy, "observers", {}).values():
        observer.start()
    analyzer_bar_callbacks = _build_analyzer_bar_callbacks(strategy)
    observer_nexts = _build_observer_nexts(strategy)
    has_analyzer_bar_callbacks = bool(analyzer_bar_callbacks)
    has_observer_nexts = bool(observer_nexts)

    limit = cerebro.datas[0].buflen()
    # Calculate min_period from all indicators (mostly for BT facade)
    strategy_min_period = getattr(type(strategy), "min_period", 0)
    if callable(strategy_min_period):
        strategy_min_period = strategy_min_period(strategy)
    explicit_strategy_min_period = int(strategy_min_period or 0)
    min_period = int(getattr(strategy, "_manual_min_period", 0))
    for ind in indicators + indicators_bt:
        if hasattr(ind, "min_period"):
            m = ind.min_period
            if callable(m):
                m = m()
            min_period = max(min_period, int(m))
    if min_period == 0:
        min_period = 1

    broker = cerebro.broker
    use_clocked_multi_data_runner = (
        isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and len(cerebro.datas) > 1
    )
    clocked_multi_data_runner = (
        _build_clocked_multi_data_runner(cerebro.datas) if use_clocked_multi_data_runner else None
    )
    data_advance_plan = (
        None if clocked_multi_data_runner is not None else _build_data_advance_plan(cerebro.datas)
    )
    data_advance_methods = tuple(data._advance for data in cerebro.datas)
    bar_advancers = _build_bar_advancers(
        strategy,
        cerebro.datas,
        indicators + indicators_bt,
        include_data=data_advance_plan is None and clocked_multi_data_runner is None,
    )

    if data_advance_plan is None:

        def advance_datas(i: int) -> None:
            return None
    else:

        def advance_datas(i: int) -> None:
            for advance_data, cursor in zip(
                data_advance_methods,
                data_advance_plan.cursors_at(i),
                strict=False,
            ):
                if cursor >= 0:
                    advance_data(cursor)
                else:
                    advance_data(-1)

    if hasattr(strategy, "_set_bar_advancers"):
        if data_advance_plan is None:
            strategy._set_bar_advancers(bar_advancers)
        else:

            def advance_bar(i: int) -> None:
                advance_datas(i)
                for advance in bar_advancers:
                    advance(i)

            strategy._set_bar_advancers((advance_bar,))
        strategy_pre_next = strategy._pre_next
    else:

        def strategy_pre_next(cursor: int) -> None:
            advance_datas(cursor)
            for advance in bar_advancers:
                advance(cursor)

    notify_cashvalue = None
    if type(strategy).notify_cashvalue is not CoreStrategy.notify_cashvalue:
        notify_cashvalue = strategy.notify_cashvalue
    broker_step = broker.step if broker else None
    broker_process_fills = broker.process_fills if broker else None
    broker_getcash = broker.getcash if broker else None
    broker_getvalue = broker.getvalue if broker else None
    begin_order_buffering = getattr(broker, "begin_order_buffering", None) if broker else None
    flush_order_buffer = getattr(broker, "flush_order_buffer", None) if broker else None
    drain_order_buffer = getattr(broker, "drain_order_buffer", None) if broker else None
    begin_terminal_order_suppression = (
        getattr(broker, "begin_terminal_order_suppression", None) if broker else None
    )
    end_terminal_order_suppression = (
        getattr(broker, "end_terminal_order_suppression", None) if broker else None
    )
    strategy_next = strategy.next

    use_rust_bar_loop = (
        isinstance(broker, RustBroker)
        and getattr(broker, "_engine", None) is not None
        and hasattr(broker._engine, "run_bar_loop")
        and len(cerebro.datas) <= 1
        and not bool(getattr(cerebro, "trade_on_close", False))
    )
    min_start = max(min_period - 1, explicit_strategy_min_period)

    class _RunStopped(Exception):
        """Internal sentinel used to break out of Rust callback-driven loops."""

    def raise_if_runstopped() -> None:
        if bool(getattr(cerebro, "_runstop", False)):
            raise _RunStopped

    def terminal_bar(i: int) -> bool:
        return i >= limit - 1

    def run_strategy_next_python(i: int) -> None:
        if terminal_bar(i) and begin_terminal_order_suppression is not None:
            begin_terminal_order_suppression()
            try:
                strategy_next()
            finally:
                if end_terminal_order_suppression is not None:
                    end_terminal_order_suppression()
            return
        if begin_order_buffering is not None:
            begin_order_buffering()
            strategy_next()
            flush_order_buffer()
            if getattr(broker, "_trade_on_close", False):
                broker_process_fills(strategy, i)
        else:
            strategy_next()

    def run_strategy_next_drained(i: int) -> list[Any] | None:
        if terminal_bar(i) and begin_terminal_order_suppression is not None:
            begin_terminal_order_suppression()
            try:
                strategy_next()
            finally:
                if end_terminal_order_suppression is not None:
                    end_terminal_order_suppression()
            return None
        begin_order_buffering()
        strategy_next()
        orders = drain_order_buffer()
        return orders if orders else None

    def on_bar(i: int) -> list[Any]:
        strategy_pre_next(i)

        # Broker Match
        if broker:
            broker_step(i)
            broker_process_fills(strategy, i)
            if notify_cashvalue is not None:
                notify_cashvalue(broker_getcash(), broker_getvalue())

        # Strategy Next
        if i >= min_start:
            run_strategy_next_python(i)
            if bool(getattr(cerebro, "_runstop", False)):
                return []
            if has_analyzer_bar_callbacks:
                _analyzer_bar_step(strategy, analyzer_bar_callbacks)
            if has_observer_nexts:
                _observer_step(observer_nexts)
        return []

    use_multi_data_rust_runner = clocked_multi_data_runner is not None
    pbar = tqdm(total=limit, desc="Backtest.run", unit="bar", leave=True, delay=0)
    if use_multi_data_rust_runner:
        if getattr(broker, "_trade_on_close", False):

            def on_rust_bar_multi(
                i: int,
                data_cursors: list[int],
                fills: list[Any],
                cash: float,
                size: float,
                price: float,
            ) -> list[Any] | None:
                pbar.update(1)
                for advance_data, cursor in zip(data_advance_methods, data_cursors, strict=False):
                    advance_data(cursor if cursor >= 0 else -1)
                for advance in bar_advancers:
                    advance(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                if notify_cashvalue is not None:
                    notify_cashvalue(broker_getcash(), broker_getvalue())
                if i >= min_start:
                    run_strategy_next_python(i)
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                return None
        elif notify_cashvalue is None:

            def on_rust_bar_multi(
                i: int,
                data_cursors: list[int],
                fills: list[Any],
                cash: float,
                size: float,
                price: float,
            ) -> list[Any] | None:
                pbar.update(1)
                for advance_data, cursor in zip(data_advance_methods, data_cursors, strict=False):
                    advance_data(cursor if cursor >= 0 else -1)
                for advance in bar_advancers:
                    advance(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                if i >= min_start:
                    orders = run_strategy_next_drained(i)
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders
                return None
        else:

            def on_rust_bar_multi(
                i: int,
                data_cursors: list[int],
                fills: list[Any],
                cash: float,
                size: float,
                price: float,
            ) -> list[Any] | None:
                pbar.update(1)
                for advance_data, cursor in zip(data_advance_methods, data_cursors, strict=False):
                    advance_data(cursor if cursor >= 0 else -1)
                for advance in bar_advancers:
                    advance(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                notify_cashvalue(broker_getcash(), broker_getvalue())
                if i >= min_start:
                    orders = run_strategy_next_drained(i)
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders
                return None

        try:
            clocked_multi_data_runner.run(broker._engine, broker, on_rust_bar_multi, 0, limit)
        except _RunStopped:
            pass
    elif use_rust_bar_loop:
        if notify_cashvalue is None:

            def on_rust_bar(
                i: int, fills: list[Any], cash: float, size: float, price: float
            ) -> list[Any] | None:
                pbar.update(1)
                strategy_pre_next(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                if i >= min_start:
                    orders = run_strategy_next_drained(i)
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders
                return None
        else:

            def on_rust_bar(
                i: int, fills: list[Any], cash: float, size: float, price: float
            ) -> list[Any] | None:
                pbar.update(1)
                strategy_pre_next(i)
                broker._curr_idx = i
                broker._step_fills_from_collect = fills
                broker._rust_state_cache = (i, cash, size, price)
                if fills:
                    broker_process_fills(strategy, i)
                notify_cashvalue(broker_getcash(), broker_getvalue())
                if i >= min_start:
                    orders = run_strategy_next_drained(i)
                    raise_if_runstopped()
                    if has_analyzer_bar_callbacks:
                        _analyzer_bar_step(strategy, analyzer_bar_callbacks)
                    if has_observer_nexts:
                        _observer_step(observer_nexts)
                    return orders
                return None

        try:
            broker._engine.run_bar_loop(broker, on_rust_bar, 0, limit)
        except _RunStopped:
            pass
    else:
        for i in range(limit):
            on_bar(i)
            pbar.update(1)
            if bool(getattr(cerebro, "_runstop", False)):
                break

    pbar.close()

    # 4. Lifecycle Stop
    strategy.stop()
    metrics_engine = getattr(strategy, "metrics_engine", None)
    lazy_stats = (
        getattr(cerebro, "stats_mode", "full") == "lazy"
        and metrics_engine is None
        and not strategy.analyzers
    )
    stats = _build_stats(cerebro, strategy, lazy_artifacts=lazy_stats)
    if metrics_engine is not None:
        metrics_engine.compute(stats)
    for ana in strategy.analyzers.values():
        on_end = getattr(ana, "on_end", None)
        if callable(on_end):
            on_end(stats)
    analyzer_results = {name: ana.get_analysis() for name, ana in strategy.analyzers.items()}
    stats.analyzers = analyzer_results
    strategy.stats = stats
    strategy.analyzer_results = analyzer_results
    cerebro.stats = stats
    cerebro.analyzer_results = analyzer_results
    for ana in strategy.analyzers.values():
        if hasattr(ana, "stop"):
            ana.stop()
    for observer in getattr(strategy, "observers", {}).values():
        observer.stop()

    return [strategy]
