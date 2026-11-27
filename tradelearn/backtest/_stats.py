"""Backtest frame construction and summary statistics builders."""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.models import Stats, SummaryDict
from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.core.annualization import resolve_periods


def _orders_frame(broker: Any) -> pd.DataFrame:
    rows = []
    for order in getattr(broker, "_orders", []):
        data = getattr(order, "data", None)
        rows.append(
            {
                "ref": order.ref,
                "datetime": order.created_ts,
                "data": getattr(data, "_name", None),
                "side": "buy" if order.isbuy() else "sell",
                "exectype": order.exectype,
                "status": order.getstatusname(),
                "price": order.price,
                "pricelimit": order.pricelimit,
                "parent_ref": getattr(order.parent, "ref", None),
                "oco_ref": getattr(order.oco, "ref", None),
                "size": order.size,
                "executed_size": order.executed.size,
                "executed_price": order.executed.price,
                "executed_dt": getattr(order, "_executed_dt_raw", order.executed.dt),
                "tag": order.info.get("tag"),
                "info": dict(order.info),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("datetime", "executed_dt"):
        if not frame.empty:
            values = frame[column]
            if pd.api.types.is_numeric_dtype(values):
                frame[column] = pd.to_datetime(values, unit="s", utc=True)
            else:
                frame[column] = pd.to_datetime(values, utc=True)
    return frame


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

    # Per-asset flat-to-flat lifecycle; partial exits retain realized PnL.
    positions: dict[Any, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def open_position(data: Any, size: float, price: float, fee: float, dt: Any) -> None:
        positions[data] = dict(size=size, price=price, commission=fee, pnl=0.0, dtopen=dt)
        rows.append(dict(datetime=dt, data=data, size=size, price=price,
                         value=abs(size) * price, commission=fee, pnl=0.0,
                         pnlcomm=-fee, isopen=True, isclosed=False, dtopen=dt, dtclose=None))

    for fill in fills.to_dict("records"):
        data = fill.get("data")
        size = float(fill.get("size", 0.0))
        price = float(fill.get("price", 0.0))
        fee = float(fill.get("commission", 0.0))
        dt = fill.get("datetime")
        if abs(size) < 1e-9:
            continue
        pos = positions.get(data)
        if pos is None:
            open_position(data, size, price, fee, dt)
            continue
        old_size = pos["size"]
        if old_size * size > 0:
            pos["price"] = (abs(old_size) * pos["price"] + abs(size) * price) / abs(old_size + size)
            pos["size"] += size
            pos["commission"] += fee
            continue

        closing_size = min(abs(old_size), abs(size))
        closing_fee = fee * closing_size / abs(size)
        pos["pnl"] += closing_size * (price - pos["price"]) * (1 if old_size > 0 else -1)
        pos["commission"] += closing_fee
        remaining = old_size + size
        if abs(size) < abs(old_size) - 1e-9:
            pos["size"] = remaining
            continue

        rows.append(dict(datetime=dt, data=data, size=0.0, price=price, value=0.0,
                         commission=pos["commission"], pnl=pos["pnl"],
                         pnlcomm=pos["pnl"] - pos["commission"], isopen=False,
                         isclosed=True, dtopen=pos["dtopen"], dtclose=dt))
        del positions[data]
        # A reversal closes one lifecycle and opens another, sharing fill fees.
        if abs(remaining) > 1e-9:
            open_position(data, remaining, price, fee - closing_fee, dt)
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

    position_sizes: dict[Any, float] = {}
    avg_prices: dict[Any, float] = {}
    realized_pnls: dict[Any, float] = {}
    rows = []
    data_name = getattr(getattr(strategy, "data", None), "_name", None)
    sizes = fills["size"].to_numpy(dtype=float, copy=False)
    prices = fills["price"].to_numpy(dtype=float, copy=False)
    datetimes = fills["datetime"].to_numpy(copy=False) if "datetime" in fills else None
    data_values = fills["data"].to_numpy(copy=False) if "data" in fills else None
    fallback_datetime = index[-1] if len(index) else None

    for row_idx, (signed_size, price) in enumerate(zip(sizes, prices, strict=True)):
        fill_data = data_values[row_idx] if data_values is not None else None
        if is_missing_scalar(fill_data):
            fill_data = data_name
        previous_size = position_sizes.get(fill_data, 0.0)
        avg_price = avg_prices.get(fill_data, 0.0)
        realized_pnl = realized_pnls.get(fill_data, 0.0)
        new_size = previous_size + signed_size
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
        position_sizes[fill_data] = position_size
        avg_prices[fill_data] = avg_price
        realized_pnls[fill_data] = realized_pnl
        mark_price = price
        value = position_size * mark_price
        fill_datetime = datetimes[row_idx] if datetimes is not None else None
        if is_missing_scalar(fill_datetime):
            fill_datetime = fallback_datetime
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


def _resolve_close_series(
    strategy: Any,
    frame: pd.DataFrame | None,
    index: pd.Index,
) -> np.ndarray | None:
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


def _resolve_close_by_data(
    strategy: Any,
    frame: pd.DataFrame | None,
    index: pd.Index,
) -> dict[str, np.ndarray]:
    closes: dict[str, np.ndarray] = {}
    if isinstance(frame, pd.DataFrame) and isinstance(frame.columns, pd.MultiIndex):
        for column in frame.columns:
            parts = tuple(str(part) for part in column)
            close_levels = [i for i, part in enumerate(parts) if part.lower() == "close"]
            if not close_levels:
                continue
            symbol_parts = [part for i, part in enumerate(parts) if i not in close_levels]
            if not symbol_parts or len(frame) != len(index):
                continue
            closes[str(symbol_parts[0])] = frame[column].to_numpy(dtype=float, copy=False)

    datas = list(getattr(strategy, "datas", []) or [])
    cerebro = getattr(strategy, "cerebro", None)
    if not datas and cerebro is not None:
        datas = list(getattr(cerebro, "datas", []) or [])
    for data_index, data in enumerate(datas):
        name = str(getattr(data, "_name", None) or f"data{data_index}")
        data_frame = getattr(data, "_frame", None)
        if (
            isinstance(data_frame, pd.DataFrame)
            and "close" in data_frame.columns
            and isinstance(data_frame.index, pd.DatetimeIndex)
        ):
            aligned = data_frame["close"].reindex(index).ffill()
            if aligned.notna().any():
                closes[name] = aligned.to_numpy(dtype=float, copy=False)
                continue
        close_arr = getattr(data, "_close", None)
        if close_arr is None:
            continue
        try:
            arr = np.asarray(close_arr, dtype=float)
        except (TypeError, ValueError):
            continue
        if len(arr) == len(index):
            closes[name] = arr
    return closes


def _build_equity_from_fills(
    strategy: Any,
    index: pd.Index,
    fills: pd.DataFrame,
    frame: pd.DataFrame | None = None,
) -> pd.Series | None:
    if fills.empty or len(index) == 0:
        return None
    closes = _resolve_close_series(strategy, frame, index)
    close_by_data = _resolve_close_by_data(strategy, frame, index)
    has_data_labels = "data" in fills.columns and fills["data"].notna().any()
    if closes is None and not close_by_data:
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
    position_sizes: dict[str, float] = {}
    values: list[float] = []
    for bar_num, ts in enumerate(bar_index):
        for fill in fills_by_time.get(pd.Timestamp(ts), ()):
            signed_size = float(fill.get("size", 0.0))
            price = float(fill.get("price", 0.0))
            commission = float(fill.get("commission", 0.0))
            cash -= signed_size * price + commission
            fill_data = fill.get("data")
            if has_data_labels and fill_data is not None and not pd.isna(fill_data):
                key = str(fill_data)
                position_sizes[key] = position_sizes.get(key, 0.0) + signed_size
            else:
                position_size += signed_size
        if has_data_labels and close_by_data:
            marked_positions = 0.0
            for data_key, size in position_sizes.items():
                close_arr = close_by_data.get(str(data_key))
                if close_arr is None:
                    return None
                marked_positions += size * float(close_arr[bar_num])
            values.append(cash + marked_positions)
        else:
            if closes is None:
                return None
            values.append(cash + position_size * float(closes[bar_num]))
    return pd.Series(values, index=index, name="equity", dtype=float)


def _build_equity_returns(
    strategy: Any,
    index: pd.Index,
    fills: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    runtime=getattr(strategy,'_runtime',None)
    samples=getattr(runtime,'_delivery_equity_samples',None)
    if samples is not None:
        if len(samples)!=len(index):raise ValueError('replay account observations do not cover delivery index')
        equity=pd.Series(samples,index=index,name='equity',dtype=float)
        returns=equity.pct_change().fillna(0.0);returns.name='returns'
        return equity,returns
    if fills is not None:
        engine_mod = sys.modules.get('tradelearn.backtest.engine')
        builder = getattr(engine_mod, '_build_equity_from_fills', _build_equity_from_fills)
        fill_equity = builder(strategy, index, fills, frame)
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


def _stats_config(
    cerebro: Any,
    periods: int | None = None,
    period_source: str | None = None,
) -> dict[str, Any]:
    config = BacktestRuntimeConfig.from_owner(cerebro)
    
    # Extract strategy name from cerebro's strategy list if available.
    strategy_name = "TradeLearn Report"
    if hasattr(cerebro, "strats") and cerebro.strats:
        strategy_cls = cerebro.strats[0][0]
        if hasattr(strategy_cls, "__name__"):
            strategy_name = strategy_cls.__name__

    result = {
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
    configured_slippage = getattr(getattr(cerebro, "broker", None), "_configured_slippage", None)
    slippage = configured_slippage() if callable(configured_slippage) else None
    if slippage is not None:
        kind, value = slippage
        result["broker"]["slippage"] = {"kind": kind, "value": value}
    if periods is not None:
        result["annualization"] = {"periods": periods, "period_source": period_source}
    return result


def _return_pct(final_value: float, broker: Any) -> float:
    start_cash = float(getattr(broker, "_cash", 0.0) or 0.0)
    if start_cash == 0.0:
        return float("nan")
    return (float(final_value) / start_cash - 1.0) * 100.0


def _trade_summary(broker: Any, trades: pd.DataFrame | None = None) -> tuple[float, float]:
    if trades is not None and not trades.empty and "isclosed" in trades.columns:
        closed = trades[trades["isclosed"].astype(bool)]
        total = float(len(closed))
        pnl = closed["pnlcomm"] if "pnlcomm" in closed.columns else closed["pnl"]
        wins = float((pnl >= 0).sum()) if total > 0 else 0.0
        return total, wins
    trade_summary = getattr(broker, "trade_summary", None)
    if callable(trade_summary):
        total, wins = trade_summary()
        return float(total), float(wins)
    return 0.0, 0.0


def _win_rate_pct(total_trades: float, wins: float) -> float:
    return wins / total_trades * 100.0 if total_trades else float("nan")


def _summary_pnl(positions: pd.DataFrame) -> tuple[float, float]:
    """Read realized P&L and margin from the last fill-time position row."""
    if positions is None or positions.empty:
        return 0.0, 0.0
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
        _f(last.get("margin_used", 0.0)),
    )


def _final_unrealized_pnl(strategy: Any) -> float:
    """Mark open broker positions using each feed's last observed finite close."""
    pnl = 0.0
    for data in strategy.datas:
        position = strategy.broker.getposition(data)
        if not position.size:
            continue
        for age in range(len(data.close)):
            close = float(data.close[-age])
            if math.isfinite(close):
                pnl += (close - position.price) * position.size * strategy.broker.get_mult(data)
                break
        else:
            return math.nan
    return pnl


def _canonical_drawdown(returns: pd.Series) -> float:
    from tradelearn.metrics.risk import max_drawdown

    return float(max_drawdown(returns))


def _summary_sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Canonical shared metric, preserving missing values and validation errors."""
    from tradelearn.metrics.risk import sharpe

    values = pd.Series(dtype=float) if returns is None else returns
    return float(sharpe(values, periods=periods))


def _summary_annual_return(returns: pd.Series, periods: int = 252) -> float:
    """Canonical shared metric, preserving missing values and validation errors."""
    from tradelearn.metrics.returns import annual_return

    values = pd.Series(dtype=float) if returns is None else returns
    return float(annual_return(values, periods=periods))


def _summary_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Canonical shared metric, preserving missing values and validation errors."""
    from tradelearn.metrics.risk import volatility

    values = pd.Series(dtype=float) if returns is None else returns
    return float(volatility(values, periods=periods))


def _summary_sortino(returns: pd.Series, periods: int = 252) -> float:
    """Canonical shared metric, preserving missing values and validation errors."""
    from tradelearn.metrics.risk import sortino

    values = pd.Series(dtype=float) if returns is None else returns
    return float(sortino(values, periods=periods))


def _summary_calmar(returns: pd.Series, periods: int = 252) -> float:
    """Canonical shared metric, preserving missing values and validation errors."""
    from tradelearn.metrics.risk import calmar

    values = pd.Series(dtype=float) if returns is None else returns
    return float(calmar(values, periods=periods))


def _summary_trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Compute closed-cycle metrics on net PnL through the common formulas."""
    from tradelearn.metrics import trade

    closed = pd.DataFrame() if trades is None else trades
    if "isclosed" in closed:
        closed = closed[closed.isclosed.astype(bool)]
    pnl = closed.get("pnlcomm", closed.get("pnl", pd.Series(dtype=float)))
    pnl = pd.Series(pnl, dtype=float).dropna()
    return {
        "profit_factor": trade.profit_factor(pnl),
        "expectancy": trade.expectancy(pnl),
        "avg_win": trade.avg_win(pnl),
        "avg_loss": trade.avg_loss(pnl),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "max_consec_wins": trade.max_consecutive_wins(pnl),
        "max_consec_losses": trade.max_consecutive_losses(pnl),
        "total_commission": float(closed.commission.sum()) if "commission" in closed else 0.0,
    }


def _summary_exposure_pct(positions: pd.DataFrame) -> float:
    if positions is None or positions.empty:
        return 0.0
    frame = pd.DataFrame(positions).copy()
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
    symbol_col = (
        lowered.get("symbol")
        or lowered.get("ticker")
        or lowered.get("asset")
        or lowered.get("data")
    )
    size_col = lowered.get("size") or lowered.get("qty") or lowered.get("quantity")
    value_col = lowered.get("value") or lowered.get("market_value") or lowered.get("position_value")
    exposure_col = size_col or value_col
    if date_col is None or exposure_col is None:
        return 0.0

    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame = frame.loc[dates.notna()].copy()
    if frame.empty:
        return 0.0
    frame["__sort_datetime"] = dates.loc[frame.index]
    frame["__date"] = frame["__sort_datetime"].dt.floor("D")
    frame["__exposure"] = pd.to_numeric(frame[exposure_col], errors="coerce").fillna(0.0)

    if symbol_col is None:
        daily = frame.sort_values("__sort_datetime").groupby("__date")["__exposure"].last()
        exposed_days = daily.abs() > 1e-9
        return float(exposed_days.sum()) / len(daily) * 100.0 if len(daily) else 0.0

    daily_wide = (
        frame.sort_values("__sort_datetime")
        .pivot_table(
            index="__date",
            columns=symbol_col,
            values="__exposure",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
        .fillna(0.0)
    )
    if daily_wide.empty:
        return 0.0
    exposed_days = daily_wide.abs().sum(axis=1) > 1e-9
    return float(exposed_days.sum()) / len(daily_wide) * 100.0


def _summary_drawdown_stats(equity: pd.Series) -> dict[str, Any]:
    if equity is None or equity.empty:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_dd_duration": pd.Timedelta(0),
            "avg_dd_duration": pd.Timedelta(0),
        }
    cummax = equity.cummax()
    dd = (cummax - equity) / cummax
    dd_mask = dd > 0
    # Calculate durations
    dd_groups = (dd_mask != dd_mask.shift()).cumsum()
    durations = []
    for _name, group in dd[dd_mask].groupby(dd_groups.to_numpy()[dd_mask.to_numpy()]):
        if len(group) > 0:
            durations.append(group.index[-1] - group.index[0])

    return {
        "max_drawdown": float(dd.max()),
        "avg_drawdown": float(dd[dd_mask].mean()) if dd_mask.any() else 0.0,
        "max_dd_duration": max(durations) if durations else pd.Timedelta(0),
        "avg_dd_duration": (
            sum(durations, pd.Timedelta(0)) / len(durations)
            if durations
            else pd.Timedelta(0)
        ),
    }


def _summary_trade_stats(trades: pd.DataFrame, start_cash: float) -> dict[str, Any]:
    empty = {
        "best_trade_pct": 0.0,
        "worst_trade_pct": 0.0,
        "avg_trade_pct": 0.0,
        "max_trade_duration": pd.Timedelta(0),
        "avg_trade_duration": pd.Timedelta(0),
        "sqn": 0.0,
        "kelly_criterion": 0.0,
    }
    if trades is None or trades.empty:
        return empty
    closed = (
        trades[trades.get("isclosed", pd.Series(dtype=bool)).astype(bool)]
        if "isclosed" in trades.columns
        else trades
    )
    if closed.empty:
        return empty

    pnl_pct = (closed["pnlcomm"] / start_cash * 100.0) if start_cash > 0 else pd.Series(dtype=float)
    durations = closed["dtclose"] - closed["dtopen"]

    wins = closed[closed["pnlcomm"] > 0]
    win_rate = len(wins) / len(closed)
    avg_win = wins["pnlcomm"].mean() if not wins.empty else 0.0
    losses = closed[closed["pnlcomm"] < 0]
    avg_loss = losses["pnlcomm"].mean() if not losses.empty else 1e-9
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0

    # Kelly: W - (1-W)/R
    kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0

    # SQN: sqrt(N) * mean / std
    mean_pnl = closed["pnlcomm"].mean()
    std_pnl = closed["pnlcomm"].std(ddof=0)
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
    runtime = getattr(strategy, "_runtime", None)
    if runtime is not None and runtime._clock.requires_union:
        index = pd.to_datetime(runtime._clock.timestamps, unit=runtime._clock.unit, utc=True)
        index = index.tz_convert(frame.index.tz) if frame.index.tz is not None else index.tz_localize(None)

    if getattr(runtime,'_delivery_equity_samples',None) is not None:
        index=index[:len(runtime._delivery_equity_samples)]

    unrealized_pnl = _final_unrealized_pnl(strategy)
    if lazy_artifacts:
        final_cash = float(strategy.broker.getcash())
        final_value = float(strategy.broker.getvalue())

        # Build summary inputs eagerly (cheap: bounded by # of fills),
        # but expose DataFrames lazily through factories.
        eager_fills = _fills_frame(strategy.broker)
        eager_trades = _trades_frame(eager_fills)
        eager_positions = _positions_frame(strategy, eager_fills, index)
        eager_equity, eager_returns = _build_equity_returns(strategy, index, eager_fills, frame)
        periods, period_source = resolve_periods(
            eager_returns, BacktestRuntimeConfig.from_owner(cerebro).periods
        )
        total_trades, winning_trades = _trade_summary(strategy.broker, eager_trades)

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
            "annual_return": _summary_annual_return(eager_returns, periods),
            "volatility": _summary_volatility(eager_returns, periods),
            "sharpe": _summary_sharpe(eager_returns, periods),
            "sortino": _summary_sortino(eager_returns, periods),
            "calmar": _summary_calmar(eager_returns, periods),
            "max_drawdown": _canonical_drawdown(eager_returns),
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
            "final_unrealized_pnl": unrealized_pnl,
        })
        
        stats = Stats(
            returns=returns_factory,
            equity=equity_factory,
            trades=trades_factory,
            positions=positions_factory,
            orders=lambda: _orders_frame(strategy.broker),
            summary=summary,
            analyzers={},
            config=_stats_config(cerebro, periods, period_source),
            periods=periods,
            period_source=period_source,
            fills=fills_factory,
        )
        return stats

    fills = _fills_frame(strategy.broker)
    equity, returns = _build_equity_returns(strategy, index, fills, frame)
    periods, period_source = resolve_periods(
        returns, BacktestRuntimeConfig.from_owner(cerebro).periods
    )
    trades = _trades_frame(fills)
    orders = _orders_frame(strategy.broker)
    positions = _positions_frame(strategy, fills, index)
    final_cash = float(strategy.broker.getcash())
    final_value = float(strategy.broker.getvalue())
    total_trades, winning_trades = _trade_summary(strategy.broker, trades)
    realized_pnl, margin_used = _summary_pnl(positions)
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
        "annual_return": _summary_annual_return(returns, periods),
        "volatility": _summary_volatility(returns, periods),
        "sharpe": _summary_sharpe(returns, periods),
        "sortino": _summary_sortino(returns, periods),
        "calmar": _summary_calmar(returns, periods),
        "max_drawdown": _canonical_drawdown(returns),
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
        config=_stats_config(cerebro, periods, period_source),
        periods=periods,
        period_source=period_source,
        fills=fills,
    )
    return stats
