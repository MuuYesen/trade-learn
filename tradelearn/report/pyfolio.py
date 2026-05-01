"""Pyfolio-compatible report entrypoints.

This module keeps the pyfolio user workflow while delegating rendering to
Tradelearn's unified :class:`tradelearn.report.Reporter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.report.reporter import Reporter


def create_full_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    benchmark_rets: pd.Series | None = None,
    output: str | Path = "pyfolio_report.html",
    format: str | None = None,
    **_: Any,
) -> Path:
    """Write a pyfolio-style full tear sheet report.

    Parameters follow pyfolio naming where possible. Unsupported pyfolio
    options are accepted via ``**kwargs`` and intentionally ignored so old
    notebooks can migrate without breaking on presentation-only flags.
    """
    return _report(
        returns=returns,
        positions=positions,
        transactions=transactions,
        market_data=market_data,
        benchmark_rets=benchmark_rets,
        output=output,
        format=format,
        strategy_name="pyfolio-tear-sheet",
    )


def create_returns_tear_sheet(
    returns: pd.Series,
    benchmark_rets: pd.Series | None = None,
    output: str | Path = "pyfolio_returns.html",
    format: str | None = None,
    **_: Any,
) -> Path:
    """Write a pyfolio-style returns tear sheet report."""
    return _report(
        returns=returns,
        benchmark_rets=benchmark_rets,
        output=output,
        format=format,
        strategy_name="pyfolio-returns",
    )


def create_simple_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    benchmark_rets: pd.Series | None = None,
    output: str | Path = "pyfolio_simple.html",
    format: str | None = None,
    **kwargs: Any,
) -> Path:
    """Write a compact pyfolio-style tear sheet report."""
    return create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark_rets,
        output=output,
        format=format,
        **kwargs,
    )


def report(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    benchmark_rets: pd.Series | None = None,
    output: str | Path = "pyfolio_report.html",
    format: str | None = None,
    **kwargs: Any,
) -> Path:
    """Write a pyfolio-compatible report using the compact Tradelearn alias."""
    return create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark_rets,
        output=output,
        format=format,
        **kwargs,
    )


def _report(
    *,
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    benchmark_rets: pd.Series | None = None,
    output: str | Path,
    format: str | None,
    strategy_name: str,
) -> Path:
    stats = {
        "returns": pd.Series(returns).copy(),
        "trades": _trades_from_transactions(transactions),
        "fills": _fills_from_transactions(transactions),
        "positions": _positions_frame(positions),
        "summary": {"strategy_name": strategy_name},
        "config": {"strategy": strategy_name, "source": "pyfolio"},
    }
    return Reporter(stats, market_data=market_data).report(
        output,
        benchmark=None if benchmark_rets is None else pd.Series(benchmark_rets).copy(),
        format=format,
    )


def _positions_frame(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None:
        return pd.DataFrame()
    frame = pd.DataFrame(positions).copy()
    if {"date", "symbol", "value"}.issubset(frame.columns):
        return frame
    if frame.empty:
        return pd.DataFrame()
    original_index_name = frame.index.name
    long = frame.reset_index()
    index_column = original_index_name if original_index_name in long.columns else "index"
    long = long.rename(columns={index_column: "date"})
    value_columns = [column for column in long.columns if column != "date"]
    return long.melt(
        id_vars=["date"],
        value_vars=value_columns,
        var_name="symbol",
        value_name="value",
    )


def _fills_from_transactions(transactions: pd.DataFrame | None) -> pd.DataFrame:
    if transactions is None:
        return pd.DataFrame()
    frame = pd.DataFrame(transactions).copy()
    if frame.empty:
        return pd.DataFrame()
    if "datetime" not in frame:
        frame["datetime"] = frame.index
    if "side" not in frame and "amount" in frame:
        frame["side"] = ["buy" if amount > 0 else "sell" for amount in frame["amount"]]
    if "size" not in frame and "amount" in frame:
        frame["size"] = frame["amount"].abs()
    columns = [
        column
        for column in ["datetime", "symbol", "side", "size", "price", "commission"]
        if column in frame
    ]
    return frame[columns].reset_index(drop=True)


def _trades_from_transactions(transactions: pd.DataFrame | None) -> pd.DataFrame:
    if transactions is None:
        return pd.DataFrame({"pnl": pd.Series(dtype="float64")})
    frame = pd.DataFrame(transactions).copy()
    if frame.empty or "pnl" not in frame:
        return pd.DataFrame({"pnl": pd.Series(dtype="float64")})
    result = frame[["pnl"]].copy()
    if "datetime" in frame:
        result["datetime"] = frame["datetime"]
    else:
        result["datetime"] = frame.index
    if "symbol" in frame:
        result["symbol"] = frame["symbol"]
    return result.reset_index(drop=True)


__all__ = [
    "create_full_tear_sheet",
    "create_returns_tear_sheet",
    "create_simple_tear_sheet",
    "report",
]
