"""Excel report export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn import metrics
from tradelearn.report.analytics import monthly_returns_matrix

REPORT_SHEETS = [
    "summary",
    "trades",
    "daily_returns",
    "monthly_returns",
    "drawdowns",
    "positions",
    "orders",
    "config",
]


def write_excel_report(reporter: Any, path: str | Path) -> Path:
    """Write a multi-sheet Excel report and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = _series(reporter._get("returns"), name="returns")
    trades = _frame(reporter._get("trades", default=pd.DataFrame()))
    positions = _frame(reporter._get("positions", default=pd.DataFrame()))
    orders = _frame(reporter._get("orders", default=pd.DataFrame()))
    config = reporter._get("config", default={}) or {}

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        _summary_frame(reporter.summary()).to_excel(writer, sheet_name="summary", index=False)
        _excel_safe_frame(trades).to_excel(writer, sheet_name="trades", index=False)
        _excel_safe_frame(returns.to_frame()).to_excel(writer, sheet_name="daily_returns")
        monthly_returns_matrix(returns).to_excel(writer, sheet_name="monthly_returns")
        _excel_safe_frame(_drawdowns(returns)).to_excel(
            writer,
            sheet_name="drawdowns",
            index=False,
        )
        _excel_safe_frame(positions).to_excel(writer, sheet_name="positions", index=False)
        _excel_safe_frame(orders).to_excel(writer, sheet_name="orders", index=False)
        _config_frame(config).to_excel(writer, sheet_name="config", index=False)
        _format_summary(writer)
        _format_monthly_heatmap(writer)
    return output


def _summary_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Return summary as metric/value rows."""
    return pd.DataFrame({"metric": list(summary), "value": list(summary.values())})


def _drawdowns(returns: pd.Series) -> pd.DataFrame:
    """Return drawdown series for the drawdowns sheet."""
    drawdown = metrics.drawdown_series(returns)
    return pd.DataFrame({"date": drawdown.index, "drawdown": drawdown.to_numpy()})


def _config_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return config as key/value rows."""
    return pd.DataFrame({"key": list(config), "value": [str(value) for value in config.values()]})


def _series(value: Any, name: str) -> pd.Series:
    """Normalize a value to a named series."""
    series = pd.Series(value).copy()
    series.name = name
    return series


def _frame(value: Any) -> pd.DataFrame:
    """Normalize a value to a frame."""
    return pd.DataFrame(value).copy()


def _excel_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with timezone-aware datetimes converted for Excel."""
    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    for column in result.columns:
        values = result[column]
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            result[column] = values.dt.tz_convert("UTC").dt.tz_localize(None)
    return result


def _format_summary(writer: pd.ExcelWriter) -> None:
    """Apply basic summary number formatting."""
    workbook = writer.book
    worksheet = writer.sheets["summary"]
    numeric = workbook.add_format({"num_format": "0.000000"})
    worksheet.set_column("A:A", 24)
    worksheet.set_column("B:B", 16, numeric)


def _format_monthly_heatmap(writer: pd.ExcelWriter) -> None:
    """Apply a native xlsxwriter heatmap to monthly returns."""
    worksheet = writer.sheets["monthly_returns"]
    worksheet.conditional_format(
        "B2:N200",
        {
            "type": "3_color_scale",
            "min_color": "#F8696B",
            "mid_color": "#FFFFFF",
            "max_color": "#63BE7B",
        },
    )
