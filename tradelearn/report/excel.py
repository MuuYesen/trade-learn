"""Excel report export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.api.types as pd_types

from tradelearn.report.analytics import monthly_returns_matrix
from tradelearn.report.sections import build_context, collect_excel_sheets

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


def write_excel_report(
    reporter: Any,
    path: str | Path,
    benchmark: pd.Series | None = None,
    sections: list[Any] | tuple[Any, ...] | None = None,
) -> Path:
    """Write a multi-sheet Excel report and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = _series(reporter._get("returns"), name="returns")
    benchmark_returns = None if benchmark is None else _series(benchmark, name="benchmark")
    rolling_beta = (
        pd.Series(dtype="float64", name="rolling_beta")
        if benchmark_returns is None
        else reporter.rolling_beta(benchmark_returns)
    )
    trades = _frame(reporter._get("trades", default=pd.DataFrame()))
    positions = _frame(reporter._get("positions", default=pd.DataFrame()))
    orders = _frame(reporter._get("orders", default=pd.DataFrame()))
    factor_ic = reporter.factor_ic()
    factor_rank_ic = reporter.factor_rank_ic()
    factor_turnover = reporter.factor_turnover()
    factor_autocorrelation = reporter.factor_autocorrelation()
    factor_long_short_returns = reporter.factor_long_short_returns()
    factor_quantile_returns = reporter.factor_quantile_returns()
    config = reporter._get("config", default={}) or {}
    custom_sheets = collect_excel_sheets(
        sections,
        build_context(reporter, benchmark=benchmark_returns),
    )

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        _summary_frame(reporter.summary(benchmark=benchmark_returns)).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        _excel_safe_frame(trades).to_excel(writer, sheet_name="trades", index=False)
        _excel_safe_frame(returns.to_frame()).to_excel(writer, sheet_name="daily_returns")
        if benchmark_returns is not None:
            _excel_safe_frame(benchmark_returns.to_frame()).to_excel(
                writer,
                sheet_name="benchmark_returns",
            )
        if not rolling_beta.empty:
            _excel_safe_frame(rolling_beta.to_frame()).to_excel(
                writer,
                sheet_name="rolling_beta",
            )
        monthly_returns_matrix(returns).to_excel(writer, sheet_name="monthly_returns")
        _excel_safe_frame(_drawdowns(reporter)).to_excel(
            writer,
            sheet_name="drawdowns",
            index=False,
        )
        _excel_safe_frame(positions).to_excel(writer, sheet_name="positions", index=False)
        _excel_safe_frame(orders).to_excel(writer, sheet_name="orders", index=False)
        if not factor_ic.empty:
            _excel_safe_frame(factor_ic.to_frame("ic")).to_excel(
                writer,
                sheet_name="factor_ic",
            )
        if not factor_rank_ic.empty:
            _excel_safe_frame(factor_rank_ic.to_frame("rank_ic")).to_excel(
                writer,
                sheet_name="factor_rank_ic",
            )
        if not factor_turnover.empty:
            _excel_safe_frame(factor_turnover.to_frame("turnover")).to_excel(
                writer,
                sheet_name="factor_turnover",
            )
        if not factor_autocorrelation.empty:
            _excel_safe_frame(factor_autocorrelation.to_frame("autocorrelation")).to_excel(
                writer,
                sheet_name="factor_autocorr",
            )
        if not factor_long_short_returns.empty:
            _excel_safe_frame(factor_long_short_returns).to_excel(
                writer,
                sheet_name="factor_long_short",
            )
        if not factor_quantile_returns.empty:
            _excel_safe_frame(factor_quantile_returns).to_excel(
                writer,
                sheet_name="factor_quantiles",
            )
        for sheet_name, frame in custom_sheets.items():
            _excel_safe_frame(frame).to_excel(writer, sheet_name=sheet_name, index=False)
        _config_frame(config).to_excel(writer, sheet_name="config", index=False)
        _format_numeric_sheets(writer)
        _format_monthly_heatmap(writer)
    return output


def _summary_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Return summary as metric/value rows."""
    return pd.DataFrame({"metric": list(summary), "value": list(summary.values())})


def _drawdowns(reporter: Any) -> pd.DataFrame:
    """Return top drawdown episodes for the drawdowns sheet."""
    return reporter.top_drawdowns(limit=10)


def _config_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return config as key/value rows."""
    rows = list(_flatten_config(config))
    return pd.DataFrame(
        {
            "key": [key for key, _ in rows],
            "value": [_excel_config_value(value) for _, value in rows],
        }
    )


def _flatten_config(config: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Return dotted config key/value rows for nested mappings."""
    rows: list[tuple[str, Any]] = []
    for key, value in config.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_flatten_config(value, prefix=dotted_key))
        else:
            rows.append((dotted_key, value))
    return rows


def _excel_config_value(value: Any) -> Any:
    """Return a scalar config value that keeps Excel-native numeric types."""
    if pd_types.is_scalar(value):
        return value
    return str(value)


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


def _format_numeric_sheets(writer: pd.ExcelWriter) -> None:
    """Apply the report's six-decimal number format to numeric report sheets."""
    workbook = writer.book
    numeric = workbook.add_format({"num_format": "0.000000"})
    writer.sheets["summary"].set_column("A:A", 24)
    writer.sheets["summary"].set_column("B:B", 16, numeric)
    for sheet_name in [
        "trades",
        "daily_returns",
        "benchmark_returns",
        "rolling_beta",
        "monthly_returns",
        "drawdowns",
        "positions",
        "orders",
        "factor_ic",
        "factor_rank_ic",
        "factor_turnover",
        "factor_autocorr",
        "factor_long_short",
        "factor_quantiles",
    ]:
        if sheet_name in writer.sheets:
            writer.sheets[sheet_name].set_column("A:Z", None, numeric)


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
