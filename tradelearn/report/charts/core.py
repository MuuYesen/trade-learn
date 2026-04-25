"""Core Bokeh chart builders for reports."""

from __future__ import annotations

import pandas as pd
from bokeh.plotting import figure
from bokeh.transform import linear_cmap


def equity_curve(equity: pd.Series, benchmark: pd.Series | None = None):
    """Return an equity curve figure."""
    frame = _plot_frame(equity, "equity")
    plot = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=260,
        sizing_mode="stretch_width",
    )
    plot.line(frame["date"], frame["equity"], line_width=2, color="#1f77b4")
    if benchmark is not None:
        benchmark_equity = (1.0 + benchmark).cumprod()
        benchmark_frame = _plot_frame(benchmark_equity, "benchmark")
        plot.line(
            benchmark_frame["date"],
            benchmark_frame["benchmark"],
            line_width=2,
            color="#ff7f0e",
            legend_label="Benchmark",
        )
    return plot


def drawdown(drawdown_series: pd.Series):
    """Return a drawdown figure."""
    frame = _plot_frame(drawdown_series, "drawdown")
    plot = figure(
        title="Drawdown",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    plot.varea(frame["date"], y1=0, y2=frame["drawdown"], color="#d62728", alpha=0.35)
    plot.line(frame["date"], frame["drawdown"], line_width=2, color="#d62728")
    return plot


def monthly_heatmap(monthly: pd.DataFrame):
    """Return a monthly returns heatmap figure."""
    values = monthly.drop(index="month_avg", errors="ignore")
    months = [column for column in range(1, 13) if column in values.columns]
    years = [str(year) for year in values.index]
    data = {"month": [], "year": [], "return": []}
    for year in values.index:
        for month in months:
            data["month"].append(str(month))
            data["year"].append(str(year))
            data["return"].append(values.loc[year, month])
    plot = figure(
        title="Monthly Returns Heatmap",
        x_range=[str(month) for month in months],
        y_range=years,
        height=240,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "return",
        palette=["#d62728", "#f7f7f7", "#2ca02c"],
        low=-0.05,
        high=0.05,
    )
    plot.rect(
        "month",
        "year",
        width=0.95,
        height=0.95,
        source=data,
        fill_color=mapper,
        line_color="white",
    )
    return plot


def rolling_sharpe(rolling: pd.Series):
    """Return a rolling Sharpe figure."""
    frame = _plot_frame(rolling, "rolling_sharpe").dropna()
    plot = figure(
        title="Rolling Sharpe",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_sharpe"], line_width=2, color="#9467bd")
    return plot


def trade_distribution(distribution: pd.DataFrame):
    """Return a trade PnL histogram figure."""
    plot = figure(title="Trade Distribution", height=220, sizing_mode="stretch_width")
    if distribution.empty:
        return plot
    plot.quad(
        top=distribution["count"],
        bottom=0,
        left=distribution["left"],
        right=distribution["right"],
        fill_color="#2ca02c",
        line_color="white",
        alpha=0.65,
    )
    mean = distribution.attrs.get("mean")
    median = distribution.attrs.get("median")
    if pd.notna(mean):
        plot.line([mean, mean], [0, distribution["count"].max()], color="#1f77b4", line_width=2)
    if pd.notna(median):
        plot.line([median, median], [0, distribution["count"].max()], color="#ff7f0e", line_width=2)
    return plot


def exposure(exposure_frame: pd.DataFrame):
    """Return a multi-asset exposure figure."""
    frame = exposure_frame.reset_index().rename(
        columns={exposure_frame.index.name or "index": "date"}
    )
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Exposure Chart",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, symbol in enumerate(exposure_frame.columns):
        plot.line(
            frame["date"],
            frame[symbol],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=str(symbol),
        )
    return plot


def correlation_matrix(correlation: pd.DataFrame):
    """Return a multi-asset correlation heatmap figure."""
    symbols = [str(symbol) for symbol in correlation.columns]
    data = {"x": [], "y": [], "correlation": []}
    for row in correlation.index:
        for symbol in correlation.columns:
            data["x"].append(str(symbol))
            data["y"].append(str(row))
            data["correlation"].append(correlation.loc[row, symbol])
    plot = figure(
        title="Correlation Matrix",
        x_range=symbols,
        y_range=symbols,
        height=260,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "correlation",
        palette=["#d62728", "#f7f7f7", "#1f77b4"],
        low=-1.0,
        high=1.0,
    )
    plot.rect(
        "x",
        "y",
        width=0.95,
        height=0.95,
        source=data,
        fill_color=mapper,
        line_color="white",
    )
    return plot


def quantile_returns(returns: pd.DataFrame):
    """Return a factor quantile returns figure."""
    frame = returns.reset_index().rename(columns={returns.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Factor Quantile Returns",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, column in enumerate(returns.columns):
        plot.line(
            frame["date"],
            frame[column],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=f"Q{column}",
        )
    return plot


def factor_ic(ic: pd.Series):
    """Return a factor IC figure."""
    frame = _plot_frame(ic, "ic").dropna()
    plot = figure(
        title="Factor IC",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["ic"], line_width=2, color="#1f77b4")
        plot.line(
            frame["date"],
            frame["ic"].expanding(min_periods=1).mean(),
            line_width=2,
            color="#ff7f0e",
            legend_label="Expanding Mean",
        )
    return plot


def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame
