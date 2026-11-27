"""Portfolio equity, return, risk, holdings, and attribution Bokeh charts."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from tradelearn.report.charts._common import (
    DEFAULT_REPORT_TIMEZONE,
    MARKET_BLUE,
    MARKET_DOWN,
    MARKET_UP,
    _add_drawdown_markers,
    _add_passive_hover,
    _align_series_to_index,
    _coerce_timezone,
    _make_static_chart,
    _normalize_dates,
    _plot_frame,
    _positions_wide,
    _style_equity_curve_legend,
    _timezone_label,
    _transactions_frame,
)


def equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    drawdowns: pd.DataFrame | None = None,
):
    """Return an equity curve figure."""
    frame = _plot_frame(equity, "equity")
    plot = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=300,
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
            legend_label=str(pd.Series(benchmark).name or "Benchmark"),
        )
    if drawdowns is not None and not drawdowns.empty:
        _add_drawdown_markers(plot, equity, drawdowns)
    _style_equity_curve_legend(plot)
    _make_static_chart(plot)
    return plot

def drawdown(drawdown_series: pd.Series):
    """Return a drawdown figure."""
    frame = _plot_frame(drawdown_series, "drawdown")
    plot = figure(
        title="Drawdown",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    plot.varea(frame["date"], y1=0, y2=frame["drawdown"], color="#d62728", alpha=0.35)
    plot.line(frame["date"], frame["drawdown"], line_width=2, color="#d62728")
    _make_static_chart(plot)
    return plot

def annual_returns(returns: pd.Series):
    """Return an annual returns bar chart."""
    annual = (1.0 + pd.Series(returns).dropna()).resample("YE").prod() - 1.0
    frame = annual.to_frame("return").reset_index()
    frame.columns = ["date", "return"]
    frame["year"] = frame["date"].dt.year.astype(str)
    frame["color"] = [MARKET_UP if value >= 0 else MARKET_DOWN for value in frame["return"]]
    plot = figure(
        title="Annual Returns",
        x_range=list(frame["year"]),
        height=380,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        plot.vbar("year", 0.7, "return", source=ColumnDataSource(frame), color="color", alpha=0.75)
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def monthly_heatmap(monthly: pd.DataFrame):
    """Return a monthly returns heatmap figure."""
    values = monthly.drop(index="month_avg", errors="ignore")
    months = [column for column in range(1, 13) if column in values.columns]
    years = [str(year) for year in values.index]
    bound = 0.10
    data = {"month": [], "year": [], "return": [], "label": [], "label_color": []}
    for year in values.index:
        for month in months:
            value = pd.to_numeric(pd.Series([values.loc[year, month]]), errors="coerce").iloc[0]
            data["month"].append(str(month))
            data["year"].append(str(year))
            data["return"].append(value)
            data["label"].append("--" if pd.isna(value) else f"{float(value):+.1%}")
            data["label_color"].append("#243247")
    plot = figure(
        title="Monthly Returns Heatmap",
        x_range=[str(month) for month in months],
        y_range=years,
        height=380,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "return",
        palette=["#d95a5a", "#e88989", "#f1baba", "#e9eef3", "#bee7c8", "#7acb86", "#3fa956"],
        low=-bound,
        high=bound,
        nan_color="#f1f4f7",
    )
    source = ColumnDataSource(data)
    plot.rect(
        "month",
        "year",
        width=0.95,
        height=0.95,
        source=source,
        fill_color=mapper,
        line_color="white",
    )
    label_renderer = plot.text(
        "month",
        "year",
        text="label",
        source=source,
        text_align="center",
        text_baseline="middle",
        text_color={"field": "label_color"},
        text_font_size="8pt",
        text_font_style="normal",
        name="monthly_return_labels",
    )
    label_renderer.level = "annotation"
    _make_static_chart(plot)
    _add_passive_hover(
        plot,
        HoverTool(
            tooltips=[
                ("Year", "@year"),
                ("Month", "@month"),
                ("Return", "@return{+0.00%}"),
            ],
        ),
    )
    return plot

def monthly_returns_distribution(returns: pd.Series):
    """Return a monthly returns distribution chart."""
    monthly = ((1.0 + pd.Series(returns).dropna()).resample("ME").prod() - 1.0).dropna()
    plot = figure(
        title="Monthly Returns Distribution",
        height=300,
        sizing_mode="stretch_width",
    )
    if not monthly.empty:
        buckets = min(20, max(1, len(monthly)))
        counts = pd.cut(monthly, bins=buckets).value_counts(sort=False)
        frame = pd.DataFrame(
            {
                "left": [interval.left for interval in counts.index],
                "right": [interval.right for interval in counts.index],
                "count": counts.to_numpy(),
            }
        )
        plot.quad(
            top="count",
            bottom=0,
            left="left",
            right="right",
            source=ColumnDataSource(frame),
            fill_color=MARKET_BLUE,
            line_color="white",
            alpha=0.70,
        )
    plot.xaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def monthly_returns_timeseries(returns: pd.Series):
    """Return a monthly returns time series."""
    monthly = ((1.0 + pd.Series(returns).dropna()).resample("ME").prod() - 1.0).dropna()
    frame = _plot_frame(monthly, "return")
    frame["color"] = [MARKET_UP if value >= 0 else MARKET_DOWN for value in frame["return"]]
    plot = figure(
        title="Monthly Returns Timeseries",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.vbar(
            "date",
            width=20 * 24 * 60 * 60 * 1000,
            top="return",
            source=ColumnDataSource(frame),
            color="color",
            alpha=0.70,
        )
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def rolling_returns(returns: pd.Series, window: int = 126):
    """Return a trailing compounded rolling return chart."""
    values = (
        (1.0 + pd.Series(returns).dropna())
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        - 1.0
    )
    frame = _plot_frame(values, "rolling_return").dropna()
    plot = figure(
        title=f"Rolling Return ({window}-Bar)",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        source = ColumnDataSource(frame)
        plot.line(
            "date",
            "rolling_return",
            source=source,
            line_width=2,
            color=MARKET_BLUE,
            name="rolling_return",
        )
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def rolling_volatility(returns: pd.Series, window: int = 126, periods: int = 252):
    """Return annualized rolling volatility."""
    values = pd.Series(returns).dropna().rolling(window, min_periods=2).std() * periods ** 0.5
    frame = _plot_frame(values, "rolling_volatility").dropna()
    plot = figure(
        title=f"Rolling Volatility ({window}-Bar)",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_volatility"], line_width=2, color="#8c564b")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def return_quantiles(returns: pd.Series):
    """Return a return quantile bar chart."""
    values = pd.Series(returns).dropna()
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    frame = pd.DataFrame(
        {
            "quantile": [f"{int(level * 100)}%" for level in quantiles.index],
            "return": quantiles.to_numpy(),
            "color": [MARKET_UP if value >= 0 else MARKET_DOWN for value in quantiles],
        }
    )
    plot = figure(
        title="Return Quantiles",
        x_range=list(frame["quantile"]),
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        plot.vbar("quantile", 0.7, "return", source=ColumnDataSource(frame), color="color")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def holdings(positions: pd.DataFrame):
    """Return number of active holdings over time."""
    wide = _positions_wide(positions)
    values = wide.drop(columns=["cash"], errors="ignore")
    counts = values.ne(0).sum(axis=1) if not values.empty else pd.Series(dtype="float64")
    frame = _plot_frame(counts, "holdings")
    plot = figure(title="Holdings", x_axis_type="datetime", height=300, sizing_mode="stretch_width")
    if not frame.empty:
        source = ColumnDataSource(frame)
        plot.line(
            "date",
            "holdings",
            source=source,
            line_width=2,
            color=MARKET_BLUE,
            name="holdings",
        )
    _make_static_chart(plot)
    return plot

def long_short_holdings(positions: pd.DataFrame):
    """Return long and short holding counts over time."""
    wide = _positions_wide(positions).drop(columns=["cash"], errors="ignore")
    long_counts = wide.gt(0).sum(axis=1) if not wide.empty else pd.Series(dtype="float64")
    short_counts = wide.lt(0).sum(axis=1) if not wide.empty else pd.Series(dtype="float64")
    frame = pd.DataFrame({"long": long_counts, "short": short_counts})
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    frame["date"] = _normalize_dates(frame["date"])
    plot = figure(
        title="Long/Short Holdings",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["long"], line_width=2, color=MARKET_UP, legend_label="Long")
        plot.line(
            frame["date"],
            frame["short"],
            line_width=2,
            color=MARKET_DOWN,
            legend_label="Short",
        )
    _make_static_chart(plot)
    return plot

def gross_leverage(positions: pd.DataFrame, equity: pd.Series | None = None):
    """Return gross leverage computed from position values."""
    wide = _positions_wide(positions)
    assets = wide.drop(columns=["cash"], errors="ignore")
    gross = assets.abs().sum(axis=1)
    if equity is not None and not pd.Series(equity).empty:
        equity_base = _align_series_to_index(pd.Series(equity), gross.index)
    else:
        equity_base = wide.sum(axis=1)
    equity_base = equity_base.abs().replace(0, pd.NA)
    leverage = (gross / equity_base).astype("float64").fillna(0.0)
    frame = _plot_frame(leverage, "gross_leverage")
    plot = figure(
        title="Gross Leverage",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        source = ColumnDataSource(frame)
        plot.line(
            "date",
            "gross_leverage",
            source=source,
            line_width=2,
            color="#8c564b",
            name="gross_leverage",
        )
    _make_static_chart(plot)
    return plot

def position_concentration(positions: pd.DataFrame):
    """Return max and median absolute position concentration."""
    wide = _positions_wide(positions)
    assets = wide.drop(columns=["cash"], errors="ignore")
    totals = assets.abs().sum(axis=1).replace(0, pd.NA)
    weights = assets.abs().div(totals, axis=0).fillna(0.0)
    active_weights = weights.where(weights > 0)
    frame = pd.DataFrame(
        {
            "max": weights.max(axis=1),
            "median": active_weights.median(axis=1).fillna(0.0),
        }
    )
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    frame["date"] = _normalize_dates(frame["date"])
    plot = figure(
        title="Position Concentration",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        source = ColumnDataSource(frame)
        plot.line(
            "date",
            "max",
            source=source,
            line_width=2,
            color=MARKET_BLUE,
            legend_label="Max",
            name="max",
        )
        plot.line(
            "date",
            "median",
            source=source,
            line_width=2,
            color="#ff7f0e",
            legend_label="Median",
            name="median",
        )
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def turnover(
    transactions: pd.DataFrame,
    positions: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
):
    """Return daily turnover from transaction notional."""
    frame = _transactions_frame(transactions)
    plot = figure(
        title="Daily Turnover",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if frame.empty:
        _make_static_chart(plot)
        return plot
    daily = frame.groupby(frame["date"].dt.floor("D"))["notional"].sum()
    if equity is not None and not pd.Series(equity).empty:
        equity_base = _align_series_to_index(pd.Series(equity), daily.index)
        daily = daily / equity_base.abs().replace(0, pd.NA)
    elif positions is not None and not positions.empty:
        wide = _positions_wide(positions)
        equity = wide.sum(axis=1).abs().replace(0, pd.NA)
        daily = daily / equity.reindex(daily.index, method="nearest").ffill()
    daily = daily.astype("float64").fillna(0.0)
    data = _plot_frame(daily, "turnover")
    source = ColumnDataSource(data)
    plot.line("date", "turnover", source=source, line_width=2, color=MARKET_BLUE, name="turnover")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def daily_volume(transactions: pd.DataFrame):
    """Return daily filled share or contract volume."""
    frame = _transactions_frame(transactions)
    plot = figure(
        title="Daily Fill Volume",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        daily = frame.assign(abs_size=frame["size"].abs()).groupby(
            frame["date"].dt.floor("D")
        )["abs_size"].sum()
        data = _plot_frame(daily, "volume")
        source = ColumnDataSource(data)
        plot.vbar(
            "date",
            width=20 * 60 * 60 * 1000,
            top="volume",
            source=source,
            color=MARKET_BLUE,
            alpha=0.70,
            name="volume",
        )
    _make_static_chart(plot)
    return plot

def _transaction_bar_hours(transactions: pd.DataFrame, timezone: str | ZoneInfo) -> pd.Series:
    """Return market-local hour-of-day values from transaction bar timestamps."""
    frame = pd.DataFrame(transactions).copy()
    if frame.empty:
        return pd.Series(dtype="float64")
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("datetime") or lowered.get("date") or lowered.get("timestamp")
    if date_col is None:
        return pd.Series(dtype="float64")
    return _market_hours(frame[date_col], timezone)

def _market_hours(values: pd.Series, timezone: str | ZoneInfo) -> pd.Series:
    """Return hour values converted to the report market timezone."""
    zone = _coerce_timezone(timezone)
    if pd.api.types.is_numeric_dtype(values):
        dates = pd.to_datetime(values, unit="s", errors="coerce", utc=True)
    else:
        dates = pd.to_datetime(values, errors="coerce")
        if not isinstance(dates.dtype, pd.DatetimeTZDtype):
            dates = dates.dt.tz_localize("UTC")
    return dates.dt.tz_convert(zone).dt.hour

def transaction_time_histogram(
    transactions: pd.DataFrame,
    *,
    timezone: str | ZoneInfo = DEFAULT_REPORT_TIMEZONE,
):
    """Return fill count histogram by market-local bar timestamp hour of day."""
    zone = _coerce_timezone(timezone)
    plot = figure(
        title="Fill Bar Time Histogram",
        x_range=[str(hour) for hour in range(24)],
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    hours = _transaction_bar_hours(transactions, zone).dropna().astype(int)
    if not hours.empty:
        counts = hours.value_counts().reindex(range(24), fill_value=0)
        data = pd.DataFrame({"hour": counts.index.astype(str), "count": counts.to_numpy()})
        plot.vbar("hour", 0.8, "count", source=ColumnDataSource(data), color=MARKET_BLUE)
    plot.xaxis.axis_label = f"Market Hour ({_timezone_label(zone)})"
    plot.yaxis.axis_label = "Transactions"
    _make_static_chart(plot)
    return plot

def rolling_sharpe(rolling: pd.Series, window: int = 126):
    """Return a rolling Sharpe figure."""
    frame = _plot_frame(rolling, "rolling_sharpe").dropna()
    plot = figure(
        title=f"Rolling Sharpe ({window}-Bar)",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_sharpe"], line_width=2, color="#9467bd")
    _make_static_chart(plot)
    return plot

def rolling_beta(rolling: pd.Series):
    """Return a rolling beta figure."""
    frame = _plot_frame(rolling, "rolling_beta").dropna()
    plot = figure(
        title="Rolling Beta",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_beta"], line_width=2, color="#17becf")
    _make_static_chart(plot)
    return plot

def trade_distribution(distribution: pd.DataFrame):
    """Return a trade PnL histogram figure."""
    plot = figure(
        title="Closed Trade PnL Distribution",
        height=300,
        sizing_mode="stretch_width",
    )
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
    plot.xaxis.axis_label = "PnL"
    plot.yaxis.axis_label = "Closed Trades"
    _make_static_chart(plot)
    return plot

def _attribution_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a date-column copy with timezone-naive timestamps for Bokeh."""
    result = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    if isinstance(result["date"].dtype, pd.DatetimeTZDtype):
        result["date"] = result["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return result

def _attribution_lines(frame: pd.DataFrame, *, title: str, percent: bool):
    """Return one line per column of a dated wide frame."""
    data = _attribution_frame(frame)
    plot = figure(
        title=title,
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for index, name in enumerate(frame.columns):
        plot.line(
            data["date"],
            data[name],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=str(name),
        )
    if percent:
        plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def attribution_exposures(exposures: pd.DataFrame):
    """Return daily portfolio factor exposures over time."""
    return _attribution_lines(exposures, title="Factor Exposures", percent=False)

def attribution_cumulative(contributions: pd.DataFrame):
    """Return compounded return contribution per factor."""
    cumulative = (1.0 + contributions.fillna(0.0)).cumprod() - 1.0
    return _attribution_lines(cumulative, title="Cumulative Attribution", percent=True)

def attribution_common_specific(attribution: pd.DataFrame):
    """Return compounded common, specific and total returns."""
    columns = [
        name
        for name in ("common_returns", "specific_returns", "total_returns")
        if name in attribution.columns
    ]
    frame = attribution.loc[:, columns]
    cumulative = (1.0 + frame.fillna(0.0)).cumprod() - 1.0
    return _attribution_lines(cumulative, title="Common vs Specific", percent=True)

def attribution_summary_bar(exposure_summary: pd.DataFrame):
    """Return cumulative return contribution by factor."""
    frame = pd.DataFrame(exposure_summary).copy()
    column = "cumulative_return_contribution"
    if column not in frame:
        frame[column] = pd.Series(dtype="float64")
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "factor"})
    frame["factor"] = frame["factor"].astype(str)
    frame["color"] = [MARKET_UP if value >= 0 else MARKET_DOWN for value in frame[column]]
    plot = figure(
        title="Return Contribution by Factor",
        x_range=list(frame["factor"]),
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        plot.vbar("factor", 0.7, column, source=ColumnDataSource(frame), color="color")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot
