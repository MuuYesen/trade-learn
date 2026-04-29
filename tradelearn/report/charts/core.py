"""Core Bokeh chart builders for reports."""

from __future__ import annotations

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.transform import linear_cmap


def market_replay(
    market_data: pd.DataFrame,
    fills: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
):
    """Return a 1.x-style market replay grid with equity, P/L, OHLC, and volume."""
    frame = _market_frame(market_data)
    if frame.empty:
        return price_trades(market_data, fills)

    replay_range = None
    plots = []

    if equity is not None and not equity.empty:
        equity_frame = _plot_frame(equity, "equity").dropna()
        if not equity_frame.empty:
            equity_plot = figure(
                title="Equity",
                x_axis_type="datetime",
                height=140,
                sizing_mode="stretch_width",
                tools="xpan,xwheel_zoom,box_zoom,reset,save",
                active_drag="xpan",
                active_scroll="xwheel_zoom",
            )
            equity_plot.line(
                equity_frame["date"],
                equity_frame["equity"] / equity_frame["equity"].iloc[0],
                line_width=2,
                color="#1f77b4",
                legend_label="Strategy",
            )
            equity_plot.yaxis.axis_label = "Relative Equity"
            replay_range = equity_plot.x_range
            plots.append(equity_plot)

    fills_frame = (
        _fills_plot_frame(fills)
        if fills is not None and not fills.empty and {"datetime", "price"}.issubset(fills.columns)
        else pd.DataFrame()
    )
    closed_fills = pd.DataFrame()
    if not fills_frame.empty and "trade_closed" in fills_frame:
        closed_fills = fills_frame[fills_frame["trade_closed"].astype(bool)]
    if not closed_fills.empty and "pnl" in closed_fills:
        pl_plot = _linked_figure(
            "Profit / Loss",
            height=100,
            x_range=replay_range,
        )
        pl_plot.add_layout(
            Span(location=0, dimension="width", line_dash="dashed", line_color="#666666")
        )
        colors = ["#2ca02c" if pnl >= 0 else "#d62728" for pnl in closed_fills["pnl"]]
        pl_plot.scatter(
            closed_fills["date"],
            closed_fills["pnl"],
            size=9,
            color=colors,
            legend_label="Closed P/L",
        )
        pl_plot.yaxis.axis_label = "P/L"
        replay_range = pl_plot.x_range
        plots.append(pl_plot)

    price_plot = _linked_figure(
        "OHLC / Trades",
        height=360,
        x_range=replay_range,
    )
    source = ColumnDataSource(frame)
    has_ohlc = {"open", "high", "low", "close"}.issubset(frame.columns)
    if has_ohlc:
        inc = frame["close"] >= frame["open"]
        width = _bar_width_ms(frame["date"])
        price_plot.segment(
            "date",
            "high",
            "date",
            "low",
            source=source,
            color="#2f3b45",
            line_width=1,
        )
        price_plot.vbar(
            frame.loc[inc, "date"],
            width,
            frame.loc[inc, "open"],
            frame.loc[inc, "close"],
            fill_color="#3aa76d",
            line_color="#2f3b45",
            alpha=0.82,
            legend_label="Up",
        )
        price_plot.vbar(
            frame.loc[~inc, "date"],
            width,
            frame.loc[~inc, "open"],
            frame.loc[~inc, "close"],
            fill_color="#d65f5f",
            line_color="#2f3b45",
            alpha=0.82,
            legend_label="Down",
        )
        renderers = price_plot.renderers[-3:]
    else:
        renderer = price_plot.line(
            "date",
            "close",
            source=source,
            line_width=2,
            color="#1f77b4",
            legend_label="Close",
        )
        renderers = [renderer]
    if not fills_frame.empty:
        buys = fills_frame[fills_frame["side"].str.lower().eq("buy")]
        sells = fills_frame[fills_frame["side"].str.lower().eq("sell")]
        if not buys.empty:
            price_plot.scatter(
                buys["date"],
                buys["price"],
                marker="triangle",
                size=11,
                color="#169c5a",
                line_color="#0f5f39",
                legend_label="Buy",
            )
        if not sells.empty:
            price_plot.scatter(
                sells["date"],
                sells["price"],
                marker="inverted_triangle",
                size=11,
                color="#c43c39",
                line_color="#7f2524",
                legend_label="Sell",
            )
    price_plot.yaxis.axis_label = "Price"
    if has_ohlc:
        _add_ohlc_hover(price_plot, renderers)
    else:
        _add_close_hover(price_plot, renderers)
    replay_range = price_plot.x_range
    plots.append(price_plot)

    if "volume" in frame and frame["volume"].notna().any():
        volume_plot = _linked_figure(
            "Volume",
            height=110,
            x_range=replay_range,
        )
        volume_plot.vbar(
            frame["date"],
            _bar_width_ms(frame["date"]),
            frame["volume"],
            color="#8aa1b2",
            alpha=0.72,
        )
        volume_plot.yaxis.axis_label = "Volume"
        plots.append(volume_plot)

    for plot in plots[:-1]:
        plot.xaxis.visible = False
    for plot in plots:
        plot.toolbar.logo = None
        if plot.legend:
            plot.legend.location = "top_left"
            plot.legend.click_policy = "hide"
    return gridplot(
        plots,
        ncols=1,
        sizing_mode="stretch_width",
        toolbar_location="right",
        merge_tools=True,
    )


def price_trades(market_data: pd.DataFrame, fills: pd.DataFrame | None = None):
    """Return a price curve with buy/sell fill markers."""
    frame = _market_frame(market_data)
    plot = figure(
        title="Price / Trades",
        x_axis_type="datetime",
        height=360,
        sizing_mode="stretch_width",
    )
    if frame.empty:
        return plot
    plot.line(
        frame["date"],
        frame["close"],
        line_width=2,
        color="#1f77b4",
        legend_label="Close",
    )
    if fills is not None and not fills.empty and {"datetime", "price"}.issubset(fills.columns):
        fill_frame = _fills_plot_frame(fills)
        buys = fill_frame[fill_frame["side"].str.lower().eq("buy")]
        sells = fill_frame[fill_frame["side"].str.lower().eq("sell")]
        if not buys.empty:
            plot.scatter(
                buys["date"],
                buys["price"],
                marker="triangle",
                size=10,
                color="#2ca02c",
                legend_label="Buy",
            )
        if not sells.empty:
            plot.scatter(
                sells["date"],
                sells["price"],
                marker="inverted_triangle",
                size=10,
                color="#d62728",
                legend_label="Sell",
            )
    plot.legend.location = "top_left"
    return plot


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
    if drawdowns is not None and not drawdowns.empty:
        _add_drawdown_markers(plot, equity, drawdowns)
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


def rolling_beta(rolling: pd.Series):
    """Return a rolling beta figure."""
    frame = _plot_frame(rolling, "rolling_beta").dropna()
    plot = figure(
        title="Rolling Beta",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_beta"], line_width=2, color="#17becf")
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


def factor_long_short_returns(returns: pd.DataFrame):
    """Return a factor long-short returns figure."""
    frame = returns.reset_index().rename(columns={returns.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Factor Long-Short Returns",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = {
        "long": "#2ca02c",
        "short": "#d62728",
        "spread": "#1f77b4",
    }
    for column in returns.columns:
        plot.line(
            frame["date"],
            frame[column],
            line_width=2,
            color=colors.get(str(column), "#9467bd"),
            legend_label=str(column),
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


def factor_rank_ic(rank_ic: pd.Series):
    """Return a factor rank IC figure."""
    frame = _plot_frame(rank_ic, "rank_ic").dropna()
    plot = figure(
        title="Factor Rank IC",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rank_ic"], line_width=2, color="#1f77b4")
        plot.line(
            frame["date"],
            frame["rank_ic"].expanding(min_periods=1).mean(),
            line_width=2,
            color="#ff7f0e",
            legend_label="Expanding Mean",
        )
    return plot


def factor_turnover(turnover: pd.Series, autocorrelation: pd.Series):
    """Return a factor turnover and autocorrelation figure."""
    turnover_frame = _plot_frame(turnover, "turnover").dropna()
    autocorrelation_frame = _plot_frame(autocorrelation, "autocorrelation").dropna()
    plot = figure(
        title="Factor Turnover",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not turnover_frame.empty:
        plot.line(
            turnover_frame["date"],
            turnover_frame["turnover"],
            line_width=2,
            color="#1f77b4",
            legend_label="Turnover",
        )
    if not autocorrelation_frame.empty:
        plot.line(
            autocorrelation_frame["date"],
            autocorrelation_frame["autocorrelation"],
            line_width=2,
            color="#ff7f0e",
            legend_label="Autocorrelation",
        )
    return plot


def _add_drawdown_markers(plot, equity: pd.Series, drawdowns: pd.DataFrame) -> None:
    """Add peak and valley markers for drawdown episodes."""
    for column, name, color, marker in [
        ("peak", "drawdown_peak", "#2ca02c", "triangle"),
        ("valley", "drawdown_valley", "#d62728", "inverted_triangle"),
    ]:
        if column not in drawdowns:
            continue
        dates = pd.to_datetime(drawdowns[column]).dropna()
        values = equity.reindex(dates).dropna()
        if values.empty:
            continue
        marker_frame = _plot_frame(values, "equity")
        plot.scatter(
            marker_frame["date"],
            marker_frame["equity"],
            marker=marker,
            size=9,
            color=color,
            legend_label=name.replace("_", " ").title(),
            name=name,
        )


def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame


def _market_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    """Return normalized OHLCV data for plotting."""
    if market_data.empty:
        return pd.DataFrame(columns=["date", "close"])
    frame = market_data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "close" not in frame:
        return pd.DataFrame(columns=["date", "close"])
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date", "close"])


def _fills_plot_frame(fills: pd.DataFrame) -> pd.DataFrame:
    """Return fills with timezone-naive dates for plotting."""
    frame = fills.copy()
    frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame.dropna(subset=["date", "price"])


def _linked_figure(title: str, *, height: int, x_range=None):
    """Return a linked market replay figure."""
    kwargs = {"x_range": x_range} if x_range is not None else {}
    return figure(
        title=title,
        x_axis_type="datetime",
        height=height,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        active_drag="xpan",
        active_scroll="xwheel_zoom",
        **kwargs,
    )


def _bar_width_ms(dates: pd.Series) -> float:
    """Return a candle width in milliseconds from adjacent timestamps."""
    numeric = pd.to_datetime(dates).astype("int64") // 1_000_000
    diffs = pd.Series(numeric).diff().dropna()
    if diffs.empty:
        return 24 * 60 * 60 * 1000 * 0.7
    return float(diffs.median() * 0.7)


def _add_ohlc_hover(plot, renderers) -> None:
    """Attach OHLC hover tool to the main market replay plot."""
    plot.add_tools(
        HoverTool(
            renderers=list(renderers),
            mode="vline",
            formatters={"@date": "datetime"},
            tooltips=[
                ("Date", "@date{%F %T}"),
                ("Open", "@open{0,0.0000}"),
                ("High", "@high{0,0.0000}"),
                ("Low", "@low{0,0.0000}"),
                ("Close", "@close{0,0.0000}"),
                ("Volume", "@volume{0,0}"),
            ],
        )
    )


def _add_close_hover(plot, renderers) -> None:
    """Attach close-price hover tool to the main market replay plot."""
    plot.add_tools(
        HoverTool(
            renderers=list(renderers),
            mode="vline",
            formatters={"@date": "datetime"},
            tooltips=[
                ("Date", "@date{%F %T}"),
                ("Close", "@close{0,0.0000}"),
            ],
        )
    )
