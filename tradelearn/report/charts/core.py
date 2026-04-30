"""Core Bokeh chart builders for reports."""

from __future__ import annotations

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    HoverTool,
    NumeralTickFormatter,
    Range1d,
    Span,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

MARKET_UP = "#2ca36c"
MARKET_DOWN = "#d65a5a"
MARKET_UP_LINE = "#1f6f4a"
MARKET_DOWN_LINE = "#9f3434"
MARKET_BLUE = "#2f6fa8"
MARKET_MUTED = "#6f7f8d"
MARKET_GRID = "#e8edf2"
MARKET_BORDER = "#ccd7e0"
MARKET_BACKGROUND = "#fbfcfe"
MARKET_DRAWDOWN = "#fff3c4"


def market_replay(
    market_data: pd.DataFrame,
    fills: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
):
    """Return a 1.x-style market replay grid with equity, P/L, OHLC, trades, and volume."""
    frame = _market_frame(market_data)
    if frame.empty:
        return price_trades(market_data, fills)

    frame = frame.reset_index(drop=True).copy()
    frame["bar_index"] = frame.index.astype(float)
    has_ohlc = {"open", "high", "low", "close"}.issubset(frame.columns)
    has_volume = "volume" in frame and frame["volume"].notna().any()
    fills_frame = _fills_plot_frame(fills) if _fills_have_plot_columns(fills) else pd.DataFrame()
    fills_frame = _attach_bar_index(fills_frame, frame)
    trades_frame = _trade_segments(fills_frame, frame)

    x_pad = max((len(frame) - 1) / 20, 1.0)
    x_range = Range1d(
        start=float(frame["bar_index"].iloc[0] - x_pad),
        end=float(frame["bar_index"].iloc[-1] + x_pad),
        min_interval=10,
    )
    source = ColumnDataSource(frame)
    plots = []

    if equity is not None and not equity.empty:
        equity_plot = _market_section("Equity", height=150, x_range=x_range)
        equity_frame = _equity_replay_frame(equity, frame)
        if not equity_frame.empty:
            eq_source = ColumnDataSource(equity_frame)
            equity_plot.patch(
                "bar_index",
                "drawdown_fill",
                source=ColumnDataSource(
                    {
                        "bar_index": list(equity_frame["bar_index"])
                        + list(reversed(equity_frame["bar_index"].tolist())),
                        "drawdown_fill": list(equity_frame["relative_equity"])
                        + list(reversed(equity_frame["high_watermark"].tolist())),
                    }
                ),
                fill_color=MARKET_DRAWDOWN,
                line_color="#e2b74e",
                fill_alpha=0.55,
                legend_label="Drawdown",
            )
            strategy_line = equity_plot.line(
                "bar_index",
                "relative_equity",
                source=eq_source,
                line_width=2.0,
                color=MARKET_BLUE,
                legend_label="Strategy",
            )
            equity_plot.line(
                "bar_index",
                "buy_hold",
                source=eq_source,
                line_width=1.1,
                color=MARKET_MUTED,
                line_dash="dashed",
                legend_label="Buy&Hold",
            )
            peak = equity_frame["relative_equity"].idxmax()
            final = equity_frame.index[-1]
            dd = equity_frame["drawdown"]
            max_dd = dd.idxmin()
            equity_plot.scatter(
                [equity_frame.loc[peak, "bar_index"]],
                [equity_frame.loc[peak, "relative_equity"]],
                color="#49c6d8",
                line_color="#1d7d8d",
                size=9,
                legend_label=f"Peak ({equity_frame.loc[peak, 'relative_equity']:.1%})",
            )
            equity_plot.scatter(
                [equity_frame.loc[final, "bar_index"]],
                [equity_frame.loc[final, "relative_equity"]],
                color=MARKET_BLUE,
                line_color="#1a4f79",
                size=9,
                legend_label=f"Final ({equity_frame.loc[final, 'relative_equity']:.1%})",
            )
            equity_plot.scatter(
                [equity_frame.loc[max_dd, "bar_index"]],
                [equity_frame.loc[max_dd, "relative_equity"]],
                color=MARKET_DOWN,
                line_color=MARKET_DOWN_LINE,
                size=9,
                legend_label=f"Max Drawdown ({equity_frame.loc[max_dd, 'drawdown']:.1%})",
            )
            equity_plot.yaxis.axis_label = "Equity"
            equity_plot.yaxis.formatter = NumeralTickFormatter(format="0,0.[00]%")
            _add_line_hover(
                equity_plot,
                [strategy_line],
                [("Date", "@date{%F %T}"), ("Equity", "@relative_equity{0,0.[00]%}")],
            )
            plots.append(equity_plot)
    else:
        equity_plot = None

    if not trades_frame.empty:
        pl_plot = _market_section("Profit / Loss", height=120, x_range=x_range)
        pl_plot.add_layout(
            Span(location=0, dimension="width", line_color=MARKET_MUTED, line_dash="dashed")
        )
        trade_source = ColumnDataSource(trades_frame)
        win = trades_frame[trades_frame["return_pct"] >= 0]
        loss = trades_frame[trades_frame["return_pct"] < 0]
        if not win.empty:
            pl_plot.scatter(
                "exit_bar",
                "return_pct",
                source=ColumnDataSource(win),
                marker="triangle",
                fill_color=MARKET_UP,
                line_color=MARKET_UP_LINE,
                size="marker_size",
                legend_label="Winning Trades",
            )
        if not loss.empty:
            pl_plot.scatter(
                "exit_bar",
                "return_pct",
                source=ColumnDataSource(loss),
                marker="inverted_triangle",
                fill_color=MARKET_DOWN,
                line_color=MARKET_DOWN_LINE,
                size="marker_size",
                legend_label="Losing Trades",
            )
        min_ret = float(trades_frame["return_pct"].min())
        max_ret = float(trades_frame["return_pct"].max())
        ret_span = max(max_ret - min_ret, 0.01)
        marker_pad = max(ret_span * 0.75, 0.01)
        pl_plot.y_range.start = min(min_ret - marker_pad, -0.002)
        pl_plot.y_range.end = max(max_ret + marker_pad, 0.002)
        hidden = pl_plot.scatter(
            "exit_bar",
            "return_pct",
            source=trade_source,
            marker="circle",
            size=1,
            alpha=0.0,
        )
        pl_plot.yaxis.axis_label = "Profit / Loss"
        pl_plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
        _add_line_hover(
            pl_plot,
            [hidden],
            [
                ("Exit", "@exit_datetime{%F %T}"),
                ("Size", "@size{0,0.####}"),
                ("P/L", "@return_pct{+0.[000]%}"),
            ],
            vline=False,
        )
        plots.append(pl_plot)

    price_plot = _market_section("OHLC / Trades", height=430, x_range=x_range)
    if has_ohlc:
        inc = frame["close"] >= frame["open"]
        price_plot.segment(
            "bar_index",
            "high",
            "bar_index",
            "low",
            source=source,
            color="#2f3b45",
            line_width=1,
            alpha=0.78,
        )
        up = frame.loc[inc]
        down = frame.loc[~inc]
        if not up.empty:
            price_plot.vbar(
                "bar_index",
                0.8,
                "open",
                "close",
                source=ColumnDataSource(up),
                fill_color=MARKET_UP,
                line_color=MARKET_UP_LINE,
                fill_alpha=0.82,
            )
        if not down.empty:
            price_plot.vbar(
                "bar_index",
                0.8,
                "open",
                "close",
                source=ColumnDataSource(down),
                fill_color=MARKET_DOWN,
                line_color=MARKET_DOWN_LINE,
                fill_alpha=0.82,
            )
        if not up.empty:
            price_plot.scatter(
                x=[float("nan")],
                y=[float("nan")],
                marker="square",
                size=9,
                fill_color=MARKET_UP,
                line_color=MARKET_UP_LINE,
                legend_label="Up",
            )
        if not down.empty:
            price_plot.scatter(
                x=[float("nan")],
                y=[float("nan")],
                marker="square",
                size=9,
                fill_color=MARKET_DOWN,
                line_color=MARKET_DOWN_LINE,
                legend_label="Down",
            )
        price_renderer = price_plot.scatter(
            "bar_index",
            "close",
            source=source,
            marker="circle",
            size=1,
            alpha=0.0,
        )
        _add_ohlc_index_hover(price_plot, [price_renderer])
    else:
        price_renderer = price_plot.line(
            "bar_index",
            "close",
            source=source,
            line_width=1.6,
            color="#1f77b4",
            legend_label="Close",
        )
        _add_close_index_hover(price_plot, [price_renderer])

    if not trades_frame.empty:
        trade_source = ColumnDataSource(trades_frame)
        price_plot.multi_line(
            xs="line_xs",
            ys="line_ys",
            source=trade_source,
            line_color="line_color",
            line_width=4,
            line_alpha=0.62,
            line_dash="dotted",
            legend_label=f"Trades ({len(trades_frame)})",
        )
    if not fills_frame.empty:
        buys = fills_frame[fills_frame["side"].str.lower().eq("buy")]
        sells = fills_frame[fills_frame["side"].str.lower().eq("sell")]
        if not buys.empty:
            price_plot.scatter(
                "bar_index",
                "price",
                source=ColumnDataSource(buys),
                marker="triangle",
                size=17,
                color=MARKET_UP,
                line_color="white",
                legend_label="Buy",
            )
        if not sells.empty:
            price_plot.scatter(
                "bar_index",
                "price",
                source=ColumnDataSource(sells),
                marker="inverted_triangle",
                size=17,
                color=MARKET_DOWN,
                line_color="white",
                legend_label="Sell",
            )
    price_plot.yaxis.axis_label = "Price"
    plots.append(price_plot)

    if has_volume:
        volume_plot = _market_section("Volume", height=115, x_range=x_range)
        volume_source = ColumnDataSource(
            frame.assign(
                volume_color=[
                    MARKET_UP if close >= open_ else MARKET_DOWN
                    for close, open_ in zip(
                        frame["close"],
                        frame.get("open", frame["close"]),
                        strict=True,
                    )
                ]
            )
        )
        renderer = volume_plot.vbar(
            "bar_index",
            0.8,
            "volume",
            source=volume_source,
            color="volume_color",
            alpha=0.42,
        )
        volume_plot.yaxis.axis_label = "Volume"
        volume_plot.yaxis.formatter = NumeralTickFormatter(format="0 a")
        _add_line_hover(
            volume_plot,
            [renderer],
            [("Date", "@date{%F %T}"), ("Volume", "@volume{0.00 a}")],
        )
        plots.append(volume_plot)

    for plot in plots[:-1]:
        plot.xaxis.visible = False
    for plot in plots:
        _style_market_section(plot)
    _style_market_legend(equity_plot, compact=True)
    _style_market_legend(pl_plot if not trades_frame.empty else None)
    _style_market_legend(price_plot, compact=True, large_glyphs=True)

    return gridplot(
        plots,
        ncols=1,
        sizing_mode="stretch_width",
        toolbar_location=None,
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


def _fills_have_plot_columns(fills: pd.DataFrame | None) -> bool:
    """Return True when fills can be projected onto the market replay chart."""
    return fills is not None and not fills.empty and {"datetime", "price", "side"}.issubset(
        fills.columns
    )


def _attach_bar_index(fills: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest market bar index to each fill."""
    if fills.empty:
        return fills
    bars = frame[["date", "bar_index"]].sort_values("date")
    projected = pd.merge_asof(
        fills.sort_values("date"),
        bars,
        on="date",
        direction="nearest",
    )
    return projected.dropna(subset=["bar_index"])


def _equity_replay_frame(equity: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """Return equity aligned to replay bar indices with 1.x-style derived columns."""
    equity_frame = _plot_frame(pd.Series(equity).dropna(), "equity")
    if equity_frame.empty:
        return pd.DataFrame()
    bars = frame[["date", "bar_index", "close"]].sort_values("date")
    projected = pd.merge_asof(
        bars,
        equity_frame.sort_values("date"),
        on="date",
        direction="nearest",
    ).dropna(subset=["equity"])
    if projected.empty:
        return projected
    initial_equity = projected["equity"].iloc[0]
    initial_close = projected["close"].iloc[0]
    if not initial_equity or not initial_close:
        return pd.DataFrame()
    projected["relative_equity"] = projected["equity"] / initial_equity
    projected["buy_hold"] = projected["close"] / initial_close
    projected["high_watermark"] = projected["relative_equity"].cummax()
    projected["drawdown"] = projected["relative_equity"] / projected["high_watermark"] - 1.0
    return projected.reset_index(drop=True)


def _trade_segments(fills: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Build approximate 1.x-style entry-exit trade segments from fill rows."""
    if fills.empty:
        return pd.DataFrame()
    active: dict[str, dict[str, object]] = {}
    segments = []
    for _, fill in fills.sort_values("date").iterrows():
        side = str(fill.get("side", "")).lower()
        signed = float(fill.get("size", 0.0) or 0.0)
        if signed == 0:
            signed = abs(float(fill.get("qty", 0.0) or 0.0))
            signed = signed if side == "buy" else -signed
        data_name = str(fill.get("data", "") or "__default__")
        current = active.get(data_name)
        direction = 1 if signed > 0 else -1
        if current is None:
            active[data_name] = {
                "bar": float(fill["bar_index"]),
                "price": float(fill["price"]),
                "size": abs(signed),
                "direction": direction,
                "datetime": fill["date"],
            }
            continue
        if int(current["direction"]) == direction:
            current["size"] = float(current["size"]) + abs(signed)
            continue
        entry_price = float(current["price"])
        exit_price = float(fill["price"])
        entry_bar = float(current["bar"])
        exit_bar = float(fill["bar_index"])
        size = min(float(current["size"]), abs(signed))
        if entry_price:
            return_pct = (exit_price / entry_price - 1.0) * int(current["direction"])
        else:
            return_pct = 0.0
        segments.append(
            {
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "entry_datetime": current["datetime"],
                "exit_datetime": fill["date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size * int(current["direction"]),
                "return_pct": return_pct,
                "marker_size": _trade_marker_size(size),
                "line_xs": [entry_bar, exit_bar],
                "line_ys": [entry_price, exit_price],
                "line_color": MARKET_UP_LINE if return_pct >= 0 else MARKET_DOWN_LINE,
            }
        )
        remaining = float(current["size"]) - size
        if remaining > 1e-12:
            current["size"] = remaining
        elif abs(signed) > size:
            active[data_name] = {
                "bar": float(fill["bar_index"]),
                "price": float(fill["price"]),
                "size": abs(signed) - size,
                "direction": direction,
                "datetime": fill["date"],
            }
        else:
            active.pop(data_name, None)
    return pd.DataFrame(segments)


def _trade_marker_size(size: float) -> float:
    """Return a bounded marker size for the P/L section."""
    return max(8.0, min(20.0, 8.0 + abs(float(size)) ** 0.5))


def _market_section(title: str, *, height: int, x_range):
    """Return a 1.x-style linked replay figure."""
    return figure(
        title=title,
        x_axis_type="linear",
        height=height,
        x_range=x_range,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
        active_drag="xpan",
        active_scroll="xwheel_zoom",
        background_fill_color=MARKET_BACKGROUND,
        border_fill_color="white",
        outline_line_color=MARKET_BORDER,
    )


def _style_market_section(plot) -> None:
    """Apply common 1.x-inspired Bokeh styling."""
    plot.min_border_left = 8
    plot.min_border_top = 8
    plot.min_border_bottom = 8
    plot.min_border_right = 12
    plot.outline_line_color = MARKET_BORDER
    plot.outline_line_width = 1
    plot.grid.grid_line_color = MARKET_GRID
    plot.grid.grid_line_alpha = 0.75
    plot.grid.grid_line_width = 1
    plot.axis.axis_line_color = MARKET_BORDER
    plot.axis.major_tick_line_color = MARKET_BORDER
    plot.axis.minor_tick_line_color = None
    plot.axis.major_label_text_color = "#52616f"
    plot.axis.axis_label_text_color = "#52616f"
    plot.title.text_color = "#24313a"
    plot.title.text_font_size = "11pt"
    plot.title.text_font_style = "bold"
    plot.toolbar.logo = None
    plot.add_tools(CrosshairTool(dimensions="both"))
    _hide_market_legend(plot)


def _hide_market_legend(plot) -> None:
    if plot.legend:
        for legend in list(plot.legend):
            legend.visible = False


def _style_market_legend(plot, *, compact: bool = False, large_glyphs: bool = False) -> None:
    if plot is None or not plot.legend:
        return
    for legend in list(plot.legend):
        legend.visible = True
        legend.location = "top_left"
        legend.ncols = 2
        legend.border_line_width = 1
        legend.border_line_color = "#d7e0e7"
        legend.background_fill_color = "white"
        legend.background_fill_alpha = 0.88
        legend.padding = 3 if compact else 6
        legend.spacing = 0 if compact else 1
        legend.margin = 2 if compact else 4
        if compact and large_glyphs:
            legend.glyph_width = 26
            legend.glyph_height = 18
        else:
            legend.glyph_width = 18 if compact else (34 if large_glyphs else 28)
            legend.glyph_height = 12 if compact else (22 if large_glyphs else 18)
        legend.label_text_color = "#33424f"
        legend.label_text_font_size = "8pt" if compact else "9pt"
        legend.click_policy = "hide"


def _add_line_hover(plot, renderers, tooltips, *, vline: bool = True) -> None:
    """Attach a shared hover tool."""
    plot.add_tools(
        HoverTool(
            point_policy="follow_mouse",
            renderers=list(renderers),
            formatters={"@date": "datetime", "@exit_datetime": "datetime"},
            tooltips=list(tooltips),
            mode="vline" if vline else "mouse",
        )
    )


def _add_ohlc_index_hover(plot, renderers) -> None:
    """Attach OHLC hover tool for linear-index market replay plots."""
    _add_line_hover(
        plot,
        renderers,
        [
            ("Date", "@date{%F %T}"),
            ("#", "@bar_index{0,0}"),
            ("OHLC", "@open{0,0.0000}  @high{0,0.0000}  @low{0,0.0000}  @close{0,0.0000}"),
            ("Volume", "@volume{0,0}"),
        ],
    )


def _add_close_index_hover(plot, renderers) -> None:
    """Attach close-price hover tool for linear-index market replay plots."""
    _add_line_hover(
        plot,
        renderers,
        [("Date", "@date{%F %T}"), ("#", "@bar_index{0,0}"), ("Close", "@close{0,0.0000}")],
    )


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
