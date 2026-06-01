"""Core Bokeh chart builders for reports."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

import numpy as np
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
PORTFOLIO_VISIBLE_ASSET_LIMIT = 8


def market_replay(
    market_data: pd.DataFrame | Mapping[str, pd.DataFrame],
    fills: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
):
    """Return a market replay grid with equity, P/L, OHLC, trades, and volume."""
    if isinstance(market_data, Mapping):
        portfolio = _portfolio_market_replay(market_data, fills, equity, positions)
        if portfolio is not None:
            return portfolio
        market_data = next(iter(market_data.values()), pd.DataFrame())

    frame = _market_frame(market_data)
    if frame.empty:
        return price_trades(market_data, fills)

    frame = frame.reset_index(drop=True).copy()
    frame["bar_index"] = frame.index.astype(float)
    has_ohlc = {"open", "high", "low", "close"}.issubset(frame.columns)
    has_volume = "volume" in frame and frame["volume"].notna().any()
    fills_frame = _fills_plot_frame(fills) if _fills_have_plot_columns(fills) else pd.DataFrame()
    fills_frame = _attach_bar_index(fills_frame, frame)
    if has_ohlc and not fills_frame.empty:
        low_min = float(frame["low"].min())
        high_max = float(frame["high"].max())
        span = max(high_max - low_min, abs(high_max) * 1e-6, 1e-9)
        lo_bound = low_min - span * 0.5
        hi_bound = high_max + span * 0.5
        in_range = fills_frame["price"].between(lo_bound, hi_bound)
        fills_frame = fills_frame[in_range]
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
        low_min = float(frame["low"].min())
        high_max = float(frame["high"].max())
        y_pad = max((high_max - low_min) * 0.05, abs(high_max) * 0.001, 1e-9)
        price_plot.y_range = Range1d(start=low_min - y_pad, end=high_max + y_pad)
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
        price_plot.segment(
            x0="entry_bar",
            y0="entry_price",
            x1="exit_bar",
            y1="exit_price",
            source=ColumnDataSource(trades_frame),
            line_color="line_color",
            line_width=1.5,
            line_alpha=0.75,
            line_dash="dashed",
            legend_label="Trade",
        )
    if not fills_frame.empty:
        buys = fills_frame[fills_frame["side"].str.lower().eq("buy")]
        sells = fills_frame[fills_frame["side"].str.lower().eq("sell")]
        fill_renderers = []
        if not buys.empty:
            fill_renderers.append(price_plot.scatter(
                "bar_index",
                "price",
                source=ColumnDataSource(buys),
                marker="triangle",
                size=11,
                color=MARKET_UP,
                line_color="white",
                legend_label="Buy",
            ))
        if not sells.empty:
            fill_renderers.append(price_plot.scatter(
                "bar_index",
                "price",
                source=ColumnDataSource(sells),
                marker="inverted_triangle",
                size=11,
                color=MARKET_DOWN,
                line_color="white",
                legend_label="Sell",
            ))
        if fill_renderers:
            _add_passive_hover(
                price_plot,
                HoverTool(
                    renderers=fill_renderers,
                    formatters={"@date": "datetime"},
                    tooltips=[
                        ("Date", "@date{%F %T}"),
                        ("Side", "@side"),
                        ("Price", "@price{0,0.####}"),
                        ("Size", "@size{0,0.####}"),
                    ],
                ),
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
        volume_plot.yaxis.ticker.desired_num_ticks = 5
        _add_line_hover(
            volume_plot,
            [renderer],
            [("Date", "@date{%F %T}"), ("Volume", "@volume{0.00 a}")],
        )
        plots.append(volume_plot)

    for plot in plots[:-1]:
        plot.xaxis.visible = False

    # Map bar_index ticks to date labels on the bottom (visible) x-axis.
    if plots and "date" in frame.columns:
        n_ticks = min(8, len(frame))
        step = max(1, len(frame) // n_ticks)
        tick_indices = list(range(0, len(frame), step))
        last_idx = len(frame) - 1
        # Only add the last bar if it's far enough from the previous tick to avoid overlap.
        if last_idx not in tick_indices and (last_idx - tick_indices[-1]) >= step // 2:
            tick_indices.append(last_idx)
        date_labels = {
            int(frame.loc[i, "bar_index"]): str(pd.to_datetime(frame.loc[i, "date"]).strftime("%Y-%m-%d"))
            for i in tick_indices
        }
        bottom_plot = plots[-1]
        bottom_plot.xaxis.ticker = list(date_labels.keys())
        bottom_plot.xaxis.major_label_overrides = date_labels
        bottom_plot.xaxis.major_label_orientation = 0.6

    for plot in plots:
        _style_market_section(plot)
    _style_market_legend(equity_plot, compact=True)
    _style_market_legend(pl_plot if not trades_frame.empty else None)
    _style_market_legend(price_plot, compact=True, large_glyphs=True)

    return gridplot(
        plots,
        ncols=1,
        sizing_mode="stretch_width",
        toolbar_location="right",
        merge_tools=True,
    )


def _portfolio_market_replay(
    market_data: Mapping[str, pd.DataFrame],
    fills: pd.DataFrame | None,
    equity: pd.Series | None,
    positions: pd.DataFrame | None,
):
    """Return a portfolio-first replay grid for multi-asset reports."""
    asset_frame = _portfolio_asset_frame(market_data)
    if asset_frame.empty:
        return None

    base_frame = _portfolio_base_frame(asset_frame)
    if base_frame.empty:
        return None

    fills_frame = _fills_plot_frame(fills) if _fills_have_plot_columns(fills) else pd.DataFrame()
    fills_frame = _attach_portfolio_bar_index(fills_frame, base_frame, asset_frame)
    trades_frame = _trade_segments(fills_frame, base_frame)

    x_pad = max((len(base_frame) - 1) / 20, 1.0)
    x_range = Range1d(
        start=float(base_frame["bar_index"].iloc[0] - x_pad),
        end=float(base_frame["bar_index"].iloc[-1] + x_pad),
        min_interval=10,
    )
    plots = []

    if equity is not None and not equity.empty:
        equity_plot = _equity_replay_plot(equity, base_frame, x_range)
        if equity_plot is not None:
            plots.append(equity_plot)
    else:
        equity_plot = None

    allocation_plot = _allocation_replay_plot(
        positions,
        base_frame,
        x_range,
        fills_frame,
        asset_frame,
    )
    if allocation_plot is not None:
        plots.append(allocation_plot)

    if not trades_frame.empty:
        pl_plot = _profit_loss_replay_plot(trades_frame, x_range)
        plots.append(pl_plot)
    else:
        pl_plot = None

    assets_plot = _assets_replay_plot(asset_frame, fills_frame, x_range)
    plots.append(assets_plot)

    _apply_replay_date_axis(plots, base_frame)
    for plot in plots:
        _style_market_section(plot)
    _style_market_legend(equity_plot, compact=True)
    _style_market_legend(allocation_plot, compact=True)
    _style_market_legend(pl_plot)
    _style_market_legend(assets_plot, compact=True, large_glyphs=True)

    return gridplot(
        plots,
        ncols=1,
        sizing_mode="stretch_width",
        toolbar_location="right",
        merge_tools=True,
    )


def _equity_replay_plot(equity: pd.Series, frame: pd.DataFrame, x_range):
    """Return the replay equity panel."""
    equity_plot = _market_section("Equity", height=150, x_range=x_range)
    equity_frame = _equity_replay_frame(equity, frame)
    if equity_frame.empty:
        return None
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
    max_dd = equity_frame["drawdown"].idxmin()
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
    return equity_plot


def _allocation_replay_plot(
    positions: pd.DataFrame | None,
    frame: pd.DataFrame,
    x_range,
    fills: pd.DataFrame | None = None,
    asset_frame: pd.DataFrame | None = None,
):
    """Return a stacked allocation panel when portfolio positions are available."""
    allocation = _allocation_frame(positions, frame)
    if allocation.empty and fills is not None and asset_frame is not None:
        allocation = _allocation_frame_from_fills(fills, asset_frame, frame)
    if allocation.empty:
        return None
    plot = _market_section("Allocation", height=120, x_range=x_range)
    stackers = [
        column
        for column in allocation.columns
        if column not in {"date", "bar_index"} and allocation[column].abs().sum() > 0
    ]
    if not stackers:
        return None
    source = ColumnDataSource(allocation)
    colors = _palette(len(stackers))
    renderers = plot.varea_stack(
        stackers,
        x="bar_index",
        color=colors,
        alpha=0.78,
        source=source,
        legend_label=stackers,
    )
    plot.yaxis.axis_label = "Allocation"
    plot.yaxis.formatter = NumeralTickFormatter(format="0%")
    plot.y_range.start = 0.0
    plot.y_range.end = max(1.0, float(allocation[stackers].sum(axis=1).max()) * 1.05)
    _add_line_hover(
        plot,
        renderers,
        [("Date", "@date{%F %T}")],
    )
    return plot


def _profit_loss_replay_plot(trades_frame: pd.DataFrame, x_range):
    """Return the replay trade P/L panel."""
    plot = _market_section("Profit / Loss", height=105, x_range=x_range)
    plot.add_layout(
        Span(location=0, dimension="width", line_color=MARKET_MUTED, line_dash="dashed")
    )
    trade_source = ColumnDataSource(trades_frame)
    win = trades_frame[trades_frame["return_pct"] >= 0]
    loss = trades_frame[trades_frame["return_pct"] < 0]
    if not win.empty:
        plot.scatter(
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
        plot.scatter(
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
    plot.y_range.start = min(min_ret - marker_pad, -0.002)
    plot.y_range.end = max(max_ret + marker_pad, 0.002)
    hidden = plot.scatter(
        "exit_bar",
        "return_pct",
        source=trade_source,
        marker="circle",
        size=1,
        alpha=0.0,
    )
    plot.yaxis.axis_label = "Profit / Loss"
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _add_line_hover(
        plot,
        [hidden],
        [
            ("Exit", "@exit_datetime{%F %T}"),
            ("Size", "@size{0,0.####}"),
            ("P/L", "@return_pct{+0.[000]%}"),
        ],
        vline=False,
    )
    return plot


def _assets_replay_plot(asset_frame: pd.DataFrame, fills_frame: pd.DataFrame, x_range):
    """Return normalized asset price lines with buy/sell markers."""
    plot = _market_section("Normalized Assets / Trades", height=410, x_range=x_range)
    symbols = list(dict.fromkeys(asset_frame["symbol"].astype(str)))
    visible_symbols = set(
        _default_visible_portfolio_symbols(
            symbols,
            fills_frame,
            limit=PORTFOLIO_VISIBLE_ASSET_LIMIT,
        )
    )
    colors = _palette(len(symbols))
    for index, symbol in enumerate(symbols):
        symbol_frame = asset_frame[asset_frame["symbol"].eq(symbol)]
        renderer = plot.line(
            "bar_index",
            "normalized_close",
            source=ColumnDataSource(symbol_frame),
            line_width=1.7,
            color=colors[index % len(colors)],
            legend_label=symbol,
        )
        renderer.visible = symbol in visible_symbols
        _add_line_hover(
            plot,
            [renderer],
            [
                ("Date", "@date{%F %T}"),
                ("Symbol", "@symbol"),
                ("Normalized", "@normalized_close{0,0.[00]}"),
                ("Close", "@close{0,0.####}"),
            ],
        )

    if not fills_frame.empty:
        fills_frame = fills_frame[fills_frame["symbol"].astype(str).isin(visible_symbols)]
        buys = fills_frame[fills_frame["side"].str.lower().eq("buy")]
        sells = fills_frame[fills_frame["side"].str.lower().eq("sell")]
        fill_renderers = []
        if not buys.empty:
            fill_renderers.append(
                plot.scatter(
                    "bar_index",
                    "normalized_price",
                    source=ColumnDataSource(buys),
                    marker="triangle",
                    size=8,
                    color=MARKET_UP,
                    line_color="white",
                    fill_alpha=0.58,
                    line_alpha=0.82,
                    legend_label="Buy",
                )
            )
        if not sells.empty:
            fill_renderers.append(
                plot.scatter(
                    "bar_index",
                    "normalized_price",
                    source=ColumnDataSource(sells),
                    marker="inverted_triangle",
                    size=8,
                    color=MARKET_DOWN,
                    line_color="white",
                    fill_alpha=0.58,
                    line_alpha=0.82,
                    legend_label="Sell",
                )
            )
        if fill_renderers:
            _add_passive_hover(
                plot,
                HoverTool(
                    renderers=fill_renderers,
                    formatters={"@date": "datetime"},
                    tooltips=[
                        ("Date", "@date{%F %T}"),
                        ("Symbol", "@symbol"),
                        ("Side", "@side"),
                        ("Price", "@price{0,0.####}"),
                        ("Size", "@size{0,0.####}"),
                    ],
                ),
            )
    plot.yaxis.axis_label = "Normalized Price"
    return plot


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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
        height=220,
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
    _make_static_chart(plot)
    return plot


def monthly_returns_distribution(returns: pd.Series):
    """Return a monthly returns distribution chart."""
    monthly = ((1.0 + pd.Series(returns).dropna()).resample("ME").prod() - 1.0).dropna()
    plot = figure(
        title="Monthly Returns Distribution",
        height=220,
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
        height=220,
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


def rolling_returns(returns: pd.Series):
    """Return a cumulative rolling returns chart."""
    curve = (1.0 + pd.Series(returns).dropna()).cumprod() - 1.0
    frame = _plot_frame(curve, "rolling_returns")
    plot = figure(
        title="Rolling Returns",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_returns"], line_width=2, color=MARKET_BLUE)
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot


def rolling_volatility(returns: pd.Series, window: int = 126, periods: int = 252):
    """Return annualized rolling volatility."""
    values = pd.Series(returns).dropna().rolling(window, min_periods=2).std() * periods ** 0.5
    frame = _plot_frame(values, "rolling_volatility").dropna()
    plot = figure(
        title="Rolling Volatility",
        x_axis_type="datetime",
        height=220,
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
        height=220,
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
    plot = figure(title="Holdings", x_axis_type="datetime", height=220, sizing_mode="stretch_width")
    if not frame.empty:
        plot.line(frame["date"], frame["holdings"], line_width=2, color=MARKET_BLUE)
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
        height=220,
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


def gross_leverage(positions: pd.DataFrame):
    """Return gross leverage computed from position values."""
    wide = _positions_wide(positions)
    assets = wide.drop(columns=["cash"], errors="ignore")
    gross = assets.abs().sum(axis=1)
    equity = wide.sum(axis=1).replace(0, pd.NA)
    leverage = (gross / equity.abs()).astype("float64").fillna(0.0)
    frame = _plot_frame(leverage, "gross_leverage")
    plot = figure(
        title="Gross Leverage",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["gross_leverage"], line_width=2, color="#8c564b")
    _make_static_chart(plot)
    return plot


def position_concentration(positions: pd.DataFrame):
    """Return max and median absolute position concentration."""
    wide = _positions_wide(positions)
    assets = wide.drop(columns=["cash"], errors="ignore")
    totals = assets.abs().sum(axis=1).replace(0, pd.NA)
    weights = assets.abs().div(totals, axis=0).fillna(0.0)
    frame = pd.DataFrame(
        {
            "max": weights.max(axis=1),
            "median": weights.median(axis=1),
        }
    )
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    frame["date"] = _normalize_dates(frame["date"])
    plot = figure(
        title="Position Concentration",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["max"], line_width=2, color=MARKET_BLUE, legend_label="Max")
        plot.line(
            frame["date"],
            frame["median"],
            line_width=2,
            color="#ff7f0e",
            legend_label="Median",
        )
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot


def turnover(transactions: pd.DataFrame, positions: pd.DataFrame | None = None):
    """Return daily turnover from transaction notional."""
    frame = _transactions_frame(transactions)
    plot = figure(title="Turnover", x_axis_type="datetime", height=220, sizing_mode="stretch_width")
    if frame.empty:
        _make_static_chart(plot)
        return plot
    daily = frame.groupby(frame["date"].dt.floor("D"))["notional"].sum()
    if positions is not None and not positions.empty:
        wide = _positions_wide(positions)
        equity = wide.sum(axis=1).abs().replace(0, pd.NA)
        daily = daily / equity.reindex(daily.index, method="nearest").ffill()
    daily = daily.astype("float64").fillna(0.0)
    data = _plot_frame(daily, "turnover")
    plot.line(data["date"], data["turnover"], line_width=2, color=MARKET_BLUE)
    _make_static_chart(plot)
    return plot


def daily_volume(transactions: pd.DataFrame):
    """Return daily traded share or contract volume."""
    frame = _transactions_frame(transactions)
    plot = figure(
        title="Daily Volume",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        daily = frame.groupby(frame["date"].dt.floor("D"))["size"].sum().abs()
        data = _plot_frame(daily, "volume")
        plot.vbar(
            data["date"],
            width=20 * 60 * 60 * 1000,
            top=data["volume"],
            color=MARKET_BLUE,
            alpha=0.70,
        )
    _make_static_chart(plot)
    return plot


def transaction_time_histogram(transactions: pd.DataFrame):
    """Return transaction count histogram by hour of day."""
    frame = _transactions_frame(transactions)
    plot = figure(
        title="Transaction Time Histogram",
        x_range=[str(hour) for hour in range(24)],
        height=220,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        counts = frame["date"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        data = pd.DataFrame({"hour": counts.index.astype(str), "count": counts.to_numpy()})
        plot.vbar("hour", 0.8, "count", source=ColumnDataSource(data), color=MARKET_BLUE)
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
    return plot


def factor_quantile_returns_bar(stats: pd.DataFrame):
    """Return mean forward return by factor quantile."""
    frame = pd.DataFrame(stats).copy()
    if "mean" not in frame:
        frame["mean"] = pd.Series(dtype="float64")
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "quantile"})
    frame["quantile"] = frame["quantile"].astype(str)
    frame["color"] = [MARKET_UP if value >= 0 else MARKET_DOWN for value in frame["mean"]]
    plot = figure(
        title="Mean Return by Quantile",
        x_range=list(frame["quantile"]),
        height=220,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        plot.vbar("quantile", 0.7, "mean", source=ColumnDataSource(frame), color="color")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot


def factor_quantile_returns_violin(forward_returns: pd.DataFrame):
    """Return Alphalens-style forward return distribution by factor quantile."""
    frame = _quantile_forward_frame(forward_returns)
    quantiles = (
        [str(value) for value in sorted(frame["quantile"].unique())]
        if not frame.empty
        else []
    )
    plot = figure(
        title="Quantile Returns Violin",
        x_range=(-0.5, max(0.5, len(quantiles) - 0.5)),
        height=260,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        for quantile in quantiles:
            quantile_mask = frame["quantile"].astype(str).eq(quantile)
            values = frame.loc[quantile_mask, "forward_return"].dropna()
            if values.empty:
                continue
            bins = min(18, max(4, len(values)))
            counts = pd.cut(values, bins=bins).value_counts(sort=False)
            if counts.empty or counts.max() == 0:
                continue
            centers = [(interval.left + interval.right) / 2 for interval in counts.index]
            widths = [0.42 * count / counts.max() for count in counts.to_numpy()]
            x_center = quantiles.index(quantile)
            xs = [x_center + width for width in widths] + [
                x_center - width for width in reversed(widths)
            ]
            ys = centers + list(reversed(centers))
            plot.patch(xs, ys, fill_color=MARKET_BLUE, fill_alpha=0.28, line_color=MARKET_BLUE)
            plot.scatter([x_center], [float(values.median())], size=7, color="#ff7f0e")
        plot.xaxis.ticker = list(range(len(quantiles)))
        plot.xaxis.major_label_overrides = {
            index: f"Q{value}" for index, value in enumerate(quantiles)
        }
        plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot


def factor_quantile_spread(spread: pd.Series):
    """Return top-minus-bottom quantile spread time series."""
    frame = _plot_frame(pd.Series(spread).dropna(), "spread")
    plot = figure(
        title="Quantile Spread",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.add_layout(
            Span(location=0, dimension="width", line_color=MARKET_MUTED, line_dash="dashed")
        )
        plot.line(frame["date"], frame["spread"], line_width=2, color=MARKET_BLUE)
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
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
    _make_static_chart(plot)
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
    _make_static_chart(plot)
    return plot


def factor_ic_histogram(ic: pd.Series):
    """Return an IC distribution histogram."""
    values = pd.Series(ic).dropna()
    plot = figure(title="IC Histogram", height=220, sizing_mode="stretch_width")
    if not values.empty:
        buckets = min(20, max(1, len(values)))
        counts = pd.cut(values, bins=buckets).value_counts(sort=False)
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
        mean = float(values.mean())
        plot.line([mean, mean], [0, max(1, int(counts.max()))], color="#ff7f0e", line_width=2)
    _make_static_chart(plot)
    return plot


def factor_ic_qq(ic: pd.Series):
    """Return an IC normal QQ chart."""
    values = pd.Series(ic).dropna().sort_values().reset_index(drop=True)
    plot = figure(title="IC QQ", height=220, sizing_mode="stretch_width")
    if not values.empty:
        dist = NormalDist()
        n = len(values)
        theoretical = [dist.inv_cdf((index + 0.5) / n) for index in range(n)]
        frame = pd.DataFrame({"theoretical": theoretical, "sample": values})
        plot.scatter(
            "theoretical",
            "sample",
            source=ColumnDataSource(frame),
            size=7,
            color=MARKET_BLUE,
            alpha=0.75,
        )
        low = min(float(frame["theoretical"].min()), float(frame["sample"].min()))
        high = max(float(frame["theoretical"].max()), float(frame["sample"].max()))
        plot.line([low, high], [low, high], line_color=MARKET_MUTED, line_dash="dashed")
    _make_static_chart(plot)
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
    _make_static_chart(plot)
    return plot


def factor_monthly_ic_heatmap(monthly_ic: pd.DataFrame):
    """Return a monthly IC heatmap figure."""
    values = monthly_ic.dropna(axis=0, how="all").dropna(axis=1, how="all")
    months = [int(column) for column in values.columns]
    years = [str(year) for year in values.index]
    data = {"month": [], "year": [], "ic": []}
    for year in values.index:
        for month in months:
            data["month"].append(str(month))
            data["year"].append(str(year))
            data["ic"].append(values.loc[year, month])
    plot = figure(
        title="Monthly IC Heatmap",
        x_range=[str(month) for month in months],
        y_range=years,
        height=240,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "ic",
        palette=["#d65a5a", "#f7f7f7", "#2ca36c"],
        low=-1.0,
        high=1.0,
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
    _make_static_chart(plot)
    return plot


def quantile_counts(counts: pd.DataFrame):
    """Return per-date factor quantile membership counts."""
    frame = pd.DataFrame(counts).reset_index().rename(
        columns={counts.index.name or "index": "date"}
    )
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Quantile Counts",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, column in enumerate(counts.columns):
        plot.line(
            frame["date"],
            frame[column],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=f"Q{column}",
        )
    _make_static_chart(plot)
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
    _make_static_chart(plot)
    return plot


def factor_events_distribution(events: pd.DataFrame | pd.MultiIndex | pd.Series):
    """Return event count distribution over time."""
    frame = _events_frame(events)
    plot = figure(
        title="Events Distribution",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        counts = frame.groupby(frame["date"].dt.floor("D")).size()
        data = _plot_frame(counts.rename("count"), "count")
        plot.vbar(
            data["date"],
            width=20 * 60 * 60 * 1000,
            top=data["count"],
            color=MARKET_BLUE,
            alpha=0.70,
        )
    _make_static_chart(plot)
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


def _normalize_dates(values: pd.Series) -> pd.Series:
    """Return timezone-naive datetimes for Bokeh."""
    dates = pd.to_datetime(values, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates


def _positions_wide(positions: pd.DataFrame) -> pd.DataFrame:
    """Return numeric date x symbol position values."""
    frame = pd.DataFrame(positions).copy()
    if frame.empty:
        return pd.DataFrame()
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
    symbol_col = lowered.get("symbol") or lowered.get("ticker") or lowered.get("asset")
    value_col = lowered.get("value") or lowered.get("market_value") or lowered.get("position_value")
    if date_col is not None and symbol_col is not None and value_col is not None:
        frame["__date"] = _normalize_dates(frame[date_col])
        frame["__value"] = pd.to_numeric(frame[value_col], errors="coerce")
        wide = frame.pivot_table(
            index="__date",
            columns=symbol_col,
            values="__value",
            aggfunc="sum",
        )
        return wide.sort_index().fillna(0.0)
    if date_col is not None:
        frame = frame.set_index(_normalize_dates(frame[date_col])).drop(columns=[date_col])
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = _normalize_dates(pd.Series(frame.index))
    numeric = frame.select_dtypes(include="number").copy()
    return numeric.sort_index().fillna(0.0)


def _transactions_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return normalized transactions with date, signed size, price, and notional."""
    frame = pd.DataFrame(transactions).copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "size", "price", "notional"])
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("datetime") or lowered.get("date") or lowered.get("timestamp")
    size_col = lowered.get("size") or lowered.get("qty") or lowered.get("quantity")
    price_col = lowered.get("price") or lowered.get("fill_price")
    amount_col = lowered.get("amount") or lowered.get("notional") or lowered.get("value")
    if date_col is None:
        return pd.DataFrame(columns=["date", "size", "price", "notional"])
    result = pd.DataFrame({"date": _normalize_dates(frame[date_col])})
    if size_col is not None:
        result["size"] = pd.to_numeric(frame[size_col], errors="coerce").fillna(0.0)
    else:
        result["size"] = 0.0
    if price_col is not None:
        result["price"] = pd.to_numeric(frame[price_col], errors="coerce").fillna(0.0)
    else:
        result["price"] = 0.0
    if amount_col is not None:
        result["notional"] = pd.to_numeric(frame[amount_col], errors="coerce").abs().fillna(0.0)
    else:
        result["notional"] = (result["size"].abs() * result["price"].abs()).fillna(0.0)
    return result.dropna(subset=["date"])


def _quantile_forward_frame(forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized quantile/forward_return frame."""
    frame = pd.DataFrame(forward_returns).copy()
    if frame.empty:
        return pd.DataFrame(columns=["quantile", "forward_return"])
    lowered = {str(column).lower(): column for column in frame.columns}
    quantile_col = lowered.get("quantile") or lowered.get("factor_quantile")
    return_col = lowered.get("forward_return") or lowered.get("return") or lowered.get("returns")
    if quantile_col is not None and return_col is not None:
        result = pd.DataFrame(
            {
                "quantile": frame[quantile_col],
                "forward_return": pd.to_numeric(frame[return_col], errors="coerce"),
            }
        )
        return result.dropna(subset=["quantile", "forward_return"])
    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels >= 2 and frame.shape[1] >= 1:
        result = frame.iloc[:, 0].reset_index()
        result.columns = [*result.columns[:-1], "forward_return"]
        quantile_name = "factor_quantile" if "factor_quantile" in result else result.columns[-2]
        return result.rename(columns={quantile_name: "quantile"})[["quantile", "forward_return"]]
    return pd.DataFrame(columns=["quantile", "forward_return"])


def _events_frame(events: pd.DataFrame | pd.MultiIndex | pd.Series) -> pd.DataFrame:
    """Return event rows with a normalized date column."""
    if isinstance(events, pd.MultiIndex):
        dates = events.get_level_values(0)
        return pd.DataFrame({"date": _normalize_dates(pd.Series(dates))})
    if isinstance(events, pd.Series):
        if isinstance(events.index, pd.MultiIndex):
            dates = events.index.get_level_values(0)
        else:
            dates = events.index
        return pd.DataFrame({"date": _normalize_dates(pd.Series(dates))})
    frame = pd.DataFrame(events).copy()
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    return pd.DataFrame({"date": _normalize_dates(frame[date_col])}).dropna()


def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame


def _portfolio_asset_frame(market_data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return normalized long-form prices for a multi-asset replay."""
    rows = []
    for symbol, data in market_data.items():
        frame = _market_frame(pd.DataFrame(data))
        if frame.empty:
            continue
        frame = frame[["date", "close"]].copy()
        frame["symbol"] = str(symbol)
        first_close = float(frame["close"].iloc[0])
        if not first_close:
            continue
        frame["normalized_close"] = frame["close"] / first_close * 100.0
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    dates = pd.Series(pd.Index(result["date"].dropna().unique()).sort_values())
    bar_map = {date: float(index) for index, date in enumerate(dates)}
    result["bar_index"] = result["date"].map(bar_map)
    return result.dropna(subset=["bar_index", "normalized_close"])


def _portfolio_base_frame(asset_frame: pd.DataFrame) -> pd.DataFrame:
    """Return an equal-weight portfolio benchmark frame for replay alignment."""
    if asset_frame.empty:
        return pd.DataFrame()
    base = (
        asset_frame.pivot_table(
            index="date",
            columns="symbol",
            values="normalized_close",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
    )
    if base.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"date": base.index, "close": base.mean(axis=1).to_numpy()})
    frame["bar_index"] = np.arange(len(frame), dtype=float)
    return frame


def _allocation_frame(positions: pd.DataFrame | None, frame: pd.DataFrame) -> pd.DataFrame:
    """Return replay-aligned positive allocation weights."""
    if positions is None:
        return pd.DataFrame()
    positions_frame = pd.DataFrame(positions).copy()
    if positions_frame.empty or not {"date", "symbol", "value"}.issubset(positions_frame.columns):
        return pd.DataFrame()
    exposure = positions_frame.pivot_table(
        index="date",
        columns="symbol",
        values="value",
        aggfunc="sum",
    ).sort_index()
    if exposure.empty:
        return pd.DataFrame()
    exposure.index = pd.to_datetime(exposure.index, errors="coerce")
    if isinstance(exposure.index.dtype, pd.DatetimeTZDtype):
        exposure.index = exposure.index.tz_convert("UTC").tz_localize(None)
    exposure = exposure.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    totals = exposure.sum(axis=1)
    weights = exposure.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    weights = weights.reindex(dates).ffill().fillna(0.0)
    weights.index = dates
    keep = list(weights.abs().mean().sort_values(ascending=False).head(8).index)
    compact = weights[keep].copy()
    other_columns = [column for column in weights.columns if column not in keep]
    if other_columns:
        compact["Others"] = weights[other_columns].sum(axis=1)
    cash = (1.0 - compact.sum(axis=1)).clip(lower=0.0)
    if cash.max() > 1e-9:
        compact["Cash"] = cash
    result = compact.reset_index().rename(columns={"date": "date"})
    result["bar_index"] = frame["bar_index"].to_numpy()
    return result


def _allocation_frame_from_fills(
    fills: pd.DataFrame,
    asset_frame: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return replay-aligned allocation weights reconstructed from fills."""
    if fills.empty or not {"symbol", "date", "size"}.issubset(fills.columns):
        return pd.DataFrame()
    close = (
        asset_frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    if close.empty:
        return pd.DataFrame()
    fills_frame = fills.copy()
    fills_frame["symbol"] = fills_frame["symbol"].astype(str)
    fills_frame["signed_size"] = pd.to_numeric(fills_frame["size"], errors="coerce").fillna(0.0)
    if "side" in fills_frame:
        side = fills_frame["side"].astype(str).str.lower()
        abs_size = fills_frame["signed_size"].abs()
        fills_frame.loc[side.eq("buy"), "signed_size"] = abs_size[side.eq("buy")]
        fills_frame.loc[side.eq("sell"), "signed_size"] = -abs_size[side.eq("sell")]
    changes = fills_frame.pivot_table(
        index="date",
        columns="symbol",
        values="signed_size",
        aggfunc="sum",
    )
    changes = changes.reindex(close.index).fillna(0.0)
    units = changes.cumsum().reindex(close.index).ffill().fillna(0.0)
    values = units.mul(close, fill_value=0.0).clip(lower=0.0)
    totals = values.sum(axis=1)
    weights = values.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    weights = weights.reindex(dates).ffill().fillna(0.0)
    weights.index = dates
    keep = list(weights.abs().mean().sort_values(ascending=False).head(8).index)
    result = weights[keep].reset_index().rename(columns={"date": "date"})
    result["bar_index"] = frame["bar_index"].to_numpy()
    return result


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
    frame = frame.dropna(subset=["date", "price"])
    if frame.empty:
        return frame
    price_numeric = pd.to_numeric(frame["price"], errors="coerce")
    return frame[np.isfinite(price_numeric) & (price_numeric > 0)]


def _fills_have_plot_columns(fills: pd.DataFrame | None) -> bool:
    """Return True when fills can be projected onto the market replay chart."""
    return fills is not None and not fills.empty and {"datetime", "price", "side"}.issubset(
        fills.columns
    )


def _attach_bar_index(fills: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest market bar index to each fill."""
    if fills.empty:
        return fills
    bars = frame[["date", "bar_index"]].copy()
    fills_sorted = fills.copy()
    
    bars["date"] = bars["date"].astype("datetime64[ns]")
    fills_sorted["date"] = fills_sorted["date"].astype("datetime64[ns]")
    
    projected = pd.merge_asof(
        fills_sorted.sort_values("date"),
        bars.sort_values("date"),
        on="date",
        direction="nearest",
    )
    return projected.dropna(subset=["bar_index"])


def _attach_portfolio_bar_index(
    fills: pd.DataFrame,
    frame: pd.DataFrame,
    asset_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach replay indices and normalized prices to portfolio fill rows."""
    if fills.empty:
        return fills
    projected = _attach_bar_index(fills, frame)
    if projected.empty:
        return projected
    first_close = (
        asset_frame.sort_values("date")
        .groupby("symbol", sort=False)["close"]
        .first()
        .to_dict()
    )
    projected["symbol"] = projected.get("symbol", projected.get("data", "")).astype(str)
    projected["data"] = projected["symbol"]
    projected["normalized_price"] = [
        float(price) / float(first_close.get(symbol, np.nan)) * 100.0
        for symbol, price in zip(projected["symbol"], projected["price"], strict=True)
    ]
    return projected.dropna(subset=["normalized_price"])


def _apply_replay_date_axis(plots: list[Any], frame: pd.DataFrame) -> None:
    """Apply sparse date labels to the bottom replay plot."""
    if not plots or "date" not in frame.columns:
        return
    n_ticks = min(8, len(frame))
    step = max(1, len(frame) // n_ticks)
    tick_indices = list(range(0, len(frame), step))
    last_idx = len(frame) - 1
    if last_idx not in tick_indices and (last_idx - tick_indices[-1]) >= step // 2:
        tick_indices.append(last_idx)
    date_labels = {
        int(frame.loc[i, "bar_index"]): str(
            pd.to_datetime(frame.loc[i, "date"]).strftime("%Y-%m-%d")
        )
        for i in tick_indices
    }
    bottom_plot = plots[-1]
    bottom_plot.xaxis.ticker = list(date_labels.keys())
    bottom_plot.xaxis.major_label_overrides = date_labels
    bottom_plot.xaxis.major_label_orientation = 0.6


def _equity_replay_frame(equity: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """Return equity aligned to replay bar indices with derived replay columns."""
    equity_frame = _plot_frame(pd.Series(equity).dropna(), "equity")
    if equity_frame.empty:
        return pd.DataFrame()
    bars = frame[["date", "bar_index", "close"]].copy()
    
    bars["date"] = bars["date"].astype("datetime64[ns]")
    equity_frame["date"] = equity_frame["date"].astype("datetime64[ns]")
    
    projected = pd.merge_asof(
        bars.sort_values("date"),
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
    """Build approximate entry-exit trade segments from fill rows."""
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
        data_name = str(fill.get("data", "") or fill.get("symbol", "") or "__default__")
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
    """Return a linked replay figure."""
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


def _palette(count: int) -> list[str]:
    """Return a stable report palette."""
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#7f7f7f",
        "#bcbd22",
        "#e377c2",
    ]
    return [colors[index % len(colors)] for index in range(max(count, 1))]


def _default_visible_portfolio_symbols(
    symbols: list[str],
    fills: pd.DataFrame,
    *,
    limit: int,
) -> list[str]:
    """Return the symbols that should be visible in dense portfolio replay charts."""
    if len(symbols) <= limit:
        return symbols

    symbol_rank = {symbol: index for index, symbol in enumerate(symbols)}
    if fills is None or fills.empty or "symbol" not in fills.columns:
        return symbols[:limit]

    counts = fills["symbol"].astype(str).value_counts()
    ranked = sorted(
        symbols,
        key=lambda symbol: (-int(counts.get(symbol, 0)), symbol_rank[symbol]),
    )
    return ranked[:limit]


def _style_market_section(plot) -> None:
    """Apply common Bokeh market Bokeh styling."""
    plot.min_border_left = 50
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
        legend.padding = 5 if compact else 6
        legend.spacing = 2 if compact else 1
        legend.margin = 4
        if compact and large_glyphs:
            legend.glyph_width = 32
            legend.glyph_height = 22
        else:
            legend.glyph_width = 18 if compact else (34 if large_glyphs else 28)
            legend.glyph_height = 12 if compact else (22 if large_glyphs else 18)
        legend.label_text_color = "#33424f"
        legend.label_text_font_size = "8pt" if compact else "9pt"
        legend.click_policy = "hide"


def _make_static_chart(plot) -> None:
    """Render report support charts without visible interactive controls."""
    plot.toolbar_location = None
    plot.toolbar.tools = []
    plot.toolbar.logo = None
    plot.toolbar.active_drag = None
    plot.toolbar.active_scroll = None
    plot.toolbar.active_inspect = None


def _add_line_hover(plot, renderers, tooltips, *, vline: bool = True) -> None:
    """Attach a shared hover tool."""
    _add_passive_hover(
        plot,
        HoverTool(
            point_policy="follow_mouse",
            renderers=list(renderers),
            formatters={"@date": "datetime", "@exit_datetime": "datetime"},
            tooltips=list(tooltips),
            mode="vline" if vline else "mouse",
        ),
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
    _add_passive_hover(
        plot,
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
        ),
    )


def _add_close_hover(plot, renderers) -> None:
    """Attach close-price hover tool to the main market replay plot."""
    _add_passive_hover(
        plot,
        HoverTool(
            renderers=list(renderers),
            mode="vline",
            formatters={"@date": "datetime"},
            tooltips=[
                ("Date", "@date{%F %T}"),
                ("Close", "@close{0,0.0000}"),
            ],
        ),
    )


def _add_passive_hover(plot, hover: HoverTool) -> None:
    """Attach hover behavior without adding repeated hover buttons to the toolbar."""
    plot.add_tools(hover)
    if hover in plot.toolbar.tools:
        plot.toolbar.tools.remove(hover)
