"""Shared Bokeh styling, layout, hover, and frame helpers for report charts."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bokeh.events import MouseLeave, MouseMove
from bokeh.models import (
    BoxAnnotation,
    CrosshairTool,
    CustomJS,
    HoverTool,
    Legend,
    Spacer,
    Span,
)
from bokeh.plotting import figure

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
MARKET_TITLE = "#2f3b52"
PORTFOLIO_OTHERS = "#a98578"
PORTFOLIO_CASH = "#b8c0c7"
PORTFOLIO_VISIBLE_ASSET_LIMIT = 8
DEFAULT_REPORT_TIMEZONE = ZoneInfo("UTC")
MARKET_LEGEND_MARGIN = 2
MARKET_LEGEND_FRAME_GAP = 8
MARKET_LEGEND_HEIGHT_COMPENSATION = 24
MARKET_LEGEND_COMPENSATED_TAG = "market_legend_height_compensated"
MARKET_LEGEND_FRAME_GAP_TAG = "market_legend_frame_gap"
ALLOCATION_LEGEND_VISIBLE_LIMIT = 15
MARKET_REPLAY_TITLE_STANDOFF = 10


def _add_drawdown_markers(plot, equity: pd.Series, drawdowns: pd.DataFrame) -> None:
    """Add peak and valley markers for drawdown episodes."""
    for _, row in drawdowns.iterrows():
        if "peak" not in row or "valley" not in row:
            continue
        dates = _normalize_dates(pd.Series([row["peak"], row["valley"]])).dropna()
        if len(dates) < 2:
            continue
        left, right = dates.iloc[0], dates.iloc[1]
        if right < left:
            left, right = right, left
        plot.add_layout(
            BoxAnnotation(
                left=left,
                right=right,
                fill_color=MARKET_DOWN,
                fill_alpha=0.08,
                line_alpha=0.0,
                level="underlay",
                name="max_drawdown_period",
            )
        )
    for series_column, name, color, marker in [
        ("peak", "drawdown_peak", "#2ca02c", "triangle"),
        ("valley", "drawdown_valley", "#d62728", "inverted_triangle"),
    ]:
        if series_column not in drawdowns:
            continue
        dates = pd.to_datetime(drawdowns[series_column]).dropna()
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
            legend_label="Peak" if series_column == "peak" else "Valley",
            name=name,
        )
    if plot.legend:
        plot.legend.location = "top_left"

def _style_equity_curve_legend(plot) -> None:
    """Use a compact in-chart legend for the overview equity curve."""
    if not plot.legend:
        return
    for legend in plot.legend:
        legend.location = "top_left"
        legend.label_text_font_size = "9px"
        legend.label_text_color = "#33424f"
        legend.glyph_width = 12
        legend.glyph_height = 8
        legend.padding = 2
        legend.spacing = 4
        legend.margin = 2
        legend.border_line_alpha = 0.25
        legend.background_fill_alpha = 0.58

def _normalize_dates(values: pd.Series) -> pd.Series:
    """Return timezone-naive datetimes for Bokeh."""
    dates = pd.to_datetime(values, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates

def _symbol_display_map(symbols: list[str]) -> dict[str, str]:
    """Return compact display labels while preserving uniqueness."""
    compact = {symbol: symbol.split(":")[-1] for symbol in symbols}
    counts = pd.Series(list(compact.values())).value_counts()
    return {
        symbol: label if counts[label] == 1 else symbol
        for symbol, label in compact.items()
    }

def _coerce_timezone(timezone: str | ZoneInfo) -> ZoneInfo:
    """Return a ZoneInfo instance for a report timezone value."""
    if isinstance(timezone, ZoneInfo):
        return timezone
    return ZoneInfo(str(timezone))

def _timezone_label(timezone: ZoneInfo) -> str:
    """Return a compact axis label for a report timezone."""
    key = getattr(timezone, "key", str(timezone))
    labels = {
        "America/New_York": "ET",
        "Asia/Shanghai": "CST",
        "Asia/Hong_Kong": "HKT",
        "UTC": "UTC",
    }
    return labels.get(key, key)

def _positions_wide(positions: pd.DataFrame) -> pd.DataFrame:
    """Return numeric date x symbol position values."""
    frame = pd.DataFrame(positions).copy()
    if frame.empty:
        return pd.DataFrame()
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
    symbol_col = (
        lowered.get("symbol")
        or lowered.get("ticker")
        or lowered.get("asset")
        or lowered.get("data")
    )
    value_col = lowered.get("value") or lowered.get("market_value") or lowered.get("position_value")
    if date_col is not None and symbol_col is not None and value_col is not None:
        frame["__date"] = _normalize_dates(frame[date_col])
        frame["__value"] = pd.to_numeric(frame[value_col], errors="coerce")
        wide = frame.pivot_table(
            index="__date",
            columns=symbol_col,
            values="__value",
            aggfunc="last",
        )
        return wide.sort_index().ffill().fillna(0.0)
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

def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame

def _align_series_to_index(series: pd.Series, index: pd.Index) -> pd.Series:
    """Return numeric series aligned to a datetime index with forward-filled values."""
    values = pd.Series(series).copy()
    if not isinstance(values.index, pd.DatetimeIndex):
        values.index = _normalize_dates(pd.Series(values.index))
    elif isinstance(values.index.dtype, pd.DatetimeTZDtype):
        values.index = values.index.tz_convert("UTC").tz_localize(None)
    target = pd.to_datetime(index, errors="coerce")
    if isinstance(target.dtype, pd.DatetimeTZDtype):
        target = target.tz_convert("UTC").tz_localize(None)
    values = pd.to_numeric(values.sort_index(), errors="coerce")
    return values.reindex(target, method="nearest").ffill()

def _market_section(title: str, *, height: int, x_range, y_range=None):
    """Return a linked replay figure."""
    kwargs = {"y_range": y_range} if y_range is not None else {}
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
        **kwargs,
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

def _allocation_stack_colors(stackers: list[str]) -> list[str]:
    """Return portfolio allocation colors with muted residual buckets."""
    asset_colors = iter(
        _palette(len([item for item in stackers if item not in {"Others", "Cash"}]))
    )
    colors = []
    for stacker in stackers:
        if stacker == "Others":
            colors.append(PORTFOLIO_OTHERS)
        elif stacker == "Cash":
            colors.append(PORTFOLIO_CASH)
        else:
            colors.append(next(asset_colors))
    return colors

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
    for axis in plot.yaxis:
        if hasattr(axis.ticker, "desired_num_ticks"):
            axis.ticker.desired_num_ticks = min(axis.ticker.desired_num_ticks, 5)
    _style_plot_title(plot, standoff=MARKET_REPLAY_TITLE_STANDOFF)
    plot.toolbar.logo = None
    plot.add_tools(CrosshairTool(dimensions="both"))
    _hide_market_legend(plot)

def _sync_replay_crosshair(plots: list[Any]) -> None:
    """Add a shared vertical guide across linked replay panels."""
    spans = []
    for plot in plots:
        span = Span(
            location=0,
            dimension="height",
            line_color="#24313a",
            line_alpha=0.55,
            line_width=1,
            visible=False,
            name="shared_replay_crosshair",
        )
        plot.add_layout(span)
        spans.append(span)

    move = CustomJS(
        args={"spans": spans},
        code="""
for (const span of spans) {
  span.location = cb_obj.x;
  span.visible = true;
}
""",
    )
    leave = CustomJS(
        args={"spans": spans},
        code="""
for (const span of spans) {
  span.visible = false;
}
""",
    )
    for plot in plots:
        plot.js_on_event(MouseMove, move)
        plot.js_on_event(MouseLeave, leave)

def _hide_market_legend(plot) -> None:
    """Hide any legends attached to a market panel."""
    if plot.legend:
        for legend in list(plot.legend):
            legend.visible = False

def _style_market_legend(plot, *, compact: bool = False, large_glyphs: bool = False) -> None:
    """Apply consistent layout and visual styling to market panel legends."""
    if plot is None or not plot.legend:
        return
    _add_legend_frame_gap(plot)
    for legend in list(plot.legend):
        legend.visible = True
        legend.location = "top_left"
        item_count = len(legend.items)
        legend.orientation = "vertical"
        legend.ncols = item_count
        legend.nrows = "auto"
        legend.border_line_width = 1
        legend.border_line_alpha = 0.65
        legend.border_line_color = "#d7e0e7"
        legend.background_fill_color = "white"
        legend.background_fill_alpha = 0.82
        legend.padding = 3
        legend.spacing = 8 if compact else 10
        legend.margin = MARKET_LEGEND_MARGIN
        if compact and large_glyphs:
            legend.glyph_width = 22
            legend.glyph_height = 16
        else:
            legend.glyph_width = 16 if compact else (24 if large_glyphs else 20)
            legend.glyph_height = 10 if compact else (16 if large_glyphs else 14)
        legend.label_text_color = "#33424f"
        legend.label_text_font_size = "8pt"
        legend.click_policy = "hide"
        _move_legend_outside(plot, legend)
    if MARKET_LEGEND_COMPENSATED_TAG not in plot.tags:
        plot.height += MARKET_LEGEND_HEIGHT_COMPENSATION
        plot.tags.append(MARKET_LEGEND_COMPENSATED_TAG)

def _move_legend_outside(plot, legend: Legend) -> None:
    """Move a legend above the plot so it does not cover or narrow plotted data."""
    if legend in plot.above:
        return
    plot.add_layout(legend, "above")

def _add_legend_frame_gap(plot) -> None:
    """Place a transparent gap between the external legend and the plot frame."""
    if MARKET_LEGEND_FRAME_GAP_TAG in plot.tags:
        return
    gap = Spacer(width=1, height=MARKET_LEGEND_FRAME_GAP, sizing_mode="fixed")
    gap.tags.append(MARKET_LEGEND_FRAME_GAP_TAG)
    plot.add_layout(gap, "above")
    plot.tags.append(MARKET_LEGEND_FRAME_GAP_TAG)

def _datetime_rect_width(dates: pd.Series) -> float:
    """Return a Bokeh datetime rect width in milliseconds."""
    clean = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().sort_values()
    if len(clean) < 2:
        return 24 * 60 * 60 * 1000 * 0.9
    diffs = clean.diff().dropna().dt.total_seconds() * 1000.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 24 * 60 * 60 * 1000 * 0.9
    return float(diffs.median() * 0.92)

def _make_static_chart(plot) -> None:
    """Render report support charts without visible interactive controls."""
    _style_plot_title(plot)
    plot.toolbar_location = None
    plot.toolbar.tools = []
    plot.toolbar.logo = None
    plot.toolbar.active_drag = None
    plot.toolbar.active_scroll = None
    plot.toolbar.active_inspect = None

def _style_plot_title(plot, *, standoff: int = 32) -> None:
    """Apply the report title treatment to Bokeh figures."""
    plot.title.text_color = MARKET_TITLE
    plot.title.text_font_size = "16px"
    plot.title.text_font_style = "bold"
    plot.title.text_font = "Arial"
    plot.title.align = "left"
    plot.title.text_align = "left"
    plot.title.offset = 0
    plot.title.standoff = standoff

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
    """Attach hover behavior to a plot."""
    plot.add_tools(hover)
