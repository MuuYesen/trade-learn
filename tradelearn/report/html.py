"""HTML tear sheet export."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.resources import INLINE


def write_html_report(reporter: Any, path: str | Path) -> Path:
    """Write a single-file HTML tear sheet and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(reporter._get("returns")).copy()
    trades = pd.DataFrame(reporter._get("trades", default=pd.DataFrame())).copy()
    summary = reporter.summary()
    script, charts = components(
        column(
            _equity_plot(reporter.equity_curve()),
            _drawdown_plot(reporter.drawdown()),
            _trade_distribution_plot(trades),
        )
    )
    output.write_text(
        _render_html(
            summary=summary,
            charts=charts,
            script=script,
            bokeh_resources=INLINE.render(),
            trades=trades,
            returns=returns,
            config=reporter._get("config", default={}) or {},
        )
    )
    return output


def _equity_plot(equity: pd.Series):
    """Return an equity curve figure."""
    frame = _plot_frame(equity, "equity")
    plot = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=260,
        sizing_mode="stretch_width",
    )
    plot.line(frame["date"], frame["equity"], line_width=2, color="#1f77b4")
    return plot


def _drawdown_plot(drawdown: pd.Series):
    """Return a drawdown figure."""
    frame = _plot_frame(drawdown, "drawdown")
    plot = figure(
        title="Drawdown",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    plot.varea(frame["date"], y1=0, y2=frame["drawdown"], color="#d62728", alpha=0.35)
    plot.line(frame["date"], frame["drawdown"], line_width=2, color="#d62728")
    return plot


def _trade_distribution_plot(trades: pd.DataFrame):
    """Return a trade PnL histogram figure."""
    plot = figure(title="Trade Distribution", height=220, sizing_mode="stretch_width")
    if "pnl" not in trades or trades.empty:
        return plot
    hist, edges = pd.cut(trades["pnl"], bins=min(20, max(1, len(trades))), retbins=True)
    counts = hist.value_counts(sort=False).to_numpy()
    plot.quad(
        top=counts,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="#2ca02c",
        line_color="white",
        alpha=0.65,
    )
    return plot


def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame


def _render_html(
    *,
    summary: dict[str, Any],
    charts: str,
    script: str,
    bokeh_resources: str,
    trades: pd.DataFrame,
    returns: pd.Series,
    config: dict[str, Any],
) -> str:
    """Render the report HTML string."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tradelearn Report</title>
  {bokeh_resources}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <header>
    <h1>Tradelearn Report</h1>
    <p>Rows: returns={len(returns)}, trades={len(trades)}</p>
  </header>
  <section>
    <h2>Summary Stats</h2>
    {_summary_table(summary)}
  </section>
  <section>
    <h2>Equity Curve</h2>
    <h2>Drawdown</h2>
    <h2>Trade Distribution</h2>
    {charts}
    {script}
  </section>
  <section>
    <h2>Configuration</h2>
    {_summary_table(config)}
  </section>
</body>
</html>
"""


def _summary_table(values: dict[str, Any]) -> str:
    """Render a dict as an HTML table."""
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _format_value(value: Any) -> str:
    """Format scalar values for HTML."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
