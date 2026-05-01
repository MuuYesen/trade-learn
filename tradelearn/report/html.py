"""HTML tear sheet export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tradelearn.report import charts

TEMPLATE_DIR = Path(__file__).with_name("templates")
TEMPLATE_NAME = "tear_sheet.html"


def write_html_report(
    reporter: Any,
    path: str | Path,
    benchmark: pd.Series | None = None,
) -> Path:
    """Write a single-file HTML tear sheet and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(reporter._get("returns")).copy()
    benchmark_returns = None if benchmark is None else pd.Series(benchmark).copy()
    trades = pd.DataFrame(reporter._get("trades", default=pd.DataFrame())).copy()
    rolling_beta = (
        pd.Series(dtype="float64", name="rolling_beta")
        if benchmark_returns is None
        else reporter.rolling_beta(benchmark_returns)
    )
    exposure = reporter.exposure()
    correlation = reporter.correlation_matrix()
    factor_ic = reporter.factor_ic()
    factor_rank_ic = reporter.factor_rank_ic()
    factor_turnover = reporter.factor_turnover()
    factor_autocorrelation = reporter.factor_autocorrelation()
    factor_long_short_returns = reporter.factor_long_short_returns()
    factor_quantile_returns = reporter.factor_quantile_returns()
    summary = reporter.summary(benchmark=benchmark_returns)
    config = reporter._get("config", default={}) or {}
    metadata = _metadata(summary, returns, config)
    top_drawdowns = reporter.top_drawdowns(limit=10)
    plots = {}
    plots["Equity Curve"] = charts.equity_curve(
        reporter.equity_curve(),
        benchmark_returns,
        drawdowns=top_drawdowns.head(5),
    )
    plots["Drawdown"] = charts.drawdown(reporter.drawdown())
    plots["Monthly Returns Heatmap"] = charts.monthly_heatmap(reporter.monthly_heatmap())
    plots["Rolling Sharpe"] = charts.rolling_sharpe(reporter.rolling_sharpe())
    plots["Trade Distribution"] = charts.trade_distribution(reporter.trade_distribution())
    price_plot = reporter.price_trades_chart()
    if price_plot is not None:
        plots["Price / Trades"] = price_plot
    if not rolling_beta.empty:
        plots["Rolling Beta"] = charts.rolling_beta(rolling_beta)
    if _has_multi_asset_exposure(exposure):
        plots["Correlation Matrix"] = charts.correlation_matrix(correlation)
        plots["Exposure Chart"] = charts.exposure(exposure)
    if not factor_ic.empty:
        plots["Factor IC"] = charts.factor_ic(factor_ic)
    if not factor_rank_ic.empty:
        plots["Factor Rank IC"] = charts.factor_rank_ic(factor_rank_ic)
    if not factor_turnover.empty or not factor_autocorrelation.empty:
        plots["Factor Turnover"] = charts.factor_turnover(
            factor_turnover,
            factor_autocorrelation,
        )
    if not factor_long_short_returns.empty:
        plots["Factor Long-Short Returns"] = charts.factor_long_short_returns(
            factor_long_short_returns
        )
    if not factor_quantile_returns.empty:
        plots["Factor Quantile Returns"] = charts.quantile_returns(factor_quantile_returns)
    plots = {title: plot for title, plot in plots.items() if _has_glyph_renderers(plot)}
    script, chart_components = components(plots)
    output.write_text(
        _render_html(
            summary=summary,
            charts=chart_components,
            script=script,
            bokeh_resources=INLINE.render(),
            metadata=metadata,
            drawdowns=top_drawdowns,
            benchmark=benchmark_returns,
            correlation=correlation,
            exposure=exposure,
            factor_ic=factor_ic,
            factor_rank_ic=factor_rank_ic,
            factor_turnover=factor_turnover,
            factor_autocorrelation=factor_autocorrelation,
            factor_long_short_returns=factor_long_short_returns,
            factor_quantile_returns=factor_quantile_returns,
            trades=trades,
            returns=returns,
            config=config,
        )
    )
    _write_artifacts(
        directory=output.parent,
        config=config,
        equity=reporter.equity_curve(),
        metadata=metadata,
        summary=summary,
        trades=trades,
        factor_ic=factor_ic,
        factor_rank_ic=factor_rank_ic,
        factor_turnover=factor_turnover,
        factor_autocorrelation=factor_autocorrelation,
        factor_long_short_returns=factor_long_short_returns,
        factor_quantile_returns=factor_quantile_returns,
        rolling_beta=rolling_beta,
    )
    return output


def _write_artifacts(
    *,
    directory: Path,
    config: dict[str, Any],
    equity: pd.Series,
    metadata: dict[str, str],
    summary: dict[str, Any],
    trades: pd.DataFrame,
    factor_ic: pd.Series,
    factor_rank_ic: pd.Series,
    factor_turnover: pd.Series,
    factor_autocorrelation: pd.Series,
    factor_long_short_returns: pd.DataFrame,
    factor_quantile_returns: pd.DataFrame,
    rolling_beta: pd.Series,
) -> None:
    """Write colocated machine-readable report artifacts."""
    equity.to_frame("equity").to_parquet(directory / "equity.parquet")
    trades.to_parquet(directory / "trades.parquet", index=False)
    if not factor_ic.empty:
        factor_ic.to_frame("ic").to_parquet(directory / "factor_ic.parquet")
    if not factor_rank_ic.empty:
        factor_rank_ic.to_frame("rank_ic").to_parquet(directory / "factor_rank_ic.parquet")
    if not factor_turnover.empty:
        factor_turnover.to_frame("turnover").to_parquet(directory / "factor_turnover.parquet")
    if not factor_autocorrelation.empty:
        factor_autocorrelation.to_frame("autocorrelation").to_parquet(
            directory / "factor_autocorrelation.parquet"
        )
    if not factor_long_short_returns.empty:
        factor_long_short_returns.to_parquet(directory / "factor_long_short_returns.parquet")
    if not factor_quantile_returns.empty:
        factor_quantile_returns.to_parquet(directory / "factor_quantile_returns.parquet")
    if not rolling_beta.empty:
        rolling_beta.to_frame("rolling_beta").to_parquet(directory / "rolling_beta.parquet")
    (directory / "stats.json").write_text(
        json.dumps(
            {
                "config": _json_safe(config),
                "metadata": metadata,
                "summary": _json_safe(summary),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _has_glyph_renderers(plot: Any) -> bool:
    """Return True when a Bokeh figure has actual plotted glyphs."""
    if getattr(plot, "renderers", ()):
        return True
    children = getattr(plot, "children", ())
    for child in children:
        item = child[0] if isinstance(child, tuple) else child
        if _has_glyph_renderers(item):
            return True
    return False


def _render_html(
    *,
    summary: dict[str, Any],
    charts: str,
    script: str,
    bokeh_resources: str,
    metadata: dict[str, str],
    drawdowns: pd.DataFrame,
    benchmark: pd.Series | None,
    correlation: pd.DataFrame,
    exposure: pd.DataFrame,
    factor_ic: pd.Series,
    factor_rank_ic: pd.Series,
    factor_turnover: pd.Series,
    factor_autocorrelation: pd.Series,
    factor_long_short_returns: pd.DataFrame,
    factor_quantile_returns: pd.DataFrame,
    trades: pd.DataFrame,
    returns: pd.Series,
    config: dict[str, Any],
) -> str:
    """Render the report HTML string."""
    template = _template_environment().get_template(TEMPLATE_NAME)
    display_config = {key: value for key, value in config.items() if key != "pipeline"}
    return template.render(
        benchmark_section=_benchmark_section(benchmark),
        bokeh_resources=bokeh_resources,
        charts=charts,
        config_table=_summary_table(display_config),
        correlation_section=_correlation_section(correlation),
        drawdowns_table=_frame_table(drawdowns),
        exposure_section=_exposure_section(exposure),
        factor_ic_section=_factor_ic_section(factor_ic),
        factor_rank_ic_section=_factor_rank_ic_section(factor_rank_ic),
        factor_turnover_section=_factor_turnover_section(
            factor_turnover,
            factor_autocorrelation,
        ),
        factor_long_short_section=_factor_long_short_section(factor_long_short_returns),
        factor_section=_factor_section(factor_quantile_returns),
        metadata={key: escape(value) for key, value in metadata.items()},
        pipeline_section=_pipeline_section(config),
        returns_count=len(returns),
        script=script,
        summary_table=_summary_cards(summary),
        trades_count=len(trades),
    )


def _template_environment() -> Environment:
    """Return the Jinja2 environment for report templates."""
    return Environment(
        autoescape=select_autoescape(["html", "xml"]),
        loader=FileSystemLoader(TEMPLATE_DIR),
    )


def _summary_table(values: dict[str, Any]) -> str:
    """Render a dict as an HTML table."""
    if not values:
        return "<table><tbody></tbody></table>"
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _summary_cards(values: dict[str, Any]) -> str:
    """Render summary values as a compact multi-column table."""
    items = list(values.items())
    columns = 3
    rows = []
    for start in range(0, len(items), columns):
        cells = []
        for key, value in items[start : start + columns]:
            cells.append(f"<th>{escape(str(key))}</th><td>{escape(_format_value(value))}</td>")
        missing = columns - len(cells)
        cells.extend("<th></th><td></td>" for _ in range(missing))
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table class=\"summary-table\"><tbody>{''.join(rows)}</tbody></table>"


def _pipeline_section(config: dict[str, Any]) -> str:
    """Render pipeline parameters as a readable experiment subsection."""
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline:
        return ""
    rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in _flatten_display("pipeline", pipeline).items()
    )
    return (
        "<h3>Pipeline Parameters</h3>"
        "<table><thead><tr><th>parameter</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _flatten_display(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_display(name, value))
        else:
            flattened[name] = value
    return flattened


def _frame_table(frame: pd.DataFrame) -> str:
    """Render a frame as an HTML table."""
    if frame.empty:
        return "<table><tbody></tbody></table>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        + "</tr>"
        for row in frame.itertuples(index=False, name=None)
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _metadata(
    summary: dict[str, Any],
    returns: pd.Series,
    config: dict[str, Any],
) -> dict[str, str]:
    """Return report header and footer metadata."""
    strategy_name = str(
        summary.get("strategy_name") or config.get("strategy") or "Tradelearn Report"
    )
    run_id = str(config.get("run_id") or summary.get("run_id") or "-")
    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "start": _format_date(returns.index.min()),
        "end": _format_date(returns.index.max()),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": _package_version(),
    }


def _exposure_section(exposure: pd.DataFrame) -> str:
    """Return the optional exposure section heading."""
    if not _has_multi_asset_exposure(exposure):
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in exposure.columns)
    return f"<h2>Exposure Chart</h2><p>Symbols: {symbols}</p>"


def _correlation_section(correlation: pd.DataFrame) -> str:
    """Return the optional correlation matrix section heading."""
    if correlation.empty or len(correlation.columns) <= 1:
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in correlation.columns)
    return f"<h2>Correlation Matrix</h2><p>Symbols: {symbols}</p>"


def _benchmark_section(benchmark: pd.Series | None) -> str:
    """Return the optional benchmark section heading."""
    if benchmark is None:
        return ""
    name = escape(str(benchmark.name or "benchmark"))
    return f"<section><h2>Benchmark</h2><p>{name}</p></section>"


def _factor_section(factor_quantile_returns: pd.DataFrame) -> str:
    """Return the optional factor quantile section heading."""
    if factor_quantile_returns.empty:
        return ""
    quantiles = ", ".join(escape(str(column)) for column in factor_quantile_returns.columns)
    return f"<h2>Factor Quantile Returns</h2><p>Quantiles: {quantiles}</p>"


def _factor_ic_section(factor_ic: pd.Series) -> str:
    """Return the optional factor IC section heading."""
    if factor_ic.empty:
        return ""
    return f"<h2>Factor IC</h2><p>Observations: {len(factor_ic)}</p>"


def _factor_rank_ic_section(factor_rank_ic: pd.Series) -> str:
    """Return the optional factor rank IC section heading."""
    if factor_rank_ic.empty:
        return ""
    return f"<h2>Factor Rank IC</h2><p>Observations: {len(factor_rank_ic)}</p>"


def _factor_turnover_section(
    factor_turnover: pd.Series,
    factor_autocorrelation: pd.Series,
) -> str:
    """Return the optional factor turnover section heading."""
    if factor_turnover.empty and factor_autocorrelation.empty:
        return ""
    observations = max(len(factor_turnover), len(factor_autocorrelation))
    return f"<h2>Factor Turnover</h2><p>Observations: {observations}</p>"


def _factor_long_short_section(factor_long_short_returns: pd.DataFrame) -> str:
    """Return the optional factor long-short section heading."""
    if factor_long_short_returns.empty:
        return ""
    columns = ", ".join(escape(str(column)) for column in factor_long_short_returns.columns)
    return f"<h2>Factor Long-Short Returns</h2><p>Series: {columns}</p>"


def _has_multi_asset_exposure(exposure: pd.DataFrame) -> bool:
    """Return whether exposure should be rendered as a multi-asset section."""
    return not exposure.empty and len(exposure.columns) > 1


def _format_date(value: Any) -> str:
    """Format report date metadata."""
    if pd.isna(value):
        return "-"
    return str(value)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable value."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _package_version() -> str:
    """Return installed package version for report footer."""
    try:
        return version("trade-learn")
    except PackageNotFoundError:
        return "unknown"


def _format_value(value: Any) -> str:
    """Format scalar values for HTML."""
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
