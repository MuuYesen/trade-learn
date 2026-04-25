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
from bokeh.layouts import column
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
    exposure = reporter.exposure()
    correlation = reporter.correlation_matrix()
    factor_quantile_returns = reporter.factor_quantile_returns()
    summary = reporter.summary(benchmark=benchmark_returns)
    config = reporter._get("config", default={}) or {}
    metadata = _metadata(summary, returns, config)
    plots = [
        charts.equity_curve(reporter.equity_curve(), benchmark_returns),
        charts.drawdown(reporter.drawdown()),
        charts.monthly_heatmap(reporter.monthly_heatmap()),
        charts.rolling_sharpe(reporter.rolling_sharpe()),
        charts.trade_distribution(reporter.trade_distribution()),
    ]
    if _has_multi_asset_exposure(exposure):
        plots.append(charts.correlation_matrix(correlation))
        plots.append(charts.exposure(exposure))
    if not factor_quantile_returns.empty:
        plots.append(charts.quantile_returns(factor_quantile_returns))
    script, chart_components = components(
        column(*plots)
    )
    output.write_text(
        _render_html(
            summary=summary,
            charts=chart_components,
            script=script,
            bokeh_resources=INLINE.render(),
            metadata=metadata,
            drawdowns=reporter.top_drawdowns(limit=10),
            benchmark=benchmark_returns,
            correlation=correlation,
            exposure=exposure,
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
        factor_quantile_returns=factor_quantile_returns,
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
    factor_quantile_returns: pd.DataFrame,
) -> None:
    """Write colocated machine-readable report artifacts."""
    equity.to_frame("equity").to_parquet(directory / "equity.parquet")
    trades.to_parquet(directory / "trades.parquet", index=False)
    if not factor_quantile_returns.empty:
        factor_quantile_returns.to_parquet(directory / "factor_quantile_returns.parquet")
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
    factor_quantile_returns: pd.DataFrame,
    trades: pd.DataFrame,
    returns: pd.Series,
    config: dict[str, Any],
) -> str:
    """Render the report HTML string."""
    template = _template_environment().get_template(TEMPLATE_NAME)
    return template.render(
        benchmark_section=_benchmark_section(benchmark),
        bokeh_resources=bokeh_resources,
        charts=charts,
        config_table=_summary_table(config),
        correlation_section=_correlation_section(correlation),
        drawdowns_table=_frame_table(drawdowns),
        exposure_section=_exposure_section(exposure),
        factor_section=_factor_section(factor_quantile_returns),
        metadata={key: escape(value) for key, value in metadata.items()},
        returns_count=len(returns),
        script=script,
        summary_table=_summary_table(summary),
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
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


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
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
