"""Shared rendering of already-computed domain tables and metric facts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.report.sections import (
    _excel_sheet_name,
    build_context,
    collect_excel_sheets,
    render_html_sections,
)


def snapshot(result: Any) -> dict[str, Any]:
    """Copy completed facts, rejecting ambiguous worksheet identities."""

    def field(name, default):
        return (
            result.get(name, default)
            if isinstance(result, Mapping)
            else getattr(result, name, default)
        )

    tables = field("tables", {})
    if not isinstance(tables, Mapping) or not tables:
        raise ValueError("analysis result requires nonempty tables mapping")
    copied = {}
    seen = {"metrics", "metadata", "figures"}
    for name, table in tables.items():
        safe_name = _excel_sheet_name(str(name))
        if safe_name.casefold() in seen:
            raise ValueError(f"duplicate or reserved sheet name: {name}")
        seen.add(safe_name.casefold())
        frame = pd.DataFrame(table).copy(deep=True)
        if not isinstance(frame.index, pd.RangeIndex) or any(
            name is not None for name in frame.index.names
        ):
            frame = frame.reset_index()
        copied[safe_name] = frame
    rows = []
    for name, metric in field("metrics", {}).items():
        fact = dict(metric) if isinstance(metric, Mapping) else {"value": metric}
        value = fact.get("value")
        if not pd.api.types.is_scalar(value):
            raise ValueError("analysis metrics must contain scalar values")
        missing = value is None or pd.isna(value)
        rows.append(
            {
                "name": str(name),
                "value": value,
                "unit": fact.get("unit", "unspecified"),
                "reason": fact.get(
                    "reason",
                    "not_provided" if value is None else "not_calculable" if missing else "",
                ),
            }
        )
    from tradelearn import __version__

    metadata = deepcopy(dict(field("metadata", {})))
    metadata.setdefault("report_schema_version", "1")
    metadata.setdefault("tradelearn_version", __version__)
    return {
        "title": str(field("title", "TradeLearn analysis")),
        "tables": copied,
        "metrics": pd.DataFrame(rows, columns=["name", "value", "unit", "reason"]),
        "metadata": metadata,
        "figures": dict(field("figures", {})),
    }


DISPLAY_DIGITS = 6


def _display(value: Any) -> tuple[str, str]:
    """Return (display text, exact text) for one cell.

    The HTML is for reading; Excel stays the lossless channel. The full
    repr is still carried on the cell so no precision is lost on screen.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA", ""
    if isinstance(value, float):
        exact = format(value, ".17g")
        shown = f"{value:.{DISPLAY_DIGITS}f}" if abs(value) < 1e9 else format(value, ".6g")
        shown = shown.rstrip("0").rstrip(".") if "." in shown else shown
        return (shown or "0"), ("" if shown == exact else exact)
    if isinstance(value, pd.Timestamp):
        text = str(value.tz_convert(None) if value.tzinfo is not None else value).split(".")[0]
        return text, ""
    return str(value), ""


def _cell(value: Any) -> str:
    shown, exact = _display(value)
    title = f' title="{escape(exact)}"' if exact else ""
    return f"<td{title}>{escape(shown)}</td>"


def _table_html(frame: pd.DataFrame) -> str:
    """Render a frame with the shared report table styling."""
    table = pd.DataFrame(frame)
    if table.empty:
        return '<div class="table-scroll"><table><tbody></tbody></table></div>'
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    rows = "".join(
        "<tr>" + "".join(_cell(value) for value in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{headers}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _metric_cards(metrics: Any) -> str:
    """Render analysis metrics as the shared KPI cards."""
    frame = pd.DataFrame(metrics)
    if frame.empty:
        return ""
    cards = []
    for row in frame.itertuples(index=False, name=None):
        values = dict(zip(frame.columns, row, strict=False))
        shown, exact = _display(values.get("value"))
        if values.get("value") is None or shown == "NA":
            shown = "not_calculable"
        note = " · ".join(
            str(values[key]) for key in ("unit", "reason") if values.get(key) not in (None, "")
        )
        title = f' title="{escape(exact)}"' if exact else ""
        cards.append(
            '<div class="kpi-card">'
            f'<div class="kpi-label">{escape(str(values.get("name", "")))}</div>'
            f'<div class="kpi-value"{title}>{escape(shown)}</div>'
            + (f'<div class="kpi-delta">{escape(note)}</div>' if note else "")
            + "</div>"
        )
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def write_html(reporter: Any, path: str | Path, sections: Any = None) -> Path:
    """Write domain facts without executing their producer."""
    from bokeh.embed import components
    from bokeh.resources import INLINE

    from tradelearn.report.html import _template_environment

    result = reporter._analysis
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figures = {name: item for name, item in result["figures"].items() if item is not None}
    rendered: list[dict[str, str]] = []
    scripts: list[str] = []
    for name, item in result["figures"].items():
        if item is None:
            rendered.append({"name": str(name), "html": "<p>Not provided</p>"})
    for name, item in figures.items():
        script, div = components(item)
        rendered.append({"name": str(name), "html": div})
        scripts.append(script)

    template = _template_environment().get_template("analysis_report.html")
    content = template.render(
        title=result["title"],
        metadata_items=[
            {"label": str(key), "value": str(value)} for key, value in result["metadata"].items()
        ],
        metric_cards=_metric_cards(result["metrics"]),
        tables=[
            {"name": str(name), "html": _table_html(table)}
            for name, table in result["tables"].items()
        ],
        figures=rendered,
        extra_sections=render_html_sections(sections, build_context(reporter)),
        bokeh_resources=INLINE.render() if figures else "",
        figure_scripts="".join(scripts),
    )
    output.write_text(content, encoding="utf-8")
    return output


def write_excel(reporter: Any, path: str | Path, sections: Any = None) -> Path:
    """Write the same facts and mark figures without an Excel representation."""
    from tradelearn.report.excel import _excel_safe_frame

    result = reporter._analysis
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "metadata": pd.DataFrame(
            [{"key": key, "value": str(value)} for key, value in result["metadata"].items()]
        ),
        "metrics": result["metrics"],
        **result["tables"],
    }
    if result["figures"]:
        tables["figures"] = pd.DataFrame(
            [
                {"name": str(name), "status": "HTML only; no Excel figure representation"}
                for name in result["figures"]
            ]
        )
    for name, table in collect_excel_sheets(sections, build_context(reporter)).items():
        if name.casefold() in {key.casefold() for key in tables}:
            raise ValueError(f"duplicate sheet name: {name}")
        tables[name] = table
    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}},
    ) as writer:
        for name, table in tables.items():
            _excel_safe_frame(table).to_excel(writer, sheet_name=name, index=False)
    return output
