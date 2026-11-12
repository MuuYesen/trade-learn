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


def write_html(reporter: Any, path: str | Path, sections: Any = None) -> Path:
    """Write domain facts without executing their producer."""
    result = reporter._analysis
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>",
        escape(result["title"]),
        "</title></head><body><h1>",
        escape(result["title"]),
        "</h1>",
    ]
    tables = {
        "metadata": pd.DataFrame(
            [{"key": key, "value": value} for key, value in result["metadata"].items()]
        ),
        "metrics": result["metrics"],
        **result["tables"],
    }
    for name, table in tables.items():
        parts.extend(
            [
                "<h2>",
                escape(name),
                "</h2>",
                table.to_html(
                    escape=True, na_rep="NA", float_format=lambda value: format(value, ".17g")
                ),
            ]
        )
    figures = {name: figure for name, figure in result["figures"].items() if figure is not None}
    for name, figure in result["figures"].items():
        if figure is None:
            parts.extend(["<h2>", escape(str(name)), "</h2><p>Not provided</p>"])
    if figures:
        from bokeh.embed import components
        from bokeh.resources import INLINE

        parts.append(INLINE.render())
        for name, figure in figures.items():
            script, div = components(figure)
            parts.extend(["<h2>", escape(str(name)), "</h2>", div, script])
    parts.extend([render_html_sections(sections, build_context(reporter)), "</body></html>"])
    output.write_text("".join(parts), encoding="utf-8")
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
