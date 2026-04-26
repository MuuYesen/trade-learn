from __future__ import annotations

import argparse
import importlib
import json
from importlib import metadata
from typing import Any

REQUIRED_TV_INDICATORS = (
    "rsi",
    "macd",
    "bb",
    "supertrend",
    "ichimoku",
    "vwap",
    "adx",
    "atr",
    "ema",
    "sma",
    "wma",
    "hma",
    "stoch",
    "cci",
    "roc",
    "mom",
    "sar",
    "highest",
    "lowest",
    "crossover",
    "crossunder",
    "dmi",
)

ALIASES = {
    "adx": "dmi(...)[2]",
}


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_pynecore_poc_report() -> dict[str, Any]:
    package: dict[str, Any] = {
        "distribution": "pynesys-pynecore",
        "version": _distribution_version("pynesys-pynecore"),
        "importable": False,
        "module": None,
    }
    ta_module: dict[str, Any] = {"importable": False, "module": "pynecore.lib.ta"}
    coverage = {
        "required": list(REQUIRED_TV_INDICATORS),
        "required_count": len(REQUIRED_TV_INDICATORS),
        "available": [],
        "available_count": 0,
        "missing": [],
        "aliases": ALIASES,
    }
    streaming_primitives = {
        "Series": False,
        "Persistent": False,
        "PersistentSeries": False,
        "inline_series": False,
    }

    try:
        pynecore = importlib.import_module("pynecore")
        package["importable"] = True
        package["module"] = getattr(pynecore, "__file__", None)
        streaming_primitives["Series"] = hasattr(pynecore, "Series")
        streaming_primitives["Persistent"] = hasattr(pynecore, "Persistent")
        streaming_primitives["PersistentSeries"] = hasattr(pynecore, "PersistentSeries")
    except Exception as exc:  # pragma: no cover - environment dependent failure path
        package["error"] = f"{type(exc).__name__}: {exc}"

    try:
        ta = importlib.import_module("pynecore.lib.ta")
        ta_module["importable"] = True
        available = [name for name in REQUIRED_TV_INDICATORS if hasattr(ta, name)]
        coverage["available"] = available
        coverage["available_count"] = len(available)
        coverage["missing"] = [
            name for name in REQUIRED_TV_INDICATORS if name not in set(available)
        ]
    except Exception as exc:  # pragma: no cover - environment dependent failure path
        ta_module["error"] = f"{type(exc).__name__}: {exc}"

    try:
        series = importlib.import_module("pynecore.core.series")
        streaming_primitives["inline_series"] = hasattr(series, "inline_series")
    except Exception:  # pragma: no cover - environment dependent failure path
        pass

    status = "pass" if not coverage["missing"] else "conditional"
    return {
        "package": package,
        "ta_module": ta_module,
        "indicator_coverage": coverage,
        "streaming_primitives": streaming_primitives,
        "decision": {
            "status": status,
            "reason": (
                "pyneCore is importable and covers most planned ta.tv indicators, "
                "but direct ichimoku/adx wrappers and TradingView golden data are still missing."
            ),
            "next_step": "implement thin ta.tv adapter for covered functions",
        },
    }


def render_report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check pyneCore PoC readiness.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = build_pynecore_poc_report()
    if args.json:
        print(render_report_json(report))
    else:
        coverage = report["indicator_coverage"]
        print(
            "pynecore-poc:"
            f"importable={report['package']['importable']} "
            f"version={report['package']['version']} "
            f"ta={report['ta_module']['importable']} "
            f"available={coverage['available_count']}/{coverage['required_count']} "
            f"missing={','.join(coverage['missing']) or '-'} "
            f"status={report['decision']['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
