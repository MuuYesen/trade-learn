from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "check_pynecore_poc.py"


def load_poc_module():
    spec = importlib.util.spec_from_file_location("check_pynecore_poc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pynecore_poc_report_captures_indicator_coverage_and_gaps() -> None:
    module = load_poc_module()

    report = module.build_pynecore_poc_report()

    assert report["package"]["importable"] is True
    assert report["package"]["distribution"] == "pynesys-pynecore"
    assert report["ta_module"]["importable"] is True
    assert report["indicator_coverage"]["required_count"] == 22
    assert report["indicator_coverage"]["available_count"] >= 20
    assert "ichimoku" in report["indicator_coverage"]["missing"]
    assert "adx" in report["indicator_coverage"]["missing"]
    assert report["indicator_coverage"]["aliases"]["adx"] == "dmi(...)[2]"
    assert report["streaming_primitives"]["Series"] is True
    assert report["streaming_primitives"]["PersistentSeries"] is True
    assert report["decision"]["status"] == "conditional"


def test_pynecore_poc_cli_outputs_json() -> None:
    module = load_poc_module()

    payload = module.render_report_json(module.build_pynecore_poc_report())
    parsed = json.loads(payload)

    assert parsed["package"]["importable"] is True
    assert parsed["decision"]["next_step"] == "implement thin ta.tv adapter for covered functions"
