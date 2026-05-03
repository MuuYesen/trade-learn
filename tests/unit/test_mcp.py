from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tradelearn.core.config import TradelearnConfig
from tradelearn.mcp import build_server


def test_mcp_server_exposes_project_tools(tmp_path: Path) -> None:
    server = build_server(
        config=TradelearnConfig(mlflow_tracking_uri="http://mlflow.local"),
        project_dir=tmp_path,
    )

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}

    assert {
        "project_info",
        "runtime_config",
        "lab_plan",
        "search_api",
        "get_api_docs",
        "list_mlflow_runs",
    }.issubset(tool_names)
    assert server.name == "tradelearn"


def test_mcp_search_api_returns_public_symbols(tmp_path: Path) -> None:
    server = build_server(project_dir=tmp_path)

    result = _call_json(server, "search_api", {"query": "TradingViewProvider"})

    assert result["query"] == "TradingViewProvider"
    assert any(item["path"] == "tradelearn.data.TradingViewProvider" for item in result["results"])


def test_mcp_get_api_docs_returns_signature_and_doc(tmp_path: Path) -> None:
    server = build_server(project_dir=tmp_path)

    result = _call_json(server, "get_api_docs", {"path": "tradelearn.data.TradingViewProvider"})

    assert result["path"] == "tradelearn.data.TradingViewProvider"
    assert "history_ohlc" in result["members"]
    assert "tvdatafeed" in result["doc"].lower()


def test_mcp_list_mlflow_runs_uses_configured_tracking_uri(tmp_path: Path) -> None:
    fake = _FakeMLflow()
    server = build_server(
        config=TradelearnConfig(mlflow_tracking_uri="http://mlflow.local"),
        project_dir=tmp_path,
        mlflow_module=fake,
    )

    result = _call_json(
        server,
        "list_mlflow_runs",
        {"experiment": "demo", "max_results": 1},
    )

    assert fake.tracking_uris == ["http://mlflow.local"]
    assert fake.experiments == ["demo"]
    assert result["experiment"] == "demo"
    assert result["runs"] == [
        {
            "run_id": "run-1",
            "status": "FINISHED",
            "start_time": 1,
            "metrics": {"return_pct": 1.2},
            "params": {"fast": "5"},
        }
    ]


def _call_json(server, name: str, arguments: dict[str, object]) -> dict[str, object]:
    result = asyncio.run(server.call_tool(name, arguments))
    if isinstance(result, tuple):
        content, structured = result
        if isinstance(structured, dict):
            return structured
    else:
        content = result
    [content] = content
    return json.loads(content.text)


class _FakeExperiment:
    experiment_id = "exp-1"


class _FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def get_experiment_by_name(self, name: str) -> _FakeExperiment:
        self.experiments.append(name)
        return _FakeExperiment()

    def search_runs(self, *, experiment_ids, max_results, order_by):
        assert experiment_ids == ["exp-1"]
        assert max_results == 1
        assert order_by == ["start_time DESC"]
        return [
            {
                "run_id": "run-1",
                "status": "FINISHED",
                "start_time": 1,
                "metrics.return_pct": 1.2,
                "params.fast": "5",
            }
        ]
