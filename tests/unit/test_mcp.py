from __future__ import annotations

import asyncio
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

    assert {"project_info", "runtime_config", "lab_plan"}.issubset(tool_names)
    assert server.name == "tradelearn"
