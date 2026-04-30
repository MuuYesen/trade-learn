"""Model Context Protocol server for tradelearn project tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from tradelearn import __version__
from tradelearn.core.config import TradelearnConfig, load_config
from tradelearn.lab import build_lab_plan, check_lab_dependencies

MCPTransport = Literal["stdio", "sse", "streamable-http"]


def build_server(
    *,
    config: TradelearnConfig | None = None,
    project_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_level: str | None = None,
) -> FastMCP:
    """Build the tradelearn MCP server.

    The server exposes lightweight project, config, and lab-planning tools.
    It intentionally does not import the backtest runtime, so MCP remains a
    tooling integration layer rather than part of the event-driven engine.
    """

    resolved_project_dir = Path.cwd() if project_dir is None else Path(project_dir)
    resolved_config = config or load_config(project_dir=resolved_project_dir)
    server = FastMCP(
        "tradelearn",
        instructions=(
            "Project tooling for tradelearn. Use these tools to inspect local "
            "configuration, optional lab dependencies, and launch plans."
        ),
        host=host,
        port=port,
        log_level=log_level or resolved_config.log_level,
    )

    @server.tool()
    def project_info(project_dir: str | None = None) -> dict[str, Any]:
        """Return basic local project information."""

        root = Path(project_dir).expanduser().resolve() if project_dir else resolved_project_dir
        return {
            "name": "tradelearn",
            "version": __version__,
            "project_dir": str(root),
            "package_root": str(Path(__file__).resolve().parents[1]),
            "public_modules": [
                "tradelearn.engine",
                "tradelearn.lite",
                "tradelearn.data",
                "tradelearn.indicators",
                "tradelearn.metrics",
                "tradelearn.factor",
                "tradelearn.report",
                "tradelearn.ml",
            ],
        }

    @server.tool()
    def runtime_config(
        project_dir: str | None = None,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """Load and return resolved tradelearn runtime configuration."""

        root = Path(project_dir).expanduser().resolve() if project_dir else resolved_project_dir
        config_file = Path(config_path).expanduser().resolve() if config_path else None
        loaded = load_config(project_dir=root, config_path=config_file)
        return _config_payload(loaded)

    @server.tool()
    def lab_plan(
        host: str = "127.0.0.1",
        port: int = 8888,
        notebook_dir: str = ".",
        no_browser: bool = True,
        mcp_host: str = "127.0.0.1",
        mcp_port: int = 8765,
    ) -> dict[str, Any]:
        """Return the JupyterLab + MCP launch plan without starting processes."""

        plan = build_lab_plan(
            resolved_config,
            host=host,
            port=port,
            notebook_dir=Path(notebook_dir),
            no_browser=no_browser,
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            missing_packages=check_lab_dependencies(),
        )
        return {
            "mlflow_tracking_uri": plan.mlflow_tracking_uri,
            "jupyter_command": list(plan.jupyter.args),
            "mcp_command": list(plan.mcp.args),
            "required_packages": list(plan.required_packages),
            "missing_packages": list(plan.missing_packages),
        }

    return server


def run_server(
    *,
    transport: MCPTransport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    project_dir: Path | str | None = None,
    config_path: Path | str | None = None,
) -> None:
    """Run the tradelearn MCP server."""

    project = Path.cwd() if project_dir is None else Path(project_dir)
    config_file = Path(config_path) if config_path is not None else None
    config = load_config(project_dir=project, config_path=config_file)
    server = build_server(config=config, project_dir=project, host=host, port=port)
    server.run(transport=transport)


def _config_payload(config: TradelearnConfig) -> dict[str, Any]:
    return {
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "data_cache_dir": str(config.data_cache_dir),
        "log_level": config.log_level,
        "data_offline": config.data_offline,
        "cache_ttl_seconds": config.cache_ttl_seconds,
    }
