"""Model Context Protocol server for tradelearn project tooling."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Literal

import pandas as pd
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
    mlflow_module: Any | None = None,
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

    @server.tool()
    def search_api(query: str, limit: int = 20) -> dict[str, Any]:
        """Search user-facing tradelearn API symbols by name."""

        needle = query.strip().lower()
        if not needle:
            raise ValueError("query must not be empty")
        results: list[dict[str, Any]] = []
        for module_name in _public_modules():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for name in getattr(module, "__all__", ()):
                path = f"{module_name}.{name}"
                if needle not in name.lower() and needle not in path.lower():
                    continue
                try:
                    obj = getattr(module, name)
                except Exception:
                    obj = None
                results.append(
                    {
                        "name": name,
                        "path": path,
                        "module": module_name,
                        "kind": _api_kind(obj),
                        "summary": _doc_summary(obj),
                    }
                )
                if len(results) >= int(limit):
                    return {"query": query, "results": results}
        return {"query": query, "results": results}

    @server.tool()
    def get_api_docs(path: str) -> dict[str, Any]:
        """Return signature, docstring, and public members for one API object."""

        obj = _resolve_object(path)
        members: list[str] = []
        if inspect.isclass(obj):
            members = [
                name
                for name, value in inspect.getmembers(obj)
                if not name.startswith("_") and (inspect.isfunction(value) or inspect.ismethod(value))
            ]
        return {
            "path": path,
            "kind": _api_kind(obj),
            "signature": _signature(obj),
            "doc": inspect.getdoc(obj) or "",
            "members": members,
        }

    @server.tool()
    def list_mlflow_runs(
        experiment: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """List recent MLflow runs for an experiment."""

        mlflow = mlflow_module or _import_mlflow()
        mlflow.set_tracking_uri(resolved_config.mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return {"experiment": experiment, "runs": []}
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=int(max_results),
            order_by=["start_time DESC"],
        )
        return {"experiment": experiment, "runs": _mlflow_runs_payload(runs)}

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


def _public_modules() -> tuple[str, ...]:
    return (
        "tradelearn.engine",
        "tradelearn.lite",
        "tradelearn.data",
        "tradelearn.indicators",
        "tradelearn.metrics",
        "tradelearn.factor",
        "tradelearn.report",
        "tradelearn.ml",
    )


def _resolve_object(path: str) -> Any:
    parts = path.split(".")
    if len(parts) < 2 or parts[0] != "tradelearn":
        raise ValueError("path must be a tradelearn dotted path")
    errors: list[Exception] = []
    for index in range(len(parts), 1, -1):
        module_name = ".".join(parts[:index])
        try:
            obj = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(exc)
            continue
        for attr in parts[index:]:
            obj = getattr(obj, attr)
        return obj
    raise ValueError(f"Could not resolve API path: {path}") from (errors[-1] if errors else None)


def _api_kind(obj: Any) -> str:
    if obj is None:
        return "unknown"
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return "function"
    return type(obj).__name__


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _doc_summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) if obj is not None else ""
    if not doc:
        return ""
    return doc.splitlines()[0]


def _import_mlflow() -> Any:
    import mlflow

    return mlflow


def _mlflow_runs_payload(runs: Any) -> list[dict[str, Any]]:
    if isinstance(runs, pd.DataFrame):
        rows = runs.to_dict(orient="records")
    else:
        rows = list(runs)
    return [_normalize_mlflow_run(row) for row in rows]


def _normalize_mlflow_run(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = dict(row)
    metrics = {
        key.removeprefix("metrics."): value
        for key, value in row.items()
        if str(key).startswith("metrics.")
    }
    params = {
        key.removeprefix("params."): value
        for key, value in row.items()
        if str(key).startswith("params.")
    }
    return {
        "run_id": row.get("run_id") or row.get("run.info.run_id"),
        "status": row.get("status") or row.get("run.info.status"),
        "start_time": row.get("start_time") or row.get("run.info.start_time"),
        "metrics": metrics,
        "params": params,
    }
