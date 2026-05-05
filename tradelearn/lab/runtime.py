"""JupyterLab stack planning and launch helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tradelearn.core.config import TradelearnConfig

LAB_MODULES: dict[str, str] = {
    "jupyterlab": "jupyterlab",
    "jupyterlab-git": "jupyterlab_git",
    "ipywidgets": "ipywidgets",
    "jupyter-ai": "jupyter_ai",
    "pygwalker": "pygwalker",
    "mcp": "mcp",
}

MLFLOW_MODULES: dict[str, str] = {
    "mlflow": "mlflow",
}


@dataclass(frozen=True)
class LabCommand:
    """Resolved command and environment for one lab stack process."""

    args: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class LabPlan:
    """Resolved JupyterLab + MCP + optional MLflow launch plan."""

    jupyter: LabCommand
    mcp: LabCommand
    mlflow: LabCommand | None
    mlflow_tracking_uri: str
    required_packages: tuple[str, ...]
    missing_packages: tuple[str, ...]
    missing_mlflow_packages: tuple[str, ...]


class _Process(Protocol):
    def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def poll(self) -> int | None: ...


def required_lab_packages() -> tuple[str, ...]:
    """Return the packages expected from the ``lab`` extra."""

    return tuple(LAB_MODULES)


def required_mlflow_packages() -> tuple[str, ...]:
    """Return the packages expected from the ``mlflow`` extra."""

    return tuple(MLFLOW_MODULES)


def check_lab_dependencies(
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Return missing lab optional dependencies by package name."""

    return tuple(
        package for package, module in LAB_MODULES.items() if find_spec(module) is None
    )


def check_mlflow_dependencies(
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Return missing MLflow optional dependencies by package name."""

    return tuple(
        package for package, module in MLFLOW_MODULES.items() if find_spec(module) is None
    )


def build_mlflow_command(
    config: TradelearnConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 5050,
    project_dir: Path = Path("."),
    backend_store_uri: str | None = None,
    artifacts_destination: str | None = None,
) -> LabCommand:
    """Build the local MLflow server command used by ``tradelearn mlflow`` and lab."""

    root = Path(project_dir)
    default_db = root / ".tradelearn" / "mlflow" / "mlflow.db"
    default_artifacts = root / ".tradelearn" / "mlflow" / "artifacts"
    resolved_backend = backend_store_uri or f"sqlite:///{default_db}"
    resolved_artifacts = artifacts_destination or f"file://{default_artifacts.resolve()}"
    args = (
        "python",
        "-m",
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        resolved_backend,
        "--serve-artifacts",
        "--artifacts-destination",
        resolved_artifacts,
    )
    return LabCommand(args=args, env=_config_env(config))


def build_lab_plan(
    config: TradelearnConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8888,
    mcp_host: str = "127.0.0.1",
    mcp_port: int = 8765,
    mlflow_host: str = "127.0.0.1",
    mlflow_port: int = 5050,
    notebook_dir: Path = Path("."),
    no_browser: bool = False,
    missing_packages: tuple[str, ...] | None = None,
    missing_mlflow_packages: tuple[str, ...] | None = None,
) -> LabPlan:
    """Build a launch plan for JupyterLab, MCP, and optional local MLflow."""

    runtime_env = _config_env(config)
    jupyter_args = [
        "python",
        "-m",
        "jupyterlab",
        f"--ip={host}",
        f"--port={port}",
        f"--notebook-dir={notebook_dir}",
    ]
    if no_browser:
        jupyter_args.append("--no-browser")
    mlflow_missing = (
        missing_mlflow_packages if missing_mlflow_packages is not None else ()
    )
    mlflow_command = (
        None
        if mlflow_missing
        else build_mlflow_command(
            config,
            host=mlflow_host,
            port=mlflow_port,
            project_dir=notebook_dir,
        )
    )
    return LabPlan(
        jupyter=LabCommand(tuple(jupyter_args), runtime_env),
        mcp=LabCommand(
            (
                "tradelearn",
                "mcp",
                "--transport",
                "streamable-http",
                "--host",
                mcp_host,
                "--port",
                str(mcp_port),
            ),
            runtime_env,
        ),
        mlflow=mlflow_command,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        required_packages=required_lab_packages(),
        missing_packages=missing_packages if missing_packages is not None else (),
        missing_mlflow_packages=mlflow_missing,
    )


def start_lab_stack(
    plan: LabPlan,
    *,
    popen: Callable[..., _Process] = subprocess.Popen,
) -> int:
    """Start optional MLflow, MCP, then JupyterLab; tie services to Jupyter lifetime."""

    processes: list[_Process] = []
    if plan.mlflow is not None:
        _prepare_mlflow_storage(plan.mlflow.args)
        processes.append(popen(plan.mlflow.args, env=_process_env(plan.mlflow.env)))
    mcp_process = popen(plan.mcp.args, env=_process_env(plan.mcp.env))
    processes.append(mcp_process)
    jupyter_process = popen(plan.jupyter.args, env=_process_env(plan.jupyter.env))
    processes.append(jupyter_process)
    try:
        return jupyter_process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    process.terminate()
                    # Wait up to 2 seconds for clean exit
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                except Exception:
                    pass


def start_mlflow_server(
    command: LabCommand,
    *,
    popen: Callable[..., _Process] = subprocess.Popen,
) -> int:
    """Start only the local MLflow server and wait for it to exit."""

    _prepare_mlflow_storage(command.args)
    process = popen(command.args, env=_process_env(command.env))
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def _process_env(overrides: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(overrides)
    return env


def _config_env(config: TradelearnConfig) -> dict[str, str]:
    env = {
        "MLFLOW_TRACKING_URI": config.mlflow_tracking_uri,
        "TRADELEARN_DATA_CACHE_DIR": str(config.data_cache_dir),
        "TRADELEARN_LOG_LEVEL": config.log_level,
        "TRADELEARN_DATA_OFFLINE": "true" if config.data_offline else "false",
    }
    if config.cache_ttl_seconds is not None:
        env["TRADELEARN_CACHE_TTL_SECONDS"] = str(config.cache_ttl_seconds)
    return env


def _prepare_mlflow_storage(args: tuple[str, ...]) -> None:
    if "--backend-store-uri" in args:
        uri = args[args.index("--backend-store-uri") + 1]
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    if "--artifacts-destination" in args:
        uri = args[args.index("--artifacts-destination") + 1]
        if uri.startswith("file://"):
            Path(uri.removeprefix("file://")).mkdir(parents=True, exist_ok=True)


__all__ = [
    "LabCommand",
    "LabPlan",
    "build_lab_plan",
    "build_mlflow_command",
    "check_lab_dependencies",
    "check_mlflow_dependencies",
    "required_lab_packages",
    "required_mlflow_packages",
    "start_lab_stack",
    "start_mlflow_server",
]
