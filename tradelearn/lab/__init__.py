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
}


@dataclass(frozen=True)
class LabCommand:
    """Resolved command and environment for one lab stack process."""

    args: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class LabPlan:
    """Resolved JupyterLab + MCP launch plan."""

    jupyter: LabCommand
    mcp: LabCommand
    mlflow_tracking_uri: str
    required_packages: tuple[str, ...]
    missing_packages: tuple[str, ...]


class _Process(Protocol):
    def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def poll(self) -> int | None: ...


def required_lab_packages() -> tuple[str, ...]:
    """Return the packages expected from the ``lab`` extra."""

    return tuple(LAB_MODULES)


def check_lab_dependencies(
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Return missing lab optional dependencies by package name."""

    return tuple(
        package for package, module in LAB_MODULES.items() if find_spec(module) is None
    )


def build_lab_plan(
    config: TradelearnConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8888,
    mcp_host: str = "127.0.0.1",
    mcp_port: int = 8765,
    notebook_dir: Path = Path("."),
    no_browser: bool = False,
    missing_packages: tuple[str, ...] | None = None,
) -> LabPlan:
    """Build a launch plan for JupyterLab and the tradelearn MCP entrypoint."""

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
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        required_packages=required_lab_packages(),
        missing_packages=missing_packages if missing_packages is not None else (),
    )


def start_lab_stack(
    plan: LabPlan,
    *,
    popen: Callable[..., _Process] = subprocess.Popen,
) -> int:
    """Start MCP first, then JupyterLab, and keep MCP tied to Jupyter lifetime."""

    mcp_process = popen(plan.mcp.args, env=_process_env(plan.mcp.env))
    try:
        jupyter_process = popen(plan.jupyter.args, env=_process_env(plan.jupyter.env))
        return jupyter_process.wait()
    finally:
        if mcp_process.poll() is None:
            mcp_process.terminate()


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
