"""JupyterLab launch only; experiment tracking uses its existing configuration."""

from __future__ import annotations

import importlib.util
import os
import signal
import threading
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradelearn.core.config import TradelearnConfig

LAB_MODULES = {
    "jupyterlab": "jupyterlab",
    "ipywidgets": "ipywidgets",
    "jupyter-ai-lite": "jupyternaut_lite",
}


@dataclass(frozen=True)
class LabCommand:
    args: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class LabPlan:
    jupyter: LabCommand
    mlflow_tracking_uri: str
    required_packages: tuple[str, ...]
    missing_packages: tuple[str, ...]


def required_lab_packages() -> tuple[str, ...]:
    return tuple(LAB_MODULES)


def check_lab_dependencies(
    *, find_spec: Callable[[str], Any] = importlib.util.find_spec
) -> tuple[str, ...]:
    return tuple(package for package, module in LAB_MODULES.items() if find_spec(module) is None)


def build_lab_plan(
    config: TradelearnConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8888,
    notebook_dir: Path = Path("."),
    no_browser: bool = False,
    missing_packages: tuple[str, ...] = (),
) -> LabPlan:
    args = (
        sys.executable,
        "-m",
        "jupyterlab",
        f"--ip={host}",
        f"--port={port}",
        f"--notebook-dir={notebook_dir}",
    )
    if no_browser:
        args += ("--no-browser",)
    env = {
        "MLFLOW_TRACKING_URI": config.mlflow_tracking_uri,
        "TRADELEARN_DATA_CACHE_DIR": str(config.data_cache_dir),
        "TRADELEARN_LOG_LEVEL": config.log_level,
        "TRADELEARN_DATA_OFFLINE": "true" if config.data_offline else "false",
    }
    if config.cache_ttl_seconds is not None:
        env["TRADELEARN_CACHE_TTL_SECONDS"] = str(config.cache_ttl_seconds)
    return LabPlan(
        LabCommand(args, env), config.mlflow_tracking_uri, required_lab_packages(), missing_packages
    )


def start_lab_stack(plan: LabPlan, *, popen: Callable[..., Any] = subprocess.Popen) -> int:
    process = popen(plan.jupyter.args, env={**os.environ, **plan.jupyter.env})
    terminated = False

    def stop(signum, frame):
        nonlocal terminated
        terminated = True
        raise KeyboardInterrupt

    owns_handler = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, stop) if owns_handler else None
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 128 + signal.SIGTERM if terminated else 0
    finally:
        if owns_handler:
            signal.signal(signal.SIGTERM, previous)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
