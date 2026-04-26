from __future__ import annotations

from pathlib import Path

from tradelearn.core.config import TradelearnConfig
from tradelearn.lab import (
    build_lab_plan,
    check_lab_dependencies,
    required_lab_packages,
    start_lab_stack,
)


def test_check_lab_dependencies_reports_missing_packages() -> None:
    missing = check_lab_dependencies(
        find_spec=lambda module: object() if module in {"jupyterlab", "ipywidgets"} else None
    )

    assert missing == ("jupyterlab-git", "jupyter-ai", "pygwalker")


def test_build_lab_plan_sets_commands_and_mlflow_environment(tmp_path: Path) -> None:
    config = TradelearnConfig(mlflow_tracking_uri="http://mlflow.local")

    plan = build_lab_plan(
        config,
        host="0.0.0.0",
        port=9999,
        notebook_dir=tmp_path,
        no_browser=True,
    )

    assert plan.mlflow_tracking_uri == "http://mlflow.local"
    assert plan.required_packages == tuple(required_lab_packages())
    assert plan.jupyter.args[:3] == ("python", "-m", "jupyterlab")
    assert "--ip=0.0.0.0" in plan.jupyter.args
    assert "--port=9999" in plan.jupyter.args
    assert f"--notebook-dir={tmp_path}" in plan.jupyter.args
    assert "--no-browser" in plan.jupyter.args
    assert plan.jupyter.env["MLFLOW_TRACKING_URI"] == "http://mlflow.local"
    assert plan.mcp.args == ("tradelearn", "mcp", "--dry-run")
    assert plan.mcp.env["MLFLOW_TRACKING_URI"] == "http://mlflow.local"


def test_start_lab_stack_starts_mcp_then_jupyter() -> None:
    started: list[tuple[str, ...]] = []
    terminated: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, args: tuple[str, ...], env: dict[str, str]) -> None:
            self.args = args
            self.env = env
            started.append(args)

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            terminated.append(self.args)

        def poll(self) -> None:
            return None

    def fake_popen(args: tuple[str, ...], *, env: dict[str, str]) -> FakeProcess:
        return FakeProcess(args, env)

    plan = build_lab_plan(TradelearnConfig(), notebook_dir=Path("."))

    exit_code = start_lab_stack(plan, popen=fake_popen)

    assert exit_code == 0
    assert started[0] == ("tradelearn", "mcp", "--dry-run")
    assert started[1][:3] == ("python", "-m", "jupyterlab")
    assert terminated == [("tradelearn", "mcp", "--dry-run")]
