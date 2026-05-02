from __future__ import annotations

from pathlib import Path

from tradelearn.core.config import TradelearnConfig
from tradelearn.lab import (
    build_mlflow_command,
    build_lab_plan,
    check_lab_dependencies,
    check_mlflow_dependencies,
    required_lab_packages,
    required_mlflow_packages,
    start_lab_stack,
    start_mlflow_server,
)


def test_check_lab_dependencies_reports_missing_packages() -> None:
    missing = check_lab_dependencies(
        find_spec=lambda module: object()
        if module in {"jupyterlab", "ipywidgets", "mcp"}
        else None
    )

    assert missing == ("jupyterlab-git", "jupyter-ai", "pygwalker")


def test_check_mlflow_dependencies_reports_missing_package() -> None:
    missing = check_mlflow_dependencies(find_spec=lambda module: None)

    assert missing == ("mlflow",)
    assert required_mlflow_packages() == ("mlflow",)


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
    assert plan.mlflow is not None
    assert plan.mlflow.args[:4] == ("python", "-m", "mlflow", "server")
    assert "--port" in plan.mlflow.args
    assert "5050" in plan.mlflow.args
    assert plan.jupyter.args[:3] == ("python", "-m", "jupyterlab")
    assert "--ip=0.0.0.0" in plan.jupyter.args
    assert "--port=9999" in plan.jupyter.args
    assert f"--notebook-dir={tmp_path}" in plan.jupyter.args
    assert "--no-browser" in plan.jupyter.args
    assert plan.jupyter.env["MLFLOW_TRACKING_URI"] == "http://mlflow.local"
    assert plan.mcp.args == (
        "tradelearn",
        "mcp",
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    )
    assert plan.mcp.env["MLFLOW_TRACKING_URI"] == "http://mlflow.local"
    assert plan.mcp.env["TRADELEARN_LOG_LEVEL"] == "INFO"


def test_build_lab_plan_skips_mlflow_when_missing(tmp_path: Path) -> None:
    missing_plan = build_lab_plan(
        TradelearnConfig(),
        notebook_dir=tmp_path,
        missing_mlflow_packages=("mlflow",),
    )

    assert missing_plan.mlflow is None
    assert missing_plan.missing_mlflow_packages == ("mlflow",)


def test_build_mlflow_command_uses_local_sqlite_and_artifacts(tmp_path: Path) -> None:
    command = build_mlflow_command(
        TradelearnConfig(mlflow_tracking_uri="http://127.0.0.1:5050"),
        project_dir=tmp_path,
        host="0.0.0.0",
        port=5051,
    )

    assert command.args[:4] == ("python", "-m", "mlflow", "server")
    assert command.args[command.args.index("--host") + 1] == "0.0.0.0"
    assert command.args[command.args.index("--port") + 1] == "5051"
    backend = command.args[command.args.index("--backend-store-uri") + 1]
    artifacts = command.args[command.args.index("--artifacts-destination") + 1]
    assert backend == f"sqlite:///{tmp_path / '.tradelearn' / 'mlflow' / 'mlflow.db'}"
    assert artifacts == f"file://{(tmp_path / '.tradelearn' / 'mlflow' / 'artifacts').resolve()}"


def test_start_lab_stack_starts_mlflow_mcp_then_jupyter() -> None:
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
    assert started[0][:4] == ("python", "-m", "mlflow", "server")
    assert started[1][:5] == ("tradelearn", "mcp", "--transport", "streamable-http", "--host")
    assert started[2][:3] == ("python", "-m", "jupyterlab")
    assert terminated == [started[1], started[0]]


def test_start_mlflow_server_waits_for_process() -> None:
    started: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, args: tuple[str, ...], env: dict[str, str]) -> None:
            self.args = args
            self.env = env
            started.append(args)

        def wait(self) -> int:
            return 7

        def terminate(self) -> None:
            raise AssertionError("single MLflow command should wait, not terminate")

        def poll(self) -> None:
            return 7

    command = build_mlflow_command(TradelearnConfig(), project_dir=Path("."))

    exit_code = start_mlflow_server(command, popen=lambda args, *, env: FakeProcess(args, env))

    assert exit_code == 7
    assert started == [command.args]
