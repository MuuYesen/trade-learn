from __future__ import annotations

from typer.testing import CliRunner

from tradelearn.cli import app

runner = CliRunner()


def test_cli_version_option() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "tradelearn" in result.output


def test_cli_exposes_stage4_command_surface() -> None:
    for command in ("doctor", "lab", "mlflow", "mcp", "new"):
        result = runner.invoke(app, [command, "--help"])

        assert result.exit_code == 0
        assert command in result.output


def test_cli_new_creates_project_skeleton(tmp_path) -> None:
    result = runner.invoke(app, ["new", "demo", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    project = tmp_path / "demo"
    assert (project / ".tradelearn" / "config.yaml").exists()
    strategy = project / "strategy.py"
    assert strategy.exists()
    text = strategy.read_text(encoding="utf-8")
    assert "import tradelearn.engine as bt" in text
    assert "class DemoStrategy(bt.Strategy):" in text
    assert (project / "notebooks" / "01_explore.ipynb").exists()
    assert (project / "notebooks" / "02_factor.ipynb").exists()
    assert (project / "notebooks" / "03_backtest.ipynb").exists()


def test_cli_removed_data_and_run_commands() -> None:
    data_result = runner.invoke(app, ["data", "--help"])
    run_result = runner.invoke(app, ["run", "--help"])

    assert data_result.exit_code != 0
    assert run_result.exit_code == 2
    assert "No such command" in data_result.output
    assert "No such command" in run_result.output


def test_cli_lab_help_has_no_preview_option() -> None:
    result = runner.invoke(app, ["lab", "--help"])

    assert result.exit_code == 0
    assert "--dry" + "-run" not in result.output
    assert "--skip-dependency-check" not in result.output
    assert "--mlflow-port" in result.output


def test_cli_mcp_help_has_no_preview_option() -> None:
    result = runner.invoke(app, ["mcp", "--help"])

    assert result.exit_code == 0
    assert "--dry" + "-run" not in result.output
    assert "--transport" in result.output


def test_cli_lab_dependency_check_prevents_starting_when_missing(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "mlflow:\n  tracking_uri: http://mlflow.local\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "lab",
            "--config",
            str(config),
            "--port",
            "9999",
            "--no-browser",
        ],
    )

    assert result.exit_code == 2
    assert "jupyter" in result.output.lower()
    assert "tradelearn mcp --transport streamable-http" in result.output
    assert "python -m mlflow server" in result.output
    assert "http://mlflow.local" in result.output
    assert "--port=9999" in result.output
    assert "Install lab extras" in result.output


def test_cli_mlflow_help_has_no_preview_option() -> None:
    result = runner.invoke(
        app,
        [
            "mlflow",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--dry" + "-run" not in result.output
    assert "--skip-dependency-check" not in result.output
    assert "--backend-store-uri" in result.output


def test_cli_doctor_reports_lab_dependency_status(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["doctor", "--lab", "--mlflow", "--config", str(tmp_path / "missing.yaml")],
    )

    assert result.exit_code == 0
    assert "lab_required=" in result.output
    assert "lab_missing=" in result.output
    assert "mlflow_required=mlflow" in result.output
    assert "mlflow_missing=" in result.output
