from __future__ import annotations

from typer.testing import CliRunner

from tradelearn.cli import app

runner = CliRunner()


def test_cli_version_option() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "tradelearn" in result.output


def test_cli_exposes_stage4_command_surface() -> None:
    for command in ("lab", "new", "data", "run", "mcp", "doctor"):
        result = runner.invoke(app, [command, "--help"])

        assert result.exit_code == 0
        assert command in result.output


def test_cli_new_creates_project_skeleton(tmp_path) -> None:
    result = runner.invoke(app, ["new", "demo", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    project = tmp_path / "demo"
    assert (project / ".tradelearn" / "config.yaml").exists()
    assert (project / "strategy.py").exists()
    assert (project / "notebooks" / "01_explore.ipynb").exists()
    assert (project / "notebooks" / "02_factor.ipynb").exists()
    assert (project / "notebooks" / "03_backtest.ipynb").exists()


def test_cli_data_and_run_use_config_file(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n  cache_dir: custom-cache\n  offline: true\n"
        "mlflow:\n  tracking_uri: http://mlflow.local\n",
        encoding="utf-8",
    )

    data_result = runner.invoke(app, ["data", "--config", str(config)])
    run_result = runner.invoke(app, ["run", "--config", str(config), "--dry-run"])

    assert data_result.exit_code == 0
    assert "custom-cache" in data_result.output
    assert "offline=True" in data_result.output
    assert run_result.exit_code == 0
    assert "http://mlflow.local" in run_result.output


def test_cli_lab_and_mcp_support_dry_run(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "mlflow:\n  tracking_uri: http://mlflow.local\n",
        encoding="utf-8",
    )

    lab = runner.invoke(
        app,
        [
            "lab",
            "--dry-run",
            "--config",
            str(config),
            "--port",
            "9999",
            "--no-browser",
        ],
    )
    mcp = runner.invoke(app, ["mcp", "--dry-run"])

    assert lab.exit_code == 0
    assert "jupyter" in lab.output.lower()
    assert "tradelearn mcp --transport streamable-http" in lab.output
    assert "http://mlflow.local" in lab.output
    assert "--port=9999" in lab.output
    assert mcp.exit_code == 0
    assert "tradelearn mcp --transport stdio" in mcp.output


def test_cli_doctor_reports_lab_dependency_status(tmp_path) -> None:
    result = runner.invoke(app, ["doctor", "--lab", "--config", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 0
    assert "lab_required=" in result.output
    assert "lab_missing=" in result.output
