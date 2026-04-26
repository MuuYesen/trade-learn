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


def test_cli_lab_and_mcp_support_dry_run() -> None:
    lab = runner.invoke(app, ["lab", "--dry-run"])
    mcp = runner.invoke(app, ["mcp", "--dry-run"])

    assert lab.exit_code == 0
    assert "jupyter" in lab.output.lower()
    assert mcp.exit_code == 0
    assert "mcp" in mcp.output.lower()
