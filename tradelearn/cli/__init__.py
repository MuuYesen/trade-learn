"""Typer command line interface for trade-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from tradelearn import __version__
from tradelearn.core.config import TradelearnConfig, load_config
from tradelearn.lab import build_lab_plan, check_lab_dependencies, start_lab_stack
from tradelearn.mcp import run_server

app = typer.Typer(help="trade-learn research workflow CLI.", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tradelearn {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, help="Show package version."),
    ] = None,
) -> None:
    """Root command."""


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
    lab: Annotated[
        bool,
        typer.Option("--lab", help="Include JupyterLab optional dependency diagnostics."),
    ] = False,
) -> None:
    """Print runtime configuration and basic environment diagnostics."""

    resolved = _load(config)
    typer.echo("tradelearn doctor ok")
    _print_config(resolved)
    if lab:
        missing = check_lab_dependencies()
        typer.echo("lab_required=jupyterlab,jupyterlab-git,ipywidgets,jupyter-ai,pygwalker")
        typer.echo(f"lab_missing={','.join(missing) if missing else 'none'}")


@app.command()
def data(
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
) -> None:
    """Show data-cache configuration."""

    resolved = _load(config)
    typer.echo(f"cache_dir={resolved.data_cache_dir}")
    typer.echo(f"offline={resolved.data_offline}")
    typer.echo(f"cache_ttl_seconds={resolved.cache_ttl_seconds}")


@app.command()
def run(
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print resolved run config.")] = False,
) -> None:
    """Run or inspect a configured backtest."""

    resolved = _load(config)
    if not dry_run:
        typer.echo(
            "Backtest execution requires a strategy module; rerun with --dry-run for config."
        )
        raise typer.Exit(2)
    typer.echo("tradelearn run dry-run")
    _print_config(resolved)


@app.command()
def lab(
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
    host: Annotated[str, typer.Option("--host", help="JupyterLab bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="JupyterLab port.")] = 8888,
    notebook_dir: Annotated[
        Path,
        typer.Option("--notebook-dir", help="Notebook root directory."),
    ] = Path("."),
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Pass --no-browser to JupyterLab."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print commands without starting."),
    ] = False,
    skip_dependency_check: Annotated[
        bool,
        typer.Option("--skip-dependency-check", help="Start even if lab extras are missing."),
    ] = False,
) -> None:
    """Start the lab stack."""

    resolved = _load(config)
    missing = check_lab_dependencies()
    plan = build_lab_plan(
        resolved,
        host=host,
        port=port,
        notebook_dir=notebook_dir,
        no_browser=no_browser,
        missing_packages=missing,
    )
    _print_lab_plan(plan)
    if dry_run:
        return
    if missing and not skip_dependency_check:
        typer.echo("Install lab extras before starting: pip install 'trade-learn[lab]'")
        raise typer.Exit(2)
    raise typer.Exit(start_lab_stack(plan))


@app.command()
def mcp(
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
    transport: Annotated[
        Literal["stdio", "sse", "streamable-http"],
        typer.Option("--transport", help="MCP transport."),
    ] = "stdio",
    host: Annotated[str, typer.Option("--host", help="MCP HTTP bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="MCP HTTP port.")] = 8765,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print command without starting."),
    ] = False,
) -> None:
    """Start the MCP server entrypoint."""

    if dry_run:
        typer.echo(
            "tradelearn mcp "
            f"--transport {transport} --host {host} --port {port}"
            + (f" --config {config}" if config else "")
        )
        return
    run_server(
        transport=transport,
        host=host,
        port=port,
        project_dir=Path.cwd(),
        config_path=config,
    )


@app.command()
def new(
    name: str,
    directory: Annotated[
        Path,
        typer.Option("--directory", "-d", help="Parent directory for the new project."),
    ] = Path("."),
) -> None:
    """Create a minimal trade-learn project skeleton."""

    project = directory / name
    config_dir = project / ".tradelearn"
    notebooks = project / "notebooks"
    config_dir.mkdir(parents=True, exist_ok=False)
    notebooks.mkdir(parents=True, exist_ok=False)
    (config_dir / "config.yaml").write_text(_starter_config(), encoding="utf-8")
    (project / "strategy.py").write_text(_starter_strategy(), encoding="utf-8")
    for filename, title in _starter_notebooks().items():
        (notebooks / filename).write_text(_starter_notebook(title), encoding="utf-8")
    typer.echo(f"Created tradelearn project: {project}")


def main() -> None:
    """Console script entrypoint."""

    app()


def _load(config: Path | None) -> TradelearnConfig:
    return load_config(project_dir=Path.cwd(), config_path=config)


def _print_config(config: TradelearnConfig) -> None:
    typer.echo(f"mlflow_tracking_uri={config.mlflow_tracking_uri}")
    typer.echo(f"data_cache_dir={config.data_cache_dir}")
    typer.echo(f"log_level={config.log_level}")
    typer.echo(f"offline={config.data_offline}")
    typer.echo(f"cache_ttl_seconds={config.cache_ttl_seconds}")


def _print_lab_plan(plan) -> None:
    typer.echo(f"mlflow_tracking_uri={plan.mlflow_tracking_uri}")
    typer.echo(f"mcp_command={' '.join(plan.mcp.args)}")
    typer.echo(f"jupyter_command={' '.join(plan.jupyter.args)}")
    typer.echo(f"lab_required={','.join(plan.required_packages)}")
    missing = ",".join(plan.missing_packages) if plan.missing_packages else "none"
    typer.echo(f"lab_missing={missing}")


def _starter_config() -> str:
    return (
        "mlflow:\n"
        "  tracking_uri: https://mlflow.leafquant.com\n"
        "data:\n"
        "  cache_dir: ./data\n"
        "  offline: false\n"
        "  cache_ttl_seconds: 86400\n"
        "log_level: INFO\n"
    )


def _starter_strategy() -> str:
    return (
        "from tradelearn.engine import Strategy\n\n\n"
        "class DemoStrategy(Strategy):\n"
        "    params = ((\"size\", 1),)\n\n"
        "    def next(self) -> None:\n"
        "        if not self.position:\n"
        "            self.buy(size=self.p.size)\n"
    )


def _starter_notebooks() -> dict[str, str]:
    return {
        "01_explore.ipynb": "Explore data",
        "02_factor.ipynb": "Factor research",
        "03_backtest.ipynb": "Backtest workflow",
    }


def _starter_notebook(title: str) -> str:
    return (
        "{\n"
        "  \"cells\": [\n"
        "    {\n"
        "      \"cell_type\": \"markdown\",\n"
        "      \"metadata\": {},\n"
        "      \"source\": [\"# "
        + title
        + "\\n\"]\n"
        "    }\n"
        "  ],\n"
        "  \"metadata\": {\n"
        "    \"kernelspec\": {\n"
        "      \"display_name\": \"Python 3\",\n"
        "      \"language\": \"python\",\n"
        "      \"name\": \"python3\"\n"
        "    },\n"
        "    \"language_info\": {\"name\": \"python\"}\n"
        "  },\n"
        "  \"nbformat\": 4,\n"
        "  \"nbformat_minor\": 5\n"
        "}\n"
    )
