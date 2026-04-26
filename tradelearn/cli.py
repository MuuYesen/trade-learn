"""Typer command line interface for trade-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tradelearn import __version__
from tradelearn.core.config import TradelearnConfig, load_config

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
) -> None:
    """Print runtime configuration and basic environment diagnostics."""

    resolved = _load(config)
    typer.echo("tradelearn doctor ok")
    _print_config(resolved)


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
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print commands without starting."),
    ] = False,
) -> None:
    """Start the lab stack."""

    if not dry_run:
        typer.echo("JupyterLab startup is scheduled for Stage 5; use --dry-run in Stage 4.")
        raise typer.Exit(2)
    typer.echo("Would start jupyter lab and tradelearn mcp server")


@app.command()
def mcp(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print command without starting."),
    ] = False,
) -> None:
    """Start the MCP server entrypoint."""

    if not dry_run:
        typer.echo("MCP server implementation is scheduled for Stage 7; use --dry-run in Stage 4.")
        raise typer.Exit(2)
    typer.echo("Would start tradelearn mcp server")


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
    config_dir.mkdir(parents=True, exist_ok=False)
    (config_dir / "config.yaml").write_text(_starter_config(), encoding="utf-8")
    (project / "strategy.py").write_text(_starter_strategy(), encoding="utf-8")
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
        "from tradelearn.backtest import Strategy\n\n\n"
        "class DemoStrategy(Strategy):\n"
        "    params = ((\"size\", 1),)\n\n"
        "    def next(self) -> None:\n"
        "        if not self.position:\n"
        "            self.buy(size=self.p.size)\n"
    )
