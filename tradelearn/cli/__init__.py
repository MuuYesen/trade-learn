"""Minimal local research CLI; optional components are loaded only when used."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

from tradelearn import __version__
from tradelearn.core.config import load_config
from tradelearn.core.errors import ConfigurationError

# Inspect installed distribution metadata without importing optional runtimes,
# connecting to providers, or reading credentials.
COMPONENTS = (
    "numpy",
    "pandas",
    "TA-Lib",
    "jupyterlab",
    "ipywidgets",
    "jupyter-ai-lite",
    "mlflow",
    "duckdb",
    "scikit-learn",
    "Riskfolio-Lib",
    "optuna",
    "gplearn",
    "causal-learn",
    "pynesys-pynecore",
    "KunQuant",
    "tvkit",
    "nautilus_trader",
    "tl-store",
)


def _doctor(config_path: Path | None) -> int:
    config = load_config(project_dir=Path.cwd(), config_path=config_path)
    print(f"python={platform.python_version()}")
    print(f"trade-learn={__version__}")
    print("configuration=valid")
    print(
        "mlflow_tracking="
        + ("local" if config.mlflow_tracking_uri.startswith(("sqlite:", "file:")) else "explicit")
    )
    missing_base = False
    for package in COMPONENTS:
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_base |= package in {"numpy", "pandas", "TA-Lib"}
            value = (
                "not installed (optional)"
                if package not in {"numpy", "pandas", "TA-Lib"}
                else "MISSING (base requirement)"
            )
        print(f"{package}={value}")
    print("tradelearn doctor " + ("failed (missing base requirement)" if missing_base else "ok"))
    return 1 if missing_base else 0


def _new(name: str, directory: Path) -> None:
    if name in {".", "..", ""} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("project name must be a single directory name")
    project = directory / name
    project.mkdir(parents=True, exist_ok=False)
    notebooks = project / "notebooks"
    notebooks.mkdir()
    # _agents is supported by AI Lite and visible to Jupyter Contents with its
    # default hidden-file policy. Do not enable access to all hidden files.
    skills = project / "_agents" / "skills" / "tradelearn-research"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(_starter_skill(), encoding="utf-8")
    (project / "strategy.py").write_text(_starter_strategy(), encoding="utf-8")
    for filename, title in _starter_notebooks().items():
        (notebooks / filename).write_text(_starter_notebook(title), encoding="utf-8")
    (project / "README.md").write_text(
        "# " + name + "\n\nstrategy.py defines an example Strategy; "
        "import it into Cerebro with your own Bars.\n"
        "Start `tradelearn lab --notebook-dir .` from this directory. No YAML is required.\n"
        "AI Lite discovers _agents/skills. MLflow uses local sqlite:///mlflow.db "
        "by default; no server is started.\n",
        encoding="utf-8",
    )
    print(f"Created tradelearn project: {project}")


def main() -> None:
    parser = argparse.ArgumentParser(description="trade-learn local research CLI")
    parser.add_argument("--version", action="version", version=f"tradelearn {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="Check configuration and all installed components")
    doctor.add_argument("--config", type=Path)
    new = commands.add_parser("new", help="Create a minimal project and version-matched Skills")
    new.add_argument("name")
    new.add_argument("--directory", "-d", type=Path, default=Path("."))
    lab = commands.add_parser("lab", help="Start only JupyterLab")
    lab.add_argument("--config", type=Path)
    lab.add_argument("--host", default="127.0.0.1")
    lab.add_argument("--port", type=int, default=8888)
    lab.add_argument("--notebook-dir", type=Path, default=Path("."))
    lab.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            raise SystemExit(_doctor(args.config))
        elif args.command == "new":
            _new(args.name, args.directory)
        elif args.command == "lab":
            from tradelearn.lab import build_lab_plan, check_lab_dependencies, start_lab_stack

            missing = check_lab_dependencies()
            if missing:
                parser.error("Missing " + ", ".join(missing) + "; install trade-learn[research]")
            config = load_config(project_dir=args.notebook_dir, config_path=args.config)
            plan = build_lab_plan(
                config,
                host=args.host,
                port=args.port,
                notebook_dir=args.notebook_dir,
                no_browser=args.no_browser,
            )
            raise SystemExit(start_lab_stack(plan))
        else:
            parser.print_help()
    except (ConfigurationError, ValueError, OSError) as exc:
        parser.error(str(exc))


def _starter_strategy() -> str:
    return (
        "import tradelearn.engine as bt\n\n\n"
        "class DemoStrategy(bt.Strategy):\n"
        '    params = (("size", 1),)\n\n'
        "    def next(self) -> None:\n"
        "        if not self.position:\n"
        "            self.buy(size=self.p.size)\n"
    )


def _starter_skill() -> str:
    """Provide discoverable, version-matched research API guidance."""
    return f"""---
name: tradelearn-research
description: Use for local TradeLearn research notebooks, reports and experiment logging.
---

Target package: trade-learn {__version__}. Check the installed version before using this reference.
Keep Bars/features as Pandas objects. Split time and purge using actual label_end before fitting.
Define components before execution; explicitly register them and record actual calls in ResearchRun:

```python
from tradelearn.research import ResearchRun
with ResearchRun("validation") as run:
    run.register("model", model)
    run.call("model", "fit", X_train, y_train)
```

Use `tradelearn.lite.Backtest` for convenience or Backtrader-shaped `tradelearn.engine` for control.
A report consumes completed facts. For a completed strategy result:

```python
from tradelearn.report import Reporter
Reporter(result).report("report.html")
from tradelearn.integrations.mlflow import log
logged = log(result, research_result=research_result)
```

Use Reporter.from_analysis for completed domain tables/metrics/metadata.
Do not fit or refetch during rendering.
MLflow records only supplied results. Reuse compatible native active runs.
Do not start nested runs automatically.
Model publication and attachments are explicit.
Preserve fitted Pipeline/FeatureSet/schema/versions for restoration.
Work on local research inputs under the user's task authorization.
Do not start live trading or paid provider calls from this skill.
"""


def _starter_notebooks() -> dict[str, str]:
    return {
        "01_explore.ipynb": "Explore data",
        "02_factor.ipynb": "Factor research",
        "03_backtest.ipynb": "Backtest workflow",
    }


def _starter_notebook(title: str) -> str:
    return (
        "{\n"
        '  "cells": [\n'
        "    {\n"
        '      "cell_type": "markdown",\n'
        '      "metadata": {},\n'
        '      "source": ["# ' + title + '\\n"]\n'
        "    }\n"
        "  ],\n"
        '  "metadata": {\n'
        '    "kernelspec": {\n'
        '      "display_name": "Python 3",\n'
        '      "language": "python",\n'
        '      "name": "python3"\n'
        "    },\n"
        '    "language_info": {"name": "python"}\n'
        "  },\n"
        '  "nbformat": 4,\n'
        '  "nbformat_minor": 5\n'
        "}\n"
    )
