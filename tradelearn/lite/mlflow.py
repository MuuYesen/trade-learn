"""Lite convenience forwarding to the independent MLflow integration."""

from __future__ import annotations

from typing import Any


def log_lite_run(backtest: Any, **options: Any) -> Any:
    """Log a completed Lite run, preserving the native MLflow run lifecycle."""
    if getattr(backtest, "_last_stats", None) is None:
        raise RuntimeError("run() must be called before log_mlflow()")
    from tradelearn.integrations.mlflow import log

    return log(backtest, **options)
