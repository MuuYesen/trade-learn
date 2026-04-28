from typing import Any

import numpy as np
import pandas as pd


class MetricsEngine:
    """Centralized engine for vectorized metric calculation."""

    def __init__(self) -> None:
        # Maps instance_id to {'metric': metric_key, 'params': params_dict}
        self.requested_instances: dict[str, dict[str, Any]] = {}
        # Caches computed results by instance_id
        self.results: dict[str, Any] = {}

    def request(self, instance_id: str, metric_key: str, params: Any = None) -> None:
        """Register a metric instance to be calculated at the end of the backtest."""
        if instance_id not in self.requested_instances:
            param_dict = params.asdict() if hasattr(params, "asdict") else (params or {})
            self.requested_instances[instance_id] = {
                "metric": metric_key,
                "params": param_dict,
                "raw_params": params,
            }

    def compute(self, stats: Any) -> None:
        """Compute all requested metrics using a basic DAG (base data -> composite metrics)."""
        # --- Base Data Extraction (Tier 1) ---
        returns = stats.returns if hasattr(stats, "returns") else pd.Series(dtype=float)
        equity = stats.equity if hasattr(stats, "equity") else pd.Series(dtype=float)

        clean_rets = returns.dropna() if not returns.empty else pd.Series(dtype=float)
        clean_equity = equity.dropna() if not equity.empty else pd.Series(dtype=float)

        # --- Base Computations Cache (Tier 2) ---
        base_cache: dict[str, Any] = {}

        if not clean_equity.empty:
            peak = clean_equity.cummax()
            base_cache["drawdowns"] = (peak - clean_equity) / peak

        # --- Instance Computations (Tier 3) ---
        for inst_id, req in self.requested_instances.items():
            metric = req["metric"]
            params_raw = req["raw_params"]

            if metric == "returns":
                total = float(np.prod(1.0 + clean_rets) - 1.0) if not clean_rets.empty else 0.0
                average = float(np.mean(clean_rets)) if not clean_rets.empty else 0.0
                self.results[inst_id] = {"total": total, "average": average}

            elif metric == "sharpe":
                req_params = req["params"]
                if hasattr(params_raw, "riskfreerate"):
                    rf = params_raw.riskfreerate
                elif isinstance(req_params, dict):
                    rf = req_params.get("riskfreerate", 0.0)
                else:
                    rf = 0.0
                from tradelearn.metrics import sharpe

                self.results[inst_id] = {"sharperatio": sharpe(returns, rf=rf, periods=252)}

            elif metric == "drawdown":
                if "drawdowns" in base_cache and not base_cache["drawdowns"].empty:
                    max_dd = float(base_cache["drawdowns"].max())
                    current_dd = float(base_cache["drawdowns"].iloc[-1])
                else:
                    max_dd, current_dd = 0.0, 0.0

                self.results[inst_id] = {
                    "drawdown": current_dd,
                    "maxdrawdown": max_dd,
                }
