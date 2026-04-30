"""Pyfolio-style factor risk and performance attribution helpers.

These utilities intentionally model the clean mathematical interface instead
of a commercial Barra data product.  Users can pass Barra-like exposures,
factor returns, covariance, and specific risk when those data are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRiskModel:
    """Static or dated multi-factor risk model.

    Parameters
    ----------
    exposures
        Asset factor exposure matrix.  Use a symbol index for a static model,
        or a ``(date, symbol)`` MultiIndex for dated exposures.
    factor_cov
        Factor covariance matrix with factor names as index and columns.
    specific_var
        Asset-specific variance by symbol.  Missing symbols default to zero.
    """

    exposures: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series | None = None

    def portfolio_exposure(self, weights: pd.Series | dict[str, float], date=None) -> pd.Series:
        """Return weighted factor exposure for a portfolio."""

        w = _coerce_series(weights, name="weight")
        exposure = _exposure_slice(self.exposures, date)
        aligned = exposure.reindex(w.index).fillna(0.0)
        result = aligned.mul(w, axis=0).sum(axis=0)
        result.name = "exposure"
        return result.astype("float64")

    def portfolio_variance(self, weights: pd.Series | dict[str, float], date=None) -> float:
        """Return total portfolio variance from factor and specific risk."""

        w = _coerce_series(weights, name="weight")
        beta = self.portfolio_exposure(w, date=date).reindex(self.factor_cov.index).fillna(0.0)
        cov = self.factor_cov.reindex(index=beta.index, columns=beta.index).fillna(0.0)
        factor_var = float(beta.to_numpy() @ cov.to_numpy() @ beta.to_numpy())
        specific = _specific_var(self.specific_var, w.index)
        specific_var = float((w.pow(2) * specific).sum())
        return factor_var + specific_var

    def portfolio_risk(self, weights: pd.Series | dict[str, float], date=None) -> float:
        """Return portfolio volatility."""

        variance = self.portfolio_variance(weights, date=date)
        return float(np.sqrt(max(variance, 0.0)))

    def risk_contribution(self, weights: pd.Series | dict[str, float], date=None) -> pd.DataFrame:
        """Return variance contribution by factor plus specific risk."""

        w = _coerce_series(weights, name="weight")
        beta = self.portfolio_exposure(w, date=date).reindex(self.factor_cov.index).fillna(0.0)
        cov = self.factor_cov.reindex(index=beta.index, columns=beta.index).fillna(0.0)
        marginal = pd.Series(cov.to_numpy() @ beta.to_numpy(), index=beta.index)
        factor_total = beta * marginal
        specific_total = float((w.pow(2) * _specific_var(self.specific_var, w.index)).sum())
        result = pd.DataFrame({"total": factor_total})
        result.loc["specific", "total"] = specific_total
        total = float(result["total"].sum())
        result["share"] = result["total"] / total if total else np.nan
        return result

    def active_risk(
        self,
        portfolio_weights: pd.Series | dict[str, float],
        benchmark_weights: pd.Series | dict[str, float],
        date=None,
    ) -> float:
        """Return active risk against a benchmark weight vector."""

        portfolio = _coerce_series(portfolio_weights, name="portfolio")
        benchmark = _coerce_series(benchmark_weights, name="benchmark")
        index = portfolio.index.union(benchmark.index)
        active = portfolio.reindex(index, fill_value=0.0) - benchmark.reindex(index, fill_value=0.0)
        return self.portfolio_risk(active, date=date)


@dataclass(frozen=True)
class PerformanceAttribution:
    """Attribute strategy returns to factor and specific returns.

    The data contract follows the same shape as pyfolio: ``positions`` are
    wide by asset, ``factor_returns`` are wide by factor, and dated
    ``factor_loadings`` use a ``(date, symbol)`` MultiIndex.
    """

    returns: pd.Series
    positions: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_loadings: pd.DataFrame
    pos_in_dollars: bool = False

    def exposures(self) -> pd.DataFrame:
        """Return daily portfolio factor exposures."""

        positions = _position_weights(self.positions, pos_in_dollars=self.pos_in_dollars)
        rows: list[pd.Series] = []
        for dt, weights in positions.iterrows():
            loadings = _exposure_slice(self.factor_loadings, dt)
            exposure = loadings.reindex(weights.index).fillna(0.0).mul(weights, axis=0).sum(axis=0)
            exposure.name = dt
            rows.append(exposure)
        if not rows:
            return pd.DataFrame(index=positions.index, columns=self.factor_returns.columns)
        result = pd.DataFrame(rows).reindex(columns=self.factor_returns.columns).fillna(0.0)
        result.index = pd.Index(result.index, name=positions.index.name)
        return result.astype("float64")

    def attribution(self) -> pd.DataFrame:
        """Return factor, common, specific, and total return attribution."""

        returns = self.returns.astype("float64")
        exposures = self.exposures()
        factors = self.factor_returns.reindex(
            index=returns.index,
            columns=exposures.columns,
        ).fillna(0.0)
        factor_contrib = exposures.reindex(returns.index).fillna(0.0) * factors
        common = factor_contrib.sum(axis=1).rename("common_returns")
        specific = (returns - common).rename("specific_returns")
        total = returns.rename("total_returns")
        return pd.concat([factor_contrib, common, specific, total], axis=1)

    def summary(self) -> tuple[dict[str, float], pd.DataFrame]:
        """Return scalar return attribution and factor exposure summaries."""

        frame = self.attribution()
        exposure = self.exposures()
        factor_cols = list(exposure.columns)
        summary = {
            "common_return_mean": float(frame["common_returns"].mean()),
            "specific_return_mean": float(frame["specific_returns"].mean()),
            "total_return_mean": float(frame["total_returns"].mean()),
            "common_return_cumulative": float((1.0 + frame["common_returns"]).prod() - 1.0),
            "specific_return_cumulative": float((1.0 + frame["specific_returns"]).prod() - 1.0),
            "total_return_cumulative": float((1.0 + frame["total_returns"]).prod() - 1.0),
        }
        exposure_summary = pd.DataFrame(
            {
                "average_exposure": exposure.mean(axis=0),
                "cumulative_return_contribution": (
                    (1.0 + frame[factor_cols]).prod(axis=0) - 1.0
                    if factor_cols
                    else pd.Series(dtype="float64")
                ),
            }
        )
        exposure_summary.index.name = "factor"
        return summary, exposure_summary


def _coerce_series(values: pd.Series | dict[str, float], *, name: str) -> pd.Series:
    series = values.astype("float64") if isinstance(values, pd.Series) else pd.Series(values)
    series = series.astype("float64")
    series.index = series.index.astype(str)
    series.name = name
    return series


def _specific_var(specific_var: pd.Series | None, symbols: pd.Index) -> pd.Series:
    if specific_var is None:
        return pd.Series(0.0, index=symbols, dtype="float64")
    series = specific_var.astype("float64").copy()
    series.index = series.index.astype(str)
    return series.reindex(symbols, fill_value=0.0)


def _position_weights(positions: pd.DataFrame, *, pos_in_dollars: bool) -> pd.DataFrame:
    frame = positions.copy().astype("float64")
    frame.columns = frame.columns.astype(str)
    if "cash" in frame.columns:
        frame = frame.drop(columns=["cash"])
    if not pos_in_dollars:
        return frame
    gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
    return frame.div(gross, axis=0).fillna(0.0)


def _exposure_slice(exposures: pd.DataFrame, date=None) -> pd.DataFrame:
    if isinstance(exposures.index, pd.MultiIndex):
        if date is None:
            raise ValueError("date is required for dated factor exposures")
        key = pd.Timestamp(date)
        level_values = pd.to_datetime(exposures.index.get_level_values(0))
        matches = exposures[level_values == key]
        if matches.empty:
            return pd.DataFrame(columns=exposures.columns)
        result = matches.droplevel(0)
    else:
        result = exposures
    result = result.copy().astype("float64")
    result.index = result.index.astype(str)
    return result
