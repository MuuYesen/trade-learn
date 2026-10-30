"""Portfolio and factor metrics, loaded only when their API is requested.

Scalar cash-flow arithmetic uses only the standard library; vector metrics
retain their normal NumPy/pandas requirements when explicitly imported.
"""
from importlib import import_module

__all__ = ['cash_flow_return', 'alpha', 'annual_return', 'autocorrelation', 'avg_loss', 'avg_win', 'beta', 'calmar', 'cum_returns', 'cvar', 'downside_risk', 'drawdown_series', 'excess_returns', 'expectancy', 'factor_returns', 'ic', 'ic_ir', 'information_ratio', 'log_to_simple', 'max_consecutive_losses', 'max_consecutive_wins', 'max_drawdown', 'omega', 'profit_factor', 'quantile_turnover', 'quantile_returns', 'rank_ic', 'sharpe', 'simple_returns', 'sortino', 'tail_ratio', 'turnover', 'var', 'volatility', 'win_rate']

_MODULES = {
    'cash_flow_return': 'cash_flows',
    'autocorrelation': 'factor',
    'factor_returns': 'factor',
    'ic': 'factor',
    'ic_ir': 'factor',
    'quantile_returns': 'factor',
    'quantile_turnover': 'factor',
    'rank_ic': 'factor',
    'turnover': 'factor',
    'annual_return': 'returns',
    'cum_returns': 'returns',
    'excess_returns': 'returns',
    'log_to_simple': 'returns',
    'simple_returns': 'returns',
    'alpha': 'risk',
    'beta': 'risk',
    'calmar': 'risk',
    'cvar': 'risk',
    'downside_risk': 'risk',
    'drawdown_series': 'risk',
    'information_ratio': 'risk',
    'max_drawdown': 'risk',
    'omega': 'risk',
    'sharpe': 'risk',
    'sortino': 'risk',
    'tail_ratio': 'risk',
    'var': 'risk',
    'volatility': 'risk',
    'avg_loss': 'trade',
    'avg_win': 'trade',
    'expectancy': 'trade',
    'max_consecutive_losses': 'trade',
    'max_consecutive_wins': 'trade',
    'profit_factor': 'trade',
    'win_rate': 'trade',
}


def __getattr__(name):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
