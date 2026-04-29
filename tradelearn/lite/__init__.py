from .backtest import Backtest
from .indicator import Signal
from .strategy import Strategy

SignalStrategy = Strategy

__all__ = ["Backtest", "Signal", "SignalStrategy", "Strategy"]
