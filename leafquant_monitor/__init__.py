"""Optional strategy observations; importing the package starts no writer or transport."""
from importlib import import_module

__all__ = ['MonitorSettings', 'TradeLogger', 'TradeLogStrategy']


def __getattr__(name):
    if name in {'MonitorSettings', 'TradeLogger'}:
        return getattr(import_module('.logger', __name__), name)
    if name == 'TradeLogStrategy':
        return import_module('.strategy', __name__).TradeLogStrategy
    raise AttributeError(name)
