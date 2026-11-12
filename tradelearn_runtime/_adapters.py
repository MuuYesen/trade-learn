"""Deferred adapter factories; registry entries never own accounts or sessions."""
from importlib.metadata import entry_points
import re

_factories = {}


def register(name, factory):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
        raise ValueError('adapter name must be a lowercase identifier')
    if not callable(factory):
        raise TypeError('adapter factory must be callable')
    if name in _factories and _factories[name] is not factory:
        raise ValueError(f'adapter {name!r} already registered')
    _factories[name] = factory


def resolve(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
        raise ValueError('adapter name must be a lowercase identifier')
    if name in _factories:
        return _factories[name]
    matches = list(entry_points(group='tradelearn.store_adapters', name=name))
    if len(matches) != 1:
        if matches:
            raise ValueError(f'ambiguous installed adapter {name!r}')
        raise ValueError(f'adapter {name!r} is not installed or registered')
    factory = matches[0].load()
    register(name, factory)
    return factory
