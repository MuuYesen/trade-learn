"""Shared thread-local construction context; independent of user facades."""
from contextlib import contextmanager
from threading import local
from typing import Any


# Centralized context to avoid import-time shadowing
class _GlobalContext(local):
    def __init__(self):
        self.current_data = None
        self.current_datas = []
        self.current_strategy = None

_G = _GlobalContext()
_MISSING = object()

def set_current_data(data: Any) -> None:
    _G.current_data = data

def set_current_datas(datas: list[Any]) -> None:
    _G.current_datas = datas

def set_current_strategy(strategy: Any) -> None:
    _G.current_strategy = strategy


@contextmanager
def engine_context(
    *,
    data: Any = _MISSING,
    datas: Any = _MISSING,
    strategy: Any = _MISSING,
):
    """Temporarily bind engine construction context and restore it reliably."""
    prev_data = _G.current_data
    prev_datas = list(_G.current_datas)
    prev_strategy = _G.current_strategy
    try:
        if data is not _MISSING:
            set_current_data(data)
        if datas is not _MISSING:
            set_current_datas(list(datas or []))
        if strategy is not _MISSING:
            set_current_strategy(strategy)
        yield
    finally:
        set_current_data(prev_data)
        set_current_datas(prev_datas)
        set_current_strategy(prev_strategy)
