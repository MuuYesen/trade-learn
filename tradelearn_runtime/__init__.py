"""Optional native execution integration, independent of the Backtrader facade."""
import importlib

__all__ = ["NautilusStore"]

def __getattr__(name):
    if name == "NautilusStore":
        return importlib.import_module("tradelearn_runtime.store").NautilusStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__))
