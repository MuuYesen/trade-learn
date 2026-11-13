"""Public Store namespace; optional native dependencies load on first access."""

__all__ = ["NautilusStore"]


def __getattr__(name):
    if name == "NautilusStore":
        from tradelearn_runtime import NautilusStore
        return NautilusStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
