"""Neutral component collection shared by batch and native frontends."""

from typing import Any


class AnalyzerCollection(dict):
    """Dict with Backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __iter__(self):
        return iter(self.values())

    def getnames(self):
        return list(self.keys())

    def getitems(self):
        return iter(self.items())

    def getbyname(self, name: str) -> Any:
        # The BT collection reports a missing name as ValueError.
        self.getnames().index(name)
        return self[name]
