"""Lite metadata view; all price and indicator lines are shared with Engine."""
from __future__ import annotations
from typing import Any
import pandas as pd

class LiteDataProxy:
    def __init__(self, data_feed):
        self._feed = data_feed
        self._extra_lines = {}

    def __getattr__(self, name):
        if name[:1].isupper():
            raise AttributeError(f"Column '{name}' is not available; use '{name.lower()}' instead.")
        try:
            return getattr(self._feed, name)
        except AttributeError:
            arrays = getattr(self._feed, "_arrays", {})
            if name not in arrays:
                raise AttributeError(name) from None
            if name not in self._extra_lines:
                from tradelearn.backtest.lines import LineSeries
                line = LineSeries(arrays[name])
                line._cursor_source = self._feed
                runtime = getattr(self._feed, "runtime", None)
                if runtime is not None:
                    line._runtime, line._bound_feed = runtime, runtime._feed_for(self._feed)
                self._extra_lines[name] = line
            return self._extra_lines[name]

    def __len__(self):
        return len(self._feed)

    @property
    def df(self):
        frame = _ta_frame(self._feed)
        return frame.iloc[:max(0, self._feed._cursor + 1)].copy()

    @property
    def index(self):
        return self.df.index

    @property
    def now(self):
        return self.index[-1]

    @property
    def tickers(self):
        return [self.the_ticker]

    @property
    def the_ticker(self):
        return str(self._feed._name)


def _ta_frame(data_feed):
    return data_feed._frame

__all__ = ["LiteDataProxy", "_ta_frame"]
