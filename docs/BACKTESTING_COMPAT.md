# Lite API And Backtesting.py Policy

Tradelearn v2 no longer maintains a `backtesting.py` compatibility facade.

The current user-facing APIs are:

- `tradelearn.engine`: Backtrader-style advanced API.
- `tradelearn.lite`: Tradelearn 1.x-style lightweight API.

Both APIs share the same private runtime:

```text
tradelearn.engine ─┐
                   ├─> tradelearn.backtest runtime ─> Rust matching / broker / portfolio
tradelearn.lite   ─┘
```

## Lite Syntax

Lite is intentionally small and follows Tradelearn 1.x-style semantics:

```python
from tradelearn.lite import Backtest, Strategy


class MyStrategy(Strategy):
    def init(self):
        self.sma = self.I(lambda close: close.rolling(20).mean(), self.data.close.df)

    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.position() and self.data.close[0] < self.sma[0]:
            self.position().close()
```

Rules:

- `self.data.close[0]` is the current bar.
- `self.data.close[-1]` is the previous bar.
- `self.position()` returns the Lite position proxy.
- `self.position().close()` closes the current Lite position.
- `self.I(...)` registers a vectorized indicator and exposes it bar by bar.
- `self.I(pd.DataFrame(...))` supports multi-column indicators, including `indicator[:, 0]`.

## What Is Not Supported

Lite does not support the `backtesting.py` syntax:

```python
self.data.Close[-1]     # not supported
self.position.close()   # not supported
```

This is deliberate. Mixing backtesting.py indexing with Tradelearn/Backtrader indexing would make strategy code ambiguous.

## Testing Policy

Testing is split into two responsibilities:

| Layer | Purpose | Acceptance |
|---|---|---|
| `tradelearn.engine` | Validate the shared runtime and Rust kernel against Backtrader | `benchmark_bt.py` must remain Backtrader `EXACT` |
| `tradelearn.lite` | Validate Lite syntax adapts correctly into the shared runtime | Lite surface tests and 1.x strategy smoke tests must pass |

Lite does not separately validate matching, fills, or portfolio accounting. Those belong to the shared runtime and are covered through the Backtrader parity benchmark.

The Lite acceptance rule is:

```text
Lite strategy smoke tests pass
+
engine Backtrader benchmark remains EXACT
```

No formal test should import `backtesting.py` as a required compatibility target.
