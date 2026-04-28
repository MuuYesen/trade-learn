# Backtesting.py Compatibility Layer

Tradelearn provides a high-fidelity compatibility layer for the `backtesting.py` library. This allows users to run strategies written for `backtesting.py` directly on the Tradelearn microkernel without modifying the strategy logic.

## Directory Structure
- `tradelearn/compat/backtesting/`: The facade implementation.
  - `backtest.py`: Implementation of the `Backtest` runner.
  - `strategy.py`: Implementation of the `Strategy` base class and indexing proxies.
- `examples/backtesting/`: Migrated example strategies.

## Usage
Strategies should be written using standard `backtesting.py` syntax:

```python
from backtesting import Backtest, Strategy

class MyStrategy(Strategy):
    def init(self):
        self.ma = self.I(lambda: pd.Series(self.data.Close).rolling(20).mean())

    def next(self):
        if self.data.Close[-1] > self.ma[-1]:
            self.buy()
        elif self.data.Close[-1] < self.ma[-1]:
            self.position.close()
```

To run this strategy in Tradelearn, either:
1. Ensure `tradelearn.compat.backtesting` is imported as `backtesting`.
2. Use a local `backtesting.py` shim that redirects to Tradelearn.

## Architecture
The facade maps `backtesting.py` calls to the Tradelearn microkernel:
- **Data Access**: `BacktestingDataProxy` translates capitalized attributes (`Close`, `Open`) and ensures indexing (`[-1]`) matches the current simulation bar.
- **Indicators**: `self.I()` pre-calculates indicators on the full dataset but returns a `IndicatorProxy` that prevents lookahead bias during `next()`.
- **Execution**: `buy()` and `sell()` calls are translated to the `RustBroker` via the core strategy.

## Supported Features
- [x] `init()` and `next()` lifecycle.
- [x] `self.I()` for indicator registration.
- [x] Capitalized data attribute access (`data.Close`, etc.).
- [x] Relative indexing (`[-1]`, `[-2]`) for data and indicators.
- [x] `self.position` proxy with `close()` method.
- [x] Percentage-based sizing in `buy(size=0.95)`.
- [x] Basic performance statistics (`Return`, `Win Rate`, `# Trades`).
- [ ] Interactive plotting (Placeholder).
