# 数据指南

trade-learn 策略层统一面对 OHLCV DataFrame 或 `MultiIndex(timestamp, symbol)` panel。数据层的目标是：provider 取到的数据可以直接进入 Lite / Engine，不需要用户手工拆成多份 DataFrame。

## 标准 Bars 形态

| 形态 | 索引 | 列 |
|---|---|---|
| 单标的 | `DatetimeIndex` 或可转换为时间索引 | `open`, `high`, `low`, `close`, `volume` |
| 多标的 | `MultiIndex(timestamp, symbol)` | `open`, `high`, `low`, `close`, `volume` |

列名会 normalize 成小写；额外列也会保留为小写 line。

```python
self.data.factor[0]
data.get_array("factor")
```

不推荐在策略里使用 `self.data.FACTOR`。

## 单标的读取

```python
from tradelearn.data import TradingViewProvider

provider = TradingViewProvider()
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")
```

Engine:

```python
import tradelearn.engine as bt

cerebro = bt.Cerebro()
cerebro.adddata(bars, name="NASDAQ:AAPL")
```

Lite:

```python
import tradelearn.lite as tl

stats = tl.Backtest(bars, Strategy).run()
```

## 多标的读取

```python
symbols = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"]
bars = provider.history_ohlc(symbols, start="2023-01-01", end="2024-01-01")
```

多标的返回值是一个 panel：

```text
index: timestamp, symbol
columns: open, high, low, close, volume
```

Engine 可以直接接收 panel；`Cerebro.adddata(bars)` 会按 `symbol` 层自动拆成多个 feed，feed name 默认就是 symbol：

```python
cerebro = bt.Cerebro()
cerebro.adddata(bars)
```

如果需要自定义显示名，可以传映射：

```python
cerebro.adddata(bars, name={"NASDAQ:AAPL": "AAPL", "NASDAQ:MSFT": "MSFT"})
```

Lite 也可以直接接收 panel：

```python
stats = tl.Backtest(bars, PortfolioStrategy).run()
```

策略里可以按 ticker 访问：

```python
class PortfolioStrategy(tl.Strategy):
    def next(self):
        aapl = self.datas["NASDAQ:AAPL"]
        msft = self.datas["NASDAQ:MSFT"]
        if aapl.close[0] > msft.close[0]:
            self.target_weights({"NASDAQ:AAPL": 0.8, "cash": 0.2})
```

## Symbol 对应规则

### TradingView

TradingView 数据建议使用 `EXCHANGE:SYMBOL`：

```python
provider.history_ohlc("NASDAQ:AAPL", start="2024-01-01")
provider.history_ohlc(["NASDAQ:AAPL", "NASDAQ:MSFT"], start="2024-01-01")
```

输出的 `symbol` 层保持 TradingView 形式，例如 `NASDAQ:AAPL`。

如果调用时传入裸 symbol，需要同时提供 `exchange`：

```python
provider.history_ohlc("AAPL", exchange="NASDAQ", start="2024-01-01")
```

### TDX

TDX 输出会归一为 canonical symbol：

```python
provider.history_ohlc("SZ:000001", start="2024-01-01")
provider.history_ohlc("SH:600000", start="2024-01-01")
```

裸 6 位代码会按规则推断交易所；如果无法确定，框架会要求用户显式写 `SH:` 或 `SZ:`，避免把同一个股票映射到错误市场。

```python
# 推荐
"SZ:000001"
"SH:600000"

# 不推荐: 如果存在歧义会抛错
"000001"
```

## 多标的策略里的历史窗口

Engine 和 Lite 都提供 `history_panel(lookback)`，用于把最近 N 根多资产 bar 合成一个 panel。

```python
panel = self.history_panel(20)
close = panel["close"].unstack("symbol")
```

第一次进入 `next()` 时，如果历史不足 20 根，返回的窗口会少于 20 根。需要固定窗口长度时，应使用：

```python
self.addminperiod(21)      # Engine
self.start_on_bar(21)      # Lite
```

## 数据源扩展

自定义 provider 只需要返回标准 Bars：

```python
class MyProvider:
    def history_ohlc(self, symbol, *, start=None, end=None, freq="1d"):
        ...
        return bars
```

只要返回的是单标的 OHLCV DataFrame 或多标的 `MultiIndex(timestamp, symbol)` panel，就可以直接接入 Engine / Lite。
