# 数据指南

trade-learn 策略层统一面对 OHLCV DataFrame 或 `MultiIndex(timestamp, symbol)` panel。

## 单标的

```python
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")
```

## 多标的

```python
symbols = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"]
bars = provider.history_ohlc(symbols, start="2023-01-01", end="2024-01-01")
```

多标的返回值建议是：

```text
index: timestamp, symbol
columns: open, high, low, close, volume
```

Engine:

```python
cerebro.adddata(bars)
```

Lite:

```python
bt = tl.Backtest(bars, Strategy)
```

## 额外列

普通 DataFrame 列名会 normalize 成小写。策略里统一使用小写 line 名：

```python
self.data.factor[0]
data.get_array("factor")
```

不推荐在策略里使用 `self.data.FACTOR`。
