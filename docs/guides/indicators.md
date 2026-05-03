# 指标指南

指标不下沉 Rust，保持 Python 生态可组合、可核对。

## 命名空间

| 命名空间 | 用途 |
|---|---|
| `tl.talib` / `bt.talib` | TA-Lib 风格指标 |
| `tl.tdx` / `bt.tdx` | TDX / MyTT 口径 |
| `tl.tv` / `bt.tv` | TradingView / PyneCore 口径 |
| `tl.pta` / `bt.pta` | pandas-ta-classic 口径 |

Lite:

```python
self.ma = tl.tdx.MA(self.data.close, N=20)
self.rsi = tl.tv.RSI(self.data.close, length=14)
```

Engine:

```python
self.ma = bt.tdx.MA(self.data.close, N=20)
self.rsi = bt.tv.RSI(self.data.close, length=14)
```

## 自定义指标

Lite 用 `self.I(...)`：

```python
def zscore(close, window=20):
    return (close - close.rolling(window).mean()) / close.rolling(window).std()

self.z = self.I(zscore, self.data.close.to_series(), window=20)
```

Engine 推荐自定义 `bt.Indicator`：

```python
class MidPrice(bt.Indicator):
    lines = ("mid",)

    def __init__(self):
        self.lines.mid = (self.data.high + self.data.low) / 2
```
