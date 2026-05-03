# 策略编写指南

本页由 `scripts/generate_api_reference.py` 自动生成,只说明两种策略入口的共同心智模型。
完整函数签名请看 Lite / Engine API 签名页,完整工作流请看对应指南。

Tradelearn 当前有两个用户入口:

- `tradelearn.lite`: Tradelearn 1.x 风格轻量 API,适合快速写单文件策略和研究原型。
- `tradelearn.engine`: Backtrader 风格高级 API,适合复杂事件策略、Analyzer、Observer、Sizer、多数据和参数优化。

两者共享同一套 `tradelearn.backtest` runtime 和 Rust 撮合内核。区别应该只在语法翻译层,不应该各自维护不同的业务逻辑。

## 基本规则

- `line[0]` 是当前 bar,`line[-1]` 是前一根 bar。
- 指标不下沉 Rust;使用真实 TA-Lib、pandas-ta-classic、TDX、TradingView 等 Python 指标生态批量计算。
- Engine 是 Backtrader 数值对齐主入口;修改撮合、订单、broker、生命周期后必须跑 Backtrader 对齐测试。
- Lite 只验证语法层是否正确接入同一 runtime;底层正确性仍以 Engine/Backtrader 对齐为主。

## Lite 策略

Lite 策略继承 `tradelearn.lite.Strategy`,通常在 `init()` 声明指标,在 `next()` 写交易逻辑。

```python
import pandas as pd
import tradelearn.lite as tl


class SmaCross(tl.Strategy):
    fast = 10
    slow = 30

    def init(self):
        self.fast_sma = tl.tdx.MA(self.data.close, N=self.fast)
        self.slow_sma = tl.tdx.MA(self.data.close, N=self.slow)

    def next(self):
        if not self.position() and self.fast_sma[0] > self.slow_sma[0]:
            self.buy(size=1)
        elif self.position() and self.fast_sma[0] < self.slow_sma[0]:
            self.position().close()


data = pd.read_csv('bars.csv', parse_dates=True, index_col=0)
stats = tl.Backtest(data, SmaCross, cash=100_000, match_mode='exact').run()
print(stats['final_value'])
```

Lite 常用写法:

| 需求 | 写法 |
|---|---|
| 当前价格 | `self.data.close[0]` |
| 前一根价格 | `self.data.close[-1]` |
| 内置指标 | `tl.tdx.MA(self.data.close, N=20)` |
| 自定义函数指标 | `self.I(my_func, self.data.close, window=20)` |
| 当前持仓 | `self.position()` |
| 指定 ticker 持仓 | `self.position('BTCUSDT')` |
| 买入/卖出 | `self.buy(size=...)` / `self.sell(size=...)` |
| 权益比例调仓 | `self.order_target_percent(ticker='BTCUSDT', target=0.5)` |
| 止损止盈 | `self.buy(sl=..., tp=...)` |
| 记录序列 | `self.record(signal=value)` |
| 当前权益 | `self.equity` |
| 运行存储 | `self.storage` |

Lite `run()` 返回 `LiteStats`:

```python
stats = tl.Backtest(data, SmaCross).run()
stats['final_value']
stats.summary
stats.equity
stats.trades
stats.records
stats.strategy
stats.config
```

## Engine 策略

Engine 策略继承 `tradelearn.engine.Strategy`,通常在 `__init__()` 声明指标,在 `next()` 写交易逻辑。

```python
import pandas as pd
import tradelearn.engine as bt


class SmaCross(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30),
    )

    def __init__(self):
        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)
        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)

    def next(self):
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=1)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


data = pd.read_csv('bars.csv', parse_dates=True, index_col=0)
cerebro = bt.Cerebro(match_mode='exact')
cerebro.adddata(data, name='BTCUSDT')
cerebro.addstrategy(SmaCross, fast=10, slow=30)
cerebro.broker.setcash(100_000)
strategy = cerebro.run()[0]
print(strategy.broker.getvalue())
```

Engine 常用写法:

| 需求 | 写法 |
|---|---|
| 当前价格 | `self.data.close[0]` |
| 前一根价格 | `self.data.close[-1]` |
| 当前持仓 | `self.position` 或 `self.getposition(data)` |
| 内置指标 | `bt.tdx.MA(self.data.close, N=20)` |
| 复杂自定义指标 | `class MyInd(bt.Indicator)` |
| 买入/卖出 | `self.buy(size=...)` / `self.sell(size=...)` |
| 平仓 | `self.close()` |
| 目标仓位 | `self.order_target_size(...)` / `self.order_target_percent(...)` |
| bracket | `self.buy_bracket(...)` / `self.sell_bracket(...)` |
| 多数据查询 | `self.getdatabyname('BTCUSDT')` |
| 订单通知 | `notify_order(self, order)` |
| 交易通知 | `notify_trade(self, trade)` |

## 指标写法

Engine 内置指标使用 vendor 命名空间,复杂自定义指标使用 `bt.Indicator`:

```python
self.ma20 = bt.tdx.MA(self.data.close, N=20)
self.rsi14 = bt.talib.RSI(self.data.close, timeperiod=14)
```

Lite 内置指标也直接使用 vendor 命名空间;只有自定义函数才需要 `self.I(...)`:

```python
self.ma20 = tl.tdx.MA(self.data.close, N=20)
self.macd = tl.talib.MACD(self.data.close)
self.custom = self.I(my_func, self.data.close, window=20)
```

TA-Lib / pandas-ta-classic / TDX / TradingView 指标都保留在 Python 生态中,不要写成 Rust 指标:

```python
self.talib_sma = tl.talib.SMA(self.data.close, timeperiod=20)
self.pta_sma = tl.pta.SMA(self.data.close, length=20)
self.tdx_ma = tl.tdx.MA(self.data.close, N=20)
self.tv_rsi = tl.tv.RSI(self.data.close, length=14)
```

## 测试验收

底层撮合和订单正确性以 Engine/Backtrader 对齐为主:

```bash
uv run python benchmarks/runners/benchmark_bt.py
```

Lite 策略测试重点是语法层能否跑通并接入同一 runtime,例如 `self.I(...)`、`position()`、`data.close[0]`、`record()`、`sl/tp`。
