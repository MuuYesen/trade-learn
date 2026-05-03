# Lite API 签名

`tradelearn.lite` 是 Tradelearn 1.x 风格轻量入口。本文档的签名来自运行时代码,参数说明由生成器元数据补充。

本页偏查询:用于核对函数签名、参数名和返回值。第一次写策略请先看 [Lite Guide](lite.md)、[Engine Guide](engine.md) 或 [Strategy Writing Guide](strategy.md)。

Generated from `tradelearn.lite` code signatures by `scripts/generate_api_reference.py`.

## `Backtest.__init__`

创建 Lite 回测运行器。Lite 是 Tradelearn 1.x 风格入口,不是 backtesting.py 兼容层。

```python
Backtest.__init__(self, data: 'pd.DataFrame | dict[str, pd.DataFrame]', strategy: 'type[Strategy]', cash: 'float' = 10000, commission: 'float' = 0.0, margin: 'float' = 1.0, trade_on_close: 'bool' = False, hedging: 'bool' = False, exclusive_orders: 'bool' = False, holding: 'dict[str, float] | None' = None, trade_start_date: 'str | pd.Timestamp | None' = None, lot_size: 'int' = 1, fail_fast: 'bool' = True, stats_mode: 'str' = 'full', storage: 'dict | None' = None, match_mode: 'str' = 'exact')
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `pd.DataFrame \| dict[str, pd.DataFrame]` | `required` | 单资产 DataFrame 或 `{ticker: DataFrame}` 多资产输入。 |
| `strategy` | `type[Strategy]` | `required` | `tradelearn.lite.Strategy` 子类。 |
| `cash` | `float` | `10000` | 初始资金。 |
| `commission` | `float` | `0.0` | 手续费比例。 |
| `margin` | `float` | `1.0` | 保证金配置保留参数。 |
| `trade_on_close` | `bool` | `False` | 是否在当前 bar close 成交。 |
| `hedging` | `bool` | `False` | 保留参数。 |
| `exclusive_orders` | `bool` | `False` | 保留参数。 |
| `holding` | `dict[str, float] \| None` | `None` | 初始持仓保留参数。 |
| `trade_start_date` | `str \| pd.Timestamp \| None` | `None` | 开始交易日期保留参数。 |
| `lot_size` | `int` | `1` | 最小交易单位保留参数。 |
| `fail_fast` | `bool` | `True` | 保留参数。 |
| `stats_mode` | `str` | `'full'` |  |
| `storage` | `dict \| None` | `None` | 策略可读写的共享存储。 |
| `match_mode` | `str` | `'exact'` | 撮合模式; 默认 `exact` 以复用 Engine/Backtrader 对齐路径。 |

返回: `Backtest` 实例。

## `Backtest.run`

执行回测并返回核心统计。

```python
Backtest.run(self, **kwargs) -> 'LiteStats'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `**kwargs` | `Any` | `required` | 策略参数覆盖值; 参数必须先作为策略类属性声明。 |

返回: `LiteStats`,支持 `stats['final_value']` 等 summary key,并提供 `summary`、`equity`、`returns`、`fills`、`trades`、`positions`、`orders`、`records`、`strategy`、`config`。

```python
stats = Backtest(data, MyStrategy).run(fast=10, slow=30)
```

## `Backtest.optimize`

执行简单 grid search,按 `Return [%]` 选择最优结果。

```python
Backtest.optimize(self, **kwargs) -> 'LiteStats'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `**kwargs` | `Any` | `required` | 参数网格,如 `fast=range(5, 20, 5)`。 |

返回: 最佳参数组合对应的 `pd.Series`。

## `Strategy.I`

声明指标。callable 会批量预计算; Series/array-like 会包装为渐进揭示指标。

```python
Strategy.I(self, funcval: 'Callable | pd.DataFrame | pd.Series | Any', *args, name: 'str | None' = None, plot: 'bool' = True, overlay: 'bool | None' = None, color: 'str | None' = None, scatter: 'bool' = False, **kwargs) -> 'Any'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `funcval` | `Callable \| pd.DataFrame \| pd.Series \| Any` | `required` | 指标函数、Series、DataFrame 或 array-like。 |
| `*args` | `Any` | `required` | 传给指标函数的位置参数。 |
| `name` | `str \| None` | `None` | 指标名称。 |
| `plot` | `bool` | `True` | 绘图元数据。 |
| `overlay` | `bool \| None` | `None` | 绘图元数据。 |
| `color` | `str \| None` | `None` | 绘图元数据。 |
| `scatter` | `bool` | `False` | 绘图元数据。 |
| `**kwargs` | `Any` | `required` | 传给指标函数,或保存到指标 attrs。 |

返回: `IndicatorProxy` 或 `IndicatorBundle`。

```python
def zscore(close, window=20):
    series = close.to_series() if hasattr(close, 'to_series') else close
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

self.z = self.I(zscore, self.data.close, window=20)
```

## `Strategy.position`

返回当前 ticker 的 Lite 持仓代理。

```python
Strategy.position(self, ticker: 'str | None' = None) -> 'Any'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ticker` | `str \| None` | `None` | 目标 ticker; `None` 表示主数据。 |

返回: `PositionProxy`。

## `Strategy.buy`

提交 Lite 买入订单。

```python
Strategy.buy(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ticker` | `str` | `None` | 目标 ticker; `None` 表示主数据。 |
| `size` | `float` | `1.0` | 订单数量; 百分比仓位请使用 `order_target_percent()` 或 `target_percent()`。 |
| `limit` | `float` | `None` | 限价价格。 |
| `stop` | `float` | `None` | Stop 触发价格。 |
| `sl` | `float` | `None` | 止损价格; 有 `sl` 或 `tp` 时走 bracket。 |
| `tp` | `float` | `None` | 止盈价格; 有 `sl` 或 `tp` 时走 bracket。 |
| `tag` | `object` | `None` | 写入订单 `info['tag']` 的业务标签。 |

返回: `Order` 或 bracket `list[Order]`。

## `Strategy.sell`

提交 Lite 卖出订单。

```python
Strategy.sell(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ticker` | `str` | `None` | 目标 ticker; `None` 表示主数据。 |
| `size` | `float` | `1.0` | 订单数量; 百分比仓位请使用 `order_target_percent()` 或 `target_percent()`。 |
| `limit` | `float` | `None` | 限价价格。 |
| `stop` | `float` | `None` | Stop 触发价格。 |
| `sl` | `float` | `None` | 止损价格; 有 `sl` 或 `tp` 时走 bracket。 |
| `tp` | `float` | `None` | 止盈价格; 有 `sl` 或 `tp` 时走 bracket。 |
| `tag` | `object` | `None` | 写入订单 `info['tag']` 的业务标签。 |

返回: `Order` 或 bracket `list[Order]`。

## `Strategy.order_target_percent`

按账户权益比例调整目标持仓。

```python
Strategy.order_target_percent(self, *, ticker: 'str' = None, target: 'float' = 0.0, **kwargs: 'Any')
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ticker` | `str` | `None` | 目标 ticker; `None` 表示主数据。 |
| `target` | `float` | `0.0` | 目标权益比例,如 `0.5` 表示 50%。 |
| `**kwargs` | `Any` | `required` | 透传给底层 `buy` / `sell` / `close`。 |

返回: `Order | None`。

## `Strategy.target_percent`

按 ticker 直接表达目标组合权重。

```python
Strategy.target_percent(self, ticker: 'str', target: 'float')
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ticker` | `str` | `required` | 目标 ticker。 |
| `target` | `float` | `required` | 目标权益比例,如 `0.5` 表示 50%。 |

返回: `Order | None`。

## `Strategy.target_weights`

按一组 ticker 权重调整目标组合。`cash` 可作为保留键表达现金权重。

```python
Strategy.target_weights(self, weights: 'Mapping[str, float] | pd.Series', *, close_missing: 'bool' = True) -> 'list[Any]'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `weights` | `Mapping[str, float] \| pd.Series` | `required` | ticker 到目标权重的映射,或 pandas Series。 |
| `close_missing` | `bool` | `True` | 是否把未出现在 weights 中的已知 ticker 调整为 0。 |

返回: `list[Order]`。

## `Strategy.target_equal`

把总目标权重等分给一组 ticker。

```python
Strategy.target_equal(self, tickers: 'Sequence[str]', *, weight: 'float' = 1.0, close_missing: 'bool' = True) -> 'list[Any]'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `tickers` | `Sequence[str]` | `required` | 目标 ticker 列表。 |
| `weight` | `float` | `1.0` | 这组 ticker 合计占用的权益比例。 |
| `close_missing` | `bool` | `True` | 是否把未出现在 tickers 中的已知 ticker 调整为 0。 |

返回: `list[Order]`。

## `Strategy.close_all`

清空 Lite 已知数据源对应的目标持仓。

```python
Strategy.close_all(self) -> 'list[Any]'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|

返回: `list[Order]`。

## `Strategy.record`

记录策略内部序列,结果写入 stats 的 `_records`。

```python
Strategy.record(self, name: 'str' = None, plot: 'bool' = True, overlay: 'bool' = None, color: 'str' = None, scatter: 'bool' = False, **kwargs) -> 'None'
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `str` | `None` | 记录名称。 |
| `plot` | `bool` | `True` | 绘图元数据。 |
| `overlay` | `bool` | `None` | 绘图元数据。 |
| `color` | `str` | `None` | 绘图元数据。 |
| `scatter` | `bool` | `False` | 绘图元数据。 |
| `**kwargs` | `Any` | `required` | 要记录的键值对。 |

返回: `None`。

## Lite 完整接口

下表从运行时代码自动抽取,用于检查 Lite 语法糖暴露了哪些入口。

### Backtest

| Member | Kind | Signature | Summary |
|---|---|---|---|
| `Backtest.run` | method | `(self, **kwargs) -> 'LiteStats'` | Run the backtest and return statistics. |
| `Backtest.optimize` | method | `(self, **kwargs) -> 'LiteStats'` | Simple grid search optimization. |
| `Backtest.plot` | method | `(self, *args, **kwargs)` | Return market replay charts for the most recent Lite run. |
| `Backtest.report` | method | `(self, path: 'str' = 'report.html', benchmark=None, sections=None)` | Write a Tradelearn report for the most recent Lite run. |

### Strategy

| Member | Kind | Signature | Summary |
|---|---|---|---|
| `Strategy.position` | method | `(self, ticker: 'str | None' = None) -> 'Any'` | Return the Tradelearn Lite position view for a ticker. |
| `Strategy.I` | method | `(self, funcval: 'Callable | pd.DataFrame | pd.Series | Any', *args, name: 'str | None' = None, plot: 'bool' = True, overlay: 'bool | None' = None, color: 'str | None' = None, scatter: 'bool' = False, **kwargs) -> 'Any'` | Declare a Tradelearn Lite gradually revealed indicator. |
| `Strategy.buy` | method | `(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)` |  |
| `Strategy.sell` | method | `(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)` |  |
| `Strategy.cancel` | method | `(self, order: 'Any') -> 'None'` |  |
| `Strategy.order_target_size` | method | `(self, *, ticker: 'str' = None, target: 'float' = 0, **kwargs: 'Any')` |  |
| `Strategy.order_target_value` | method | `(self, *, ticker: 'str' = None, target: 'float' = 0.0, price: 'float | None' = None, **kwargs: 'Any')` |  |
| `Strategy.order_target_percent` | method | `(self, *, ticker: 'str' = None, target: 'float' = 0.0, **kwargs: 'Any')` |  |
| `Strategy.target_percent` | method | `(self, ticker: 'str', target: 'float')` | Move one ticker toward a target portfolio weight. |
| `Strategy.target_weights` | method | `(self, weights: 'Mapping[str, float] | pd.Series', *, close_missing: 'bool' = True) -> 'list[Any]'` | Move the portfolio toward the requested ticker weights. |
| `Strategy.target_equal` | method | `(self, tickers: 'Sequence[str]', *, weight: 'float' = 1.0, close_missing: 'bool' = True) -> 'list[Any]'` | Assign an equal combined target weight to the selected tickers. |
| `Strategy.close_all` | method | `(self) -> 'list[Any]'` | Close all known Lite data-feed positions. |
| `Strategy.buy_bracket` | method | `(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None) -> 'list[Any]'` |  |
| `Strategy.sell_bracket` | method | `(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None) -> 'list[Any]'` |  |
| `Strategy.record` | method | `(self, name: 'str' = None, plot: 'bool' = True, overlay: 'bool' = None, color: 'str' = None, scatter: 'bool' = False, **kwargs) -> 'None'` |  |
| `Strategy.equity` | property | `` |  |
| `Strategy.storage` | property | `` |  |
| `Strategy.orders` | property | `` |  |
| `Strategy.trades` | method | `(self, ticker: 'str' = None) -> 'tuple[Any, ...]'` |  |
| `Strategy.closed_trades` | property | `` |  |
| `Strategy.start_on_day` | method | `(self, n: 'int') -> 'None'` |  |
| `Strategy.start_on_bar` | method | `(self, n: 'int') -> 'None'` | Start ``next`` no earlier than bar index ``n``. |
| `Strategy.prepare_data` | method | `(tickers: 'list[str]', start: 'str') -> 'pd.DataFrame | None'` |  |

### LiteDataProxy

| Member | Kind | Signature | Summary |
|---|---|---|---|
| `LiteDataProxy.df` | property | `` |  |
| `LiteDataProxy.index` | property | `` |  |
| `LiteDataProxy.now` | property | `` |  |
| `LiteDataProxy.tickers` | property | `` |  |
| `LiteDataProxy.the_ticker` | property | `` |  |

### PositionProxy

| Member | Kind | Signature | Summary |
|---|---|---|---|
| `PositionProxy.size` | property | `` |  |
| `PositionProxy.close` | method | `(self, portion: 'float' = 1.0)` |  |
| `PositionProxy.pl` | property | `` |  |
| `PositionProxy.pl_pct` | property | `` |  |
| `PositionProxy.is_long` | property | `` |  |
| `PositionProxy.is_short` | property | `` |  |
