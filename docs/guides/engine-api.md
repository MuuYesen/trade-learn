# Engine API 签名

`tradelearn.engine` 是 Backtrader 风格高级入口。本文档的签名来自运行时代码,参数说明由生成器元数据补充。

本页偏查询:用于核对函数签名、参数名和返回值。第一次写策略请先看 [Lite 指南](lite.md)、[Engine 指南](engine.md) 或 [策略编写指南](strategy.md)。

本页由 `scripts/generate_api_reference.py` 从 `tradelearn.engine` 运行时代码签名生成。

## `Cerebro.__init__`

创建 Backtrader 风格的高级事件驱动回测运行器。

```python
Cerebro.__init__(self, match_mode: 'str' = 'exact', callback_batch: 'int' = 1, trade_on_close: 'bool' = False, exactbars: 'bool' = False, stdstats: 'bool' = True, slippage: 'Any | None' = None, commission: 'Any | None' = None, mode: 'str' = 'backtest', stats_mode: 'str' = 'full', **kwargs: 'Any') -> 'None'
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `match_mode` | `str` | `'exact'` | `exact` 对齐 Backtrader bar 级撮合; `smart` 使用趋势路径推演和止损优先悲观撮合。 |
| `callback_batch` | `int` | `1` | 保留参数。正式事件驱动路径仍逐 bar callback。 |
| `trade_on_close` | `bool` | `False` | 是否在当前 bar close 成交; 默认下一根 bar open 成交。 |
| `exactbars` | `bool` | `False` | Backtrader 兼容保留参数。 |
| `stdstats` | `bool` | `True` | 是否挂载默认 observer。 |
| `slippage` | `Any \| None` | `None` | slippage 配置对象。 |
| `commission` | `Any \| None` | `None` | commission 配置对象。 |
| `mode` | `str` | `'backtest'` | `backtest` / `paper` / `live`; 当前主路径是 backtest。 |
| `stats_mode` | `str` | `'full'` |  |
| `**kwargs` | `Any` | `required` | 保留给 Backtrader facade 的额外配置。 |

返回值：`Cerebro` 实例。

```python
import tradelearn.engine as bt

cerebro = bt.Cerebro(match_mode='exact')
```

## `Cerebro.adddata`

添加一个 DataFrame 或 DataFeed 到运行器。

```python
Cerebro.adddata(self, data: 'Any', name: 'Any | None' = None) -> 'Any'
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | `required` | 包含 `open/high/low/close/volume` 的 DataFrame,或已经构造好的 DataFeed。 |
| `name` | `Any \| None` | `None` | 数据名称; 后续可通过 `datasbyname` 或 `getdatabyname()` 查询。 |

返回值：添加后的 data feed。

## `Cerebro.addstrategy`

注册策略类及策略参数。

```python
Cerebro.addstrategy(self, strategy: 'type[Strategy]', *args: 'Any', **kwargs: 'Any') -> 'None'
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `strategy` | `type[Strategy]` | `required` | `bt.Strategy` 子类。 |
| `*args` | `Any` | `required` | 传给策略构造的 positional 参数。 |
| `**kwargs` | `Any` | `required` | 传给策略参数系统的 keyword 参数。 |

返回值：`None`。

## `Cerebro.run`

执行回测并返回策略实例列表。

```python
Cerebro.run(self, **kwargs: 'Any') -> 'list[Strategy]'
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `**kwargs` | `Any` | `required` |  |

返回值：`list[Strategy]`。

```python
results = cerebro.run()
strategy = results[0]
```

## `Strategy.buy`

提交买入订单。

```python
Strategy.buy(self, data: 'Any' = None, size: 'float | None' = None, price: 'float | None' = None, exectype: 'int | None' = None, **kwargs)
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | `None` | 目标数据 feed; 默认主数据。 |
| `size` | `float \| None` | `None` | 订单数量; `None` 时使用 sizer。 |
| `price` | `float \| None` | `None` | Limit/Stop 价格。 |
| `exectype` | `int \| None` | `None` | `Order.Market` / `Order.Limit` / `Order.Stop` 等。 |
| `**kwargs` | `Any` | `required` | 订单元数据,如 `parent`、`oco`、`transmit`、`info`。 |

返回值：`Order`。

## `Strategy.sell`

提交卖出订单。

```python
Strategy.sell(self, data: 'Any' = None, size: 'float | None' = None, price: 'float | None' = None, exectype: 'int | None' = None, **kwargs)
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | `None` | 目标数据 feed; 默认主数据。 |
| `size` | `float \| None` | `None` | 订单数量; `None` 时使用 sizer。 |
| `price` | `float \| None` | `None` | Limit/Stop 价格。 |
| `exectype` | `int \| None` | `None` | `Order.Market` / `Order.Limit` / `Order.Stop` 等。 |
| `**kwargs` | `Any` | `required` | 订单元数据,如 `parent`、`oco`、`transmit`、`info`。 |

返回值：`Order`。

## `Strategy.buy_bracket`

提交买入 bracket 订单,生成 main / stop / limit 三联订单。

```python
Strategy.buy_bracket(self, data=None, size=None, price=None, plimit=None, exectype=2, valid=None, trailamount=None, trailpercent=None, oargs=None, stopprice=None, stopexec=3, stopargs=None, limitprice=None, limitexec=2, limitargs=None, **kwargs)
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | `None` | 目标数据 feed; 默认主数据。 |
| `size` | `Any` | `None` | 订单数量。 |
| `price` | `Any` | `None` | 主订单价格。 |
| `plimit` | `Any` | `None` | 主订单 limit price 别名。 |
| `exectype` | `Any` | `2` | 主订单执行类型。 |
| `valid` | `Any` | `None` | 订单有效期,当前作为元数据保留。 |
| `trailamount` | `Any` | `None` | Trailing stop 金额参数。 |
| `trailpercent` | `Any` | `None` | Trailing stop 百分比参数。 |
| `oargs` | `Any` | `None` | 主订单额外参数。 |
| `stopprice` | `Any` | `None` | 止损订单触发价格。 |
| `stopexec` | `Any` | `3` | 止损订单执行类型。 |
| `stopargs` | `Any` | `None` | 止损订单额外参数。 |
| `limitprice` | `Any` | `None` | 止盈订单价格。 |
| `limitexec` | `Any` | `2` | 止盈订单执行类型。 |
| `limitargs` | `Any` | `None` | 止盈订单额外参数。 |
| `**kwargs` | `Any` | `required` | 透传给底层订单的额外参数。 |

返回值：`list[Order]`: `[main, stop, limit]`。

## `Strategy.order_target_percent`

按账户权益比例调整目标持仓。

```python
Strategy.order_target_percent(self, data: 'Any' = None, target: 'float' = 0.0, **kwargs)
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | `None` | 目标数据 feed; 默认主数据。 |
| `target` | `float` | `0.0` | 目标权益比例,如 `0.5` 表示 50%。 |
| `**kwargs` | `Any` | `required` | 透传给 `buy` / `sell` / `close`。 |

返回值：`Order | None`。

## Engine 完整接口

下表从运行时代码自动抽取,用于补足指南未逐段展开的常用接口。

### Cerebro

| 成员 | 类型 | 签名 | 说明 |
|---|---|---|---|
| `Cerebro.setcash` | 方法 | `(self, cash: 'float') -> 'None'` |  |
| `Cerebro.getbroker` | 方法 | `(self) -> 'Any'` |  |
| `Cerebro.setbroker` | 方法 | `(self, broker: 'Any') -> 'None'` |  |
| `Cerebro.setcommission` | 方法 | `(self, commission: 'float' = 0.0, margin: 'float' = 0.0, mult: 'float' = 1.0, comminfo: 'Any' = None, name: 'str | None' = None) -> 'None'` | Set commission parameters or a custom CommInfo object. |
| `Cerebro.set_coc` | 方法 | `(self, coc: 'bool' = True) -> 'None'` |  |
| `Cerebro.adddata` | 方法 | `(self, data: 'Any', name: 'Any | None' = None) -> 'Any'` |  |
| `Cerebro.chaindata` | 方法 | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `Cerebro.rolloverdata` | 方法 | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `Cerebro.resampledata` | 方法 | `(self, data: 'Any', timeframe: 'int', compression: 'int' = 1, name: 'str | None' = None, **kwargs: 'Any') -> 'DataFeed'` |  |
| `Cerebro.replaydata` | 方法 | `(self, data: 'Any', *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `Cerebro.addstrategy` | 方法 | `(self, strategy: 'type[Strategy]', *args: 'Any', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.optstrategy` | 方法 | `(self, strategy: 'type[Strategy]', *args: 'Any', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.addanalyzer` | 方法 | `(self, analyzer: 'type[Analyzer]', *args: 'Any', _name=None, **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.addobserver` | 方法 | `(self, observer: 'type[Any]', *args: 'Any', _name=None, **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.addwriter` | 方法 | `(self, writer: 'type[Any]', *args: 'Any', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.getwriterheaders` | 方法 | `(self) -> 'list[Any]'` |  |
| `Cerebro.getwriterinfo` | 方法 | `(self) -> 'list[Any]'` |  |
| `Cerebro.getwritervalues` | 方法 | `(self) -> 'list[Any]'` |  |
| `Cerebro.addstore` | 方法 | `(self, store: 'Any') -> 'None'` |  |
| `Cerebro.addtimer` | 方法 | `(self, *args: 'Any', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.addcalendar` | 方法 | `(self, calendar: 'Any') -> 'None'` |  |
| `Cerebro.addsizer` | 方法 | `(self, sizer: 'type[Any]', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.setsizer` | 方法 | `(self, sizer: 'type[Any]', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.add_signal` | 方法 | `(self, sigtype: 'int', sigcls: 'type', *sigargs: 'Any', **sigkwargs: 'Any') -> 'None'` |  |
| `Cerebro.signal_strategy` | 方法 | `(self, stratcls: 'type', *args: 'Any', **kwargs: 'Any') -> 'None'` |  |
| `Cerebro.signal_concurrent` | 方法 | `(self, onoff: 'bool') -> 'None'` |  |
| `Cerebro.signal_accumulate` | 方法 | `(self, onoff: 'bool') -> 'None'` |  |
| `Cerebro.runstop` | 方法 | `(self) -> 'None'` |  |
| `Cerebro.plot` | 方法 | `(self, *args: 'Any', **kwargs: 'Any') -> 'list[Any]'` | Return market replay charts for the most recent run. |
| `Cerebro.run` | 方法 | `(self, **kwargs: 'Any') -> 'list[Strategy]'` |  |

### Strategy

| 成员 | 类型 | 签名 | 说明 |
|---|---|---|---|
| `Strategy.datetime` | 属性 | `` | Shortcut for self.data.datetime to match backtrader behavior. |
| `Strategy.position` | 属性 | `` |  |
| `Strategy.start` | 方法 | `(self)` |  |
| `Strategy.init` | 方法 | `(self)` |  |
| `Strategy.prenext` | 方法 | `(self)` |  |
| `Strategy.next` | 方法 | `(self)` |  |
| `Strategy.stop` | 方法 | `(self)` |  |
| `Strategy.notify_order` | 方法 | `(self, order: 'Any')` |  |
| `Strategy.notify_trade` | 方法 | `(self, trade: 'Any')` |  |
| `Strategy.notify_cashvalue` | 方法 | `(self, cash: 'float', value: 'float')` |  |
| `Strategy.getposition` | 方法 | `(self, data: 'Any' = None) -> 'Position'` |  |
| `Strategy.getdatabyname` | 方法 | `(self, name: 'str') -> 'Any'` |  |
| `Strategy.getpositionbyname` | 方法 | `(self, name: 'str') -> 'Position'` |  |
| `Strategy.setsizer` | 方法 | `(self, sizer: 'Any', name: 'Any' = None) -> 'Any'` |  |
| `Strategy.getsizer` | 方法 | `(self) -> 'Any'` |  |
| `Strategy.getsizing` | 方法 | `(self, data: 'Any' = None, isbuy: 'bool' = True) -> 'float'` |  |
| `Strategy.submit_order` | 方法 | `(self, side: 'int', data: 'Any' = None, size: 'float | None' = None, price: 'float | None' = None, exectype: 'int | None' = None, **kwargs)` | Submit a shared event-driven order through the bound broker. |
| `Strategy.buy` | 方法 | `(self, data: 'Any' = None, size: 'float | None' = None, price: 'float | None' = None, exectype: 'int | None' = None, **kwargs)` |  |
| `Strategy.sell` | 方法 | `(self, data: 'Any' = None, size: 'float | None' = None, price: 'float | None' = None, exectype: 'int | None' = None, **kwargs)` |  |
| `Strategy.close` | 方法 | `(self, data: 'Any' = None, size: 'float | None' = None, **kwargs)` |  |
| `Strategy.cancel` | 方法 | `(self, order: 'Any')` |  |
| `Strategy.order_target_size` | 方法 | `(self, data: 'Any' = None, target: 'float' = 0, **kwargs)` |  |
| `Strategy.order_target_value` | 方法 | `(self, data: 'Any' = None, target: 'float' = 0.0, price: 'float | None' = None, **kwargs)` |  |
| `Strategy.order_target_percent` | 方法 | `(self, data: 'Any' = None, target: 'float' = 0.0, **kwargs)` |  |
| `Strategy.buy_bracket` | 方法 | `(self, data=None, size=None, price=None, plimit=None, exectype=2, valid=None, trailamount=None, trailpercent=None, oargs=None, stopprice=None, stopexec=3, stopargs=None, limitprice=None, limitexec=2, limitargs=None, **kwargs)` |  |
| `Strategy.sell_bracket` | 方法 | `(self, data=None, size=None, price=None, plimit=None, exectype=2, valid=None, trailamount=None, trailpercent=None, oargs=None, stopprice=None, stopexec=3, stopargs=None, limitprice=None, limitexec=2, limitargs=None, **kwargs)` |  |
| `Strategy.addminperiod` | 方法 | `(self, minperiod: 'int') -> 'None'` | Extend the strategy warmup period used before calling ``next``. |
