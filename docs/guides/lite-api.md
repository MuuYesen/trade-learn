# Lite API（源码候选）

类属性定义可运行参数，init建立共享指标依赖。Indicator与Engine为同一对象；orders是活动订单tuple，position只表示成交。I、history_panel、target_equal、target_percent与prepare_data已退役。窗口、订单和迁移例子见[Lite指南](lite.md)。

## `Backtest`

```python
Backtest(data: 'pd.DataFrame | dict[str, pd.DataFrame]', strategy: 'type[Strategy]', cash: 'float' = 10000, commission: 'float' = 0.0, margin: 'float' = 1.0, trade_on_close: 'bool' = False, hedging: 'bool' = False, exclusive_orders: 'bool' = False, holding: 'dict[str, float] | None' = None, trade_start_date: 'str | pd.Timestamp | None' = None, lot_size: 'int' = 1, fail_fast: 'bool' = True, stats_mode: 'str' = 'full', storage: 'dict | None' = None, match_mode: 'str' = 'exact')
```

## `Backtest.run`

```python
Backtest.run(self, **kwargs) -> 'LiteStats'
```

## `Strategy.init`

```python
Strategy.init(self)
```

## `Strategy.buy`

```python
Strategy.buy(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)
```

## `Strategy.sell`

```python
Strategy.sell(self, *, ticker: 'str' = None, size: 'float' = 1.0, limit: 'float' = None, stop: 'float' = None, sl: 'float' = None, tp: 'float' = None, tag: 'object' = None)
```

## `Strategy.close`

```python
Strategy.close(self, *, ticker=None, size=None, **kwargs)
```

## `Strategy.position`

```python
Strategy.position(self, ticker: 'str | None' = None) -> 'Any'
```

## `Strategy.window`

```python
Strategy.window(self, size: 'int', data=None, ago: 'int' = 0)
```

## `Strategy.target_weights`

```python
Strategy.target_weights(self, weights: 'Mapping[str, float] | pd.Series', *, close_missing: 'bool' = True, constraints: 'Any | None' = None) -> 'list[Any]'
```

## `Strategy.close_all`

```python
Strategy.close_all(self) -> 'list[Any]'
```

## `Strategy.record`

```python
Strategy.record(self, name=None, plot=True, overlay=None, color=None, scatter=False, **kwargs)
```
