# Engine API（源码候选）

固定对照官方Backtrader 1.9.78.123。签名存在不表示所有参数已合格；未实现能力明确拒绝，支持与迁移见[Engine指南](engine.md)。Cerebro仅接收BT参数，optreturn默认True；附加研究/日志/实时/报告从独立模块接入。

## `Cerebro`

```python
Cerebro(*args, **kwargs)
```

## `Cerebro.adddata`

```python
Cerebro.adddata(self, data: 'Any', name: 'Any | None' = None) -> 'Any'
```

## `Cerebro.addstrategy`

```python
Cerebro.addstrategy(self, strategy: 'type[Strategy]', *args: 'Any', **kwargs: 'Any') -> 'int'
```

## `Cerebro.optstrategy`

```python
Cerebro.optstrategy(self, strategy: 'type[Strategy]', *args: 'Any', **kwargs: 'Any') -> 'None'
```

## `Cerebro.run`

```python
Cerebro.run(self, **kwargs: 'Any') -> 'list[Strategy]'
```

## `Strategy.buy`

```python
Strategy.buy(self, data=None, size=None, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs)
```

## `Strategy.sell`

```python
Strategy.sell(self, data=None, size=None, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs)
```

## `Strategy.buy_bracket`

```python
Strategy.buy_bracket(self, data=None, size=None, price=None, plimit=None, exectype=2, valid=None, trailamount=None, trailpercent=None, oargs=None, stopprice=None, stopexec=3, stopargs=None, limitprice=None, limitexec=2, limitargs=None, **kwargs)
```

## `Strategy.sell_bracket`

```python
Strategy.sell_bracket(self, data=None, size=None, price=None, plimit=None, exectype=2, valid=None, trailamount=None, trailpercent=None, oargs=None, stopprice=None, stopexec=3, stopargs=None, limitprice=None, limitexec=2, limitargs=None, **kwargs)
```

## `Strategy.getposition`

```python
Strategy.getposition(self, data=None, broker=None)
```

## `Strategy.order_target_size`

```python
Strategy.order_target_size(self, data: 'Any' = None, target: 'float' = 0, **kwargs)
```

## `Strategy.order_target_value`

```python
Strategy.order_target_value(self, data: 'Any' = None, target: 'float' = 0.0, price: 'float | None' = None, **kwargs)
```

## `Strategy.order_target_percent`

```python
Strategy.order_target_percent(self, data: 'Any' = None, target: 'float' = 0.0, **kwargs)
```

## `date2num`

```python
date2num(dt, tz=None)
```

## `num2date`

```python
num2date(x, tz=None, naive=True)
```
