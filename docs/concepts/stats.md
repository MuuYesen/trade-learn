# Stats 结果对象

Engine 和 Lite 都使用同一套 `Stats` 口径。Lite 返回 `LiteStats`，但字段和 Engine 的 `strategy.stats` 对齐。

## 常用字段

| 字段 | 含义 |
|---|---|
| `summary` | 最终权益、收益、交易数、胜率等摘要 |
| `equity` | 权益曲线 |
| `returns` | 收益率序列 |
| `fills` | 成交明细 |
| `trades` | 交易明细 |
| `positions` | 持仓明细 |
| `orders` | 订单明细 |
| `config` | 本次运行配置 |

Lite:

```python
stats = bt.run()
stats["final_value"]
stats.summary
stats.equity
stats.trades
stats.records
stats.strategy
stats.config
```

Engine:

```python
[strategy] = cerebro.run()
stats = strategy.stats
stats.summary
stats.equity
stats.trades
stats.config
```

## lazy stats

大样本回测可使用 `stats_mode="lazy"`。该模式先返回 summary，`equity`、`fills`、`trades`、`positions`、`orders` 等 pandas artifacts 会在访问、report 或 MLflow 上传时再 materialize。
