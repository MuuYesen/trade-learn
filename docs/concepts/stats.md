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

`summary["final_unrealized_pnl"]` 对回测结束时的所有未平仓头寸按各自最后一个有效收盘价估值，再汇总多空盈亏。没有未平仓头寸时为零。有未平仓头寸缺少有效收盘价时为 `NaN`。`positions` 保留逐笔成交时的持仓快照。

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
