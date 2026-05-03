# Report / Plot Guide

trade-learn 区分 plot 和 report。

| 方法 | 定位 | 内容 |
|---|---|---|
| `plot()` | 行情回放图 | K 线、成交点、权益、成交收益、成交区间 |
| `report()` | 一键 HTML 报告 | summary、实验参数、权益/回撤/交易分布、plot 区块、CSV/XLSX artifacts |
| `Reporter` | 高级报告对象 | 接收 returns / positions / transactions，生成 pyfolio 风格分析 |
| `FactorAnalyzer.report()` | 因子报告 | alphalens 风格分组、IC、多周期预测能力 |

Engine:

```python
cerebro.plot("plot.html")
cerebro.report("report.html")
```

Lite:

```python
bt.plot("plot.html")
bt.report("report.html")
```

独立 Reporter:

```python
from tradelearn.report import Reporter

Reporter.from_returns(returns, positions=positions, transactions=trades).report("report.html")
```
