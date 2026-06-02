# 报告 / 绘图指南

trade-learn 区分 plot 和 report。

| 方法 | 定位 | 内容 |
|---|---|---|
| `plot()` | 行情回放图 | K 线、成交点、权益、成交收益、成交区间 |
| `report()` | 一键 HTML 报告 | summary、实验参数、权益/回撤/交易分布、plot 区块、CSV/XLSX artifacts |
| `Reporter` | 高级报告对象 | 接收 returns / positions / transactions，生成 pyfolio 风格分析 |
| `FactorAnalyzer.report()` | 因子报告 | alphalens 风格分组、IC、多周期预测能力 |

Engine:

```python
cerebro.plot()
cerebro.report("report.html")
```

Lite:

```python
bt.plot()
bt.report("report.html")
```

独立 Reporter:

```python
from tradelearn.report import Reporter

Reporter.from_returns(returns, positions=positions, transactions=trades).report("report.html")
```

## 组合报告

当 `report()` 接收到多个有效数据源和多资产持仓时，HTML 报告会自动使用组合回放视图，而不是把所有资产的 K 线和交易点挤在同一个 OHLC 坐标里。

组合回放视图包含：

| 面板 | 用途 |
|---|---|
| `Equity` | 策略权益、Buy & Hold、峰值、最终值和最大回撤 |
| `Allocation` | 多资产权重堆叠，展示组合暴露如何随时间变化；`Others` 与 `Cash` 使用低饱和颜色降低干扰 |
| `Profit / Loss` | 按退出时间分桶展示平均盈利 / 亏损，灰色背景表示交易数量，hover 可查看交易数、胜负数和最好 / 最差交易 |
| `Trade Activity by Asset` | 按资产展示交易发生时间与交易规模，买卖方向在同一资产行内上下错位，箭头大小按成交额做分位数裁剪和感知缩放，hover 会显示日期与成交信息；默认展示交易最活跃的 8 个资产，可用下拉框切换到 Top 15 或全部资产 |

`Assets` 下拉框会同步控制 `Allocation` 和 `Trade Activity by Asset`。`Equity` 与 `Profit / Loss` 始终保留组合整体视角。
组合回放图的图例位于图表上方左侧，采用浅色描边样式，并保留点击隐藏 / 显示序列的交互。

单资产报告，包括只传入一个数据源的 mapping，仍保留传统 `OHLC / Trades` 视图。
