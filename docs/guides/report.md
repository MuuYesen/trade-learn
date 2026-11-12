# 报告 / 绘图指南

当前源码（2026-11-12）已把领域结果、回测摘要和独立 MLflow 接到共同 Reporter/metrics；HTML、Excel 与 `metric_facts` 保留数值、单位及缺失原因。本文描述未公开的新源码接口，最终发行与平台资格仍以实际产物/安装验收为准；参见[统一改造状态](../concepts/api-boundary.md#pending-unification-20261111)。

DataExplorer 和三个 FactorAnalyzer 的便捷 `report()` 先取得领域结构，再委托 `Reporter.from_analysis`。领域结果按数据、配置与版本复用；数据/配置变化会使缓存失效。相同结果导出 HTML/Excel 不重复分析，Excel-only 不构建 HTML 图。首次领域分析仍由对应分析器负责，Reporter 不训练模型、重跑回测或联网补输入。共同职责与验收规则唯一见[分析结果与单一报告出口](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#unified-reporter-20261111)。

trade-learn 区分 plot 和 report。

| 方法 | 定位 | 内容 |
|---|---|---|
| `plot()` | 行情回放图 | K 线、成交点、权益、成交收益、成交区间 |
| `report()` | HTML / Excel 报告 | 已有结果的摘要、实验参数、权益/回撤/交易表；图表表示依格式说明 |
| `Reporter` | 共同报告对象 | 接收 Stats、returns/positions/transactions 或已完成的领域表格 |
| `FactorAnalyzer.report()` | 因子 HTML / Excel | 复用已有分组、IC、多期限分析；Excel 不生成 HTML 图 |

Engine:

```python
from tradelearn.report import Reporter

strategies = cerebro.run()
Reporter(strategies[0].stats).report("report.html")
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
| `Allocation` | 多资产权重堆叠，展示组合暴露如何随时间变化；hover 汇总当天现金、总暴露和当前下拉数量对应的 Top Holdings；`Others` 与 `Cash` 使用低饱和颜色降低干扰 |
| `Profit / Loss` | 按退出时间分桶展示平均盈利 / 亏损，灰色背景表示交易数量，hover 可查看交易数、胜负数和最好 / 最差交易 |
| `Trade Activity by Asset` | 按资产展示交易发生时间与交易规模，买卖方向在同一资产行内上下错位，箭头大小按成交额做分位数裁剪和感知缩放，竖线标记调仓 / 成交日期，hover 会显示日期与成交信息；默认展示交易最活跃的 8 个资产，可用下拉框切换到 Top 15 或全部资产 |

`Assets` 下拉框会同步控制 `Allocation` 和 `Trade Activity by Asset`。`Equity` 与 `Profit / Loss` 始终保留组合整体视角。
组合回放图共享垂直日期参考线，图例位于图表上方左侧，采用浅色描边样式，并保留点击隐藏 / 显示序列的交互。

单资产报告，包括只传入一个数据源的 mapping，仍保留传统 `OHLC / Trades` 视图。


## 已计算领域结果的共同出口（2026-11-12 源码）

`Reporter.from_analysis(result).report(path)` 接受 Mapping 或对象：`tables` 为非空名称到 DataFrame/Series 的映射；可选 `title`、`metrics`、`metadata`、`figures`。不会调用领域分析器、fit、回测或网络。结果表在构造时复制，适合多次 HTML/Excel 导出；新的分析事实使用新 Reporter。示例：

```python
reporter = Reporter.from_analysis({
    "title": "Factor validation",
    "tables": {"IC": ic_table},
    "metrics": {"coverage": {"value": None, "unit": "ratio", "reason": "no_labels"}},
    "metadata": {"schema_version": "1", "periods": 365, "source": "frozen-validation"},
})
reporter.report("factor.html")
reporter.report("factor.xlsx")
```

表名清理后不得与另一个表重名，也不得占用 metrics/metadata/figures。HTML转义表格文本；Excel保留原数值精度及缺失。`metrics` 中普通标量单位为 unspecified；领域应明确提供 `{value, unit, reason}`。缺失不补零。figures接受已生成Bokeh图形，HTML展示，Excel列明没有图形对应形式，保留所有数值表；自定义sections沿ReportContext/ReportSection。

策略 Reporter 的纯数值导入不加载 Bokeh/Jinja2/XlsxWriter；HTML/图表/Excel按需加载。摘要在同一 Reporter 内按年化 periods 和基准内容复用，返回独立字典；不能通过修改返回值污染下一次导出。当前兼容默认 periods=252，非日频/不同市场须显式指定其合格频率，不能把默认值当自动频率识别。空收益 cumulative_return 为缺失。原Stats摘要与报告历史命名/单位差异仍逐项迁移，不能据领域适配入口存在宣称所有出口已完全同口径。

附件写入只覆盖本次明确生成的同名文件，绝不扫描删除目录中的用户文件或子目录。返回清单仅包含本次生成文件；HTML附带的stats.json保留用于数值核验。默认MLflow上传行为和运行生命周期见[MLflow指南](mlflow-lab-mcp.md)。


策略结果的 `Reporter.metric_facts()` 返回未舍入的 value、unit、reason、periods，HTML表格及stats.json、Excel的metric_facts页共同消费；不是从已格式化文字反解析数值。有效百分比字段与ratio分别标明，缺失保留NA及原因。自定义Excel页不得与保留页或清理后同名页冲突，发现冲突在写入前拒绝，避免覆盖事实。领域结果缺失figure显式Not provided，NaN标not_calculable，数组不能冒充单个标量指标。

安装及本地验收：`scripts/check_research_environment.py --out <directory>`在当前解释器的真实Notebook内核执行合成研究/回测/报告/本地MLflow测试，不调用provider或网络行情；需要research依赖及本地内核端口。正式安装/AI provider/真实数据截图资格另列，不由该合成样例代替。

### 成交盈亏依据

当生产者的 `metric_metadata.profit_factor.pnl_basis` 明确为 `pnlcomm` 时，Reporter 的交易指标使用已有净成交盈亏列；保留原始 `pnl`，不再次扣费。外部表未提供这一声明时沿用其 `pnl`，不猜测含费口径。`metric_facts` 同时记录 `pnl_basis`；LiteStats 透传共同 Stats 的元数据。
