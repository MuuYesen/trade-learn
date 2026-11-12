# Engine

本页说明当前源码候选，未代表已发行0.2.9或现役部署。公共边界引用[TradeLearn规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#engine-lite-facade-20261112)。Engine对照固定官方Backtrader 1.9.78.123；Lite保留便利外观，共用Runtime、指标状态和执行事实。

```python
import tradelearn.engine as bt
from tradelearn.indicators import talib

class Cross(bt.Strategy):
    params = (("period", 20),)
    def __init__(self):
        self.average = talib.SMA(self.data.close, timeperiod=self.p.period)
    def next(self):
        if not self.position and self.data.close[0] > self.average[0]:
            self.buy(size=1)

engine = bt.Cerebro()
engine.broker.setcash(100_000)
engine.adddata(bt.feeds.PandasData(dataname=bars), name="asset")
index = engine.addstrategy(Cross, period=20)
strategies = engine.run()
```

参数继承使用`params/self.p`；初始化用`__init__`。生命周期为`start→prenext→nextstart→next→stop`，`nextstart`默认委托`next`；单Feed通常只调用一次，多Feed只要仍有依赖恰处最小周期边界，按固定BT行为可连续调用。Lines的`[0]/[-1]`与`get(ago=0,size=n)`表示当前、过去和旧到新窗口。日期助手和Engine日期Line公开数字为BT日期序数，内部原行情、撮合及恢复时钟仍为Unix，不迁移旧持久数据单位。

`Cerebro`仅接收BT参数，未知参数拒绝。`optreturn=True`为默认；优化结果为`[[OptReturn], ...]`，`optreturn=False`返回`[[Strategy], ...]`，optcallback接收一个组合list。`maxcpus=1/None`使用本地顺序执行。cheat_on_open、exactbars、关闭preload、timer/calendar/resample/replay/chain/rollover等尚未实现的非默认能力明确拒绝，不能用接受签名宣称兼容。`add_timer`保真实名称并拒绝未实现调度。

TL额外能力从独立模块接入：

| 需求 | 入口 |
|---|---|
| 研究窗口 | `tradelearn.backtest.history.window(strategy, size=n, data=strategy.datas)` |
| 组合权重意图 | `tradelearn.backtest.targets.target_weights(strategy, weights)`；或原生`order_target_percent` |
| Runtime/Context | 内部`tradelearn.backtest.event_runtime`；用户无需另启循环 |
| 实时Store | `tradelearn_runtime.NautilusStore`，见[实时指南](live.md) |
| 日志策略 | `tradelearn.strategy_logging.tradelearn.TradeLogStrategy` |
| 评价/报告 | 普通Analyzer继续；独立Reporter消费已计算结果 |
| MLflow | `tradelearn.integrations.mlflow.log(strategies, research_result=...)`，运行后显式记录 |
| 简单网格 | `tradelearn.integrations.search.grid_search`，不属于BT facade |

Engine根不再导出DataFeed、Params、Runtime、成本类、目标约束、grid_search、日志策略、TV/TDX/PTA或Sizer具体类。Feed使用`bt.feeds.PandasData`，Sizer使用`bt.sizers.FixedSize/PercentSizer/AllInSizer`。PTA退出；指标来源通过`tradelearn.indicators`选择，资格以[指标指南](indicators.md)为准。

策略基类不自动注入研究结果；使用自己的参数与状态显式提供组件/部署bundle，研究组件登记与执行沿[研究指南](research.md)。未实现的多tradeid账本明确拒绝。回测通过不代表NA/券商组合合格；实际支持和拒绝见实时指南。

摘要数值及单位、缺值、逐Bar年化profile见[组件速查](component-usage.md#stats-与报告)。摘要annual_return/volatility不再乘100，max_drawdown保负号；显示百分数由报告负责。

Engine的order_target_size/value按BT已成交仓位与comminfo.getsize计算；Lite组合target_weights独立考虑活动根订单剩余量。resampledata的旧全量聚合替代已退出：尚未实现BT增量Feed收尾与时钟语义，调用明确拒绝，不把Pandas批量重采样称作BT兼容。

订单通知在下一Broker周期交付Submitted/Accepted/终态快照，回调对象与返回的订单对象不同，使用ref关联。Submitted快照到达时原单可能已成交，回调撤单不能追溯取消已发生的匹配；安全open-order查询复制执行信息。末根Bar提交保留Submitted历史，因无下一周期而不伪造成交通知。执行数量保留方向（卖出为负）。Lite便利层仍委托共同准入/事实，不额外启动执行循环。

批量多Feed按真实时间并集推进，各Feed长度和指标状态仅在它自己的Bar到达时前进；策略len/datetime表示共同事件时钟。主Feed结束后仍处理其他Feed，估值可沿用已知最新价，不补造OHLCV。相对valid的BT换算以目标Feed当前日期为基准，与共同实际提交时钟分别记录；需要绝对期限时显式传入datetime。writer生命周期尚未实现，addwriter也明确拒绝。

订单 `created.dt/executed.dt` 使用 BT 日期序数；独立结果 `orders` 表保留 UTC 时间，底层无行情对象不造创建价格。转换实现位于中立 `backtest.clock`，Engine 公开日期助手继续导出。

共享 `Indicator` 用内部首 Line 协议接来源指标，移除其 `to_series/wrap_indicator` 公共方法；普通 Line 的前缀接口保留。自定义指标在构造完成时注册，消费者按依赖构造顺序取得当前事件值，避免持久指标读到前一事件。

`Cerebro.plot()` 尚未实现 BT 图形返回契约，调用明确拒绝；已有结果通过独立 Reporter 导出。

当前批量Rust时间存储仅合格到整秒；500ms等subsecond输入会在截断前拒绝，避免把两事件合成同一秒。常量MicroSeconds/Ticks的名称对齐不代表这些回测时钟已实现；NA实时原生事件仍保纳秒上下文，资格另列。

broker.setcommission使用BT完整参数入口并规范化percabs；实际原生回测目前合格的是默认股票百分比费用。固定/保证金/乘数/杠杆/利息/分标的费用与任意Python override在准入前明确拒绝，不能接受后忽略。CommInfoBase/CommissionInfo的独立公式仍保固定BT计算语义；内部旧Rust经济profile继续独立保留，不伪装成BT执行资格。
