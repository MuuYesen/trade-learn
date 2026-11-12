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
| 日志 | Store集成配置，见[实时指南](live.md#可选-sqs-监控)；普通`bt.Strategy`透明装配为0.3.2之后源码，旧SDK基类仍保留兼容 |
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


## 后续BT兼容修复（尚未发布，不能外推0.3.0）

已实现的后续源码按固定BT1.9.78.123实际调用核对：`bt.ind`与`bt.indicators`指向现有共同Indicator模块；自定义/Percent/AllIn Sizer沿Strategy/NABroker及显式估算comminfo绑定，估算费用不是券商真实手续费；StopTrailLimit保原price/plimit固定价差，Rust水位改按completed close。这些能力已随正式0.3.1发布，并完成最终公网Linux/Mac定向组合资格；后续新增源码仍单独验收。

`bt.analyzers.DrawDown(fund=None)`的value模式采用有界增量观测；`get_analysis()`为支持属性访问的字典，含`len/drawdown/moneydown/max.len/max.drawdown/max.moneydown`，drawdown为BT正百分比，moneydown保账户货币单位。它与Observer复用公共观测回撤计算，公共Stats/Reporter的有符号比例单位不变。NA路径必须显式权威account_provider，缺值/布尔值/非法初始峰值不补零；该结果是未调整资金流的观测回撤，不证明TWR绩效。fund=True还需要真实fundvalue；后端未提供的fund事实不由TL另造账本。后续源码的Returns已补观测权益对数收益、真实日/周/月/年/分钟桶及常量状态计数，并通过固定BT和实时Runtime定向对照；它在stop定稿，尚未包含在0.3.1。0.3.2后续源码还补原生时间BarType周期元数据、Sharpe固定BT的默认周期/无风险换算/方差/年化选项及用户next前账户采样，采用当前桶和常量矩状态。legacyannual、完整fund/时区/会话等仍未闭合；不以明确拒绝计实现。这些尚未包含在公共0.3.1。

Position空仓price按BT归零；`len(Position)`保固定BT的整数尺寸约束，浮点尺寸调用len会按BT抛TypeError，持仓存在性使用`bool(position)`，数量读取`.size`，不截断小数数量。

后续0.3.2候选同时对齐Analyzer/Observer的构造参数和组件start先于业务start。Observer集合名称使用BT的`obsname`，普通`name`保留给用户参数；Analyzer名称使用`_name`，传入Data作构造参数不替换所属Strategy的数据上下文。增量Engine的datetime/date/time采用与预载相同的BT读视图；`on_bar`快照使用所选Data的真实源时间，缺源时间不拿cursor冒充。源码/产物/平台资格分列，当前正式0.3.1不自动获得这些改动。

`v0.3.2`标签之后的本地后续修复另核默认`Line.get()`返回：Engine预载/增量的数据、日期与指标窗口对照固定BT返回`array.array("d")`，Lite仍返回NumPy窗口；只转换返回值，缓冲区和原始时钟不变。该修复尚未进入0.3.2或任何后续正式产物；qbuffer/exactbars、窗口不足和完整Line代数资格不能据此外推。

同属标签之后本地修复的增量暖机，复用共同BT逐Data门槛：未齐数据先`prenext`，门槛处按固定BT触发`nextstart`；`len(strategy)`读取现有Runtime市场步计数，而非主Data长度。调用者明确的同时间原子组已与固定BT对照；普通逐个到达仍保既有输入顺序，不伪称自动排序回放。全局`strategy.datetime`、完整时钟Line与恢复仍未闭合，这些后续源码也不包含在0.3.2正式标签中。

随后本地主时钟读接口修复将增量`strategy.datetime`接到共同Runtime的有界日期视图，不再借用Data0最后一根时间。该视图只有日期，不注册为行情Data、不补OHLC；每个源事件的原始ns和执行采样边界不变。显式完整历史请求仍受prefix预算限制，超限在接收下一根行情前失败。固定BT的当前日期/lag/get已对照；完整日期Line代数、多数据迟到情况下Analyzer分桶与完整恢复仍需继续。这也是0.3.2标签之后的未发行源码。

`v0.3.2`之后的本地Store修复允许普通`bt.Strategy`通过`store.run(engine, monitor=...)`启用远程日志，通过`monitor=True`只启用同一原生宿主的本地日志。策略仍只写`params/self.p`及标准回调，不接收monitor_config、宿主或SDK基类。Store按运行生成内部observer class属性，原用户类不变；继承的旧SDK factories按身份去重，不启动第二Sender。普通策略默认不记录；旧`TradeLogStrategy`默认本地记录保留兼容。`monitor=True`当前限通用原生宿主，私有driver尚未接通此本地开关；拒绝不计完成。该修复仅源码/离线验证，未包含在0.3.2发行，完整保护接回与IC迁移仍独立验收。
