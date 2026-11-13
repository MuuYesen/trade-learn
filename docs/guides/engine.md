# Engine

本页说明0.3.4源码候选；已发行0.3.2与现役部署的字节和能力另行保留，不能用候选文档外推。公共边界引用[TradeLearn规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#engine-lite-facade-20261112)。Engine对照固定官方Backtrader 1.9.78.123；Lite保留便利外观，共用Runtime、指标状态和执行事实。

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

`Cerebro`仅接收BT参数，未知参数拒绝。本轮未发行源码已将默认Broker现金纠正为固定BT的10000；需要其他现金规模时仍使用`engine.broker.setcash(...)`。`optreturn=True`为默认；优化结果为`[[OptReturn], ...]`，`optreturn=False`返回`[[Strategy], ...]`，optcallback接收一个组合list。`maxcpus=1/None`使用本地顺序执行。历史默认与已配置的timer/calendar/replay/resample有限范围已实现并有固定BT/实际NA证据；自然venue、有限源EOF、native pre-venue cheat与资本来源限制按[实时指南](live.md)分列。cheat_on_open、exactbars、关闭preload、chain/rollover仍未实现，不能用接受签名或拒绝参数宣称兼容。

TL额外能力从独立模块接入：

| 需求 | 入口 |
|---|---|
| 研究窗口 | `tradelearn.backtest.history.window(strategy, size=n, data=strategy.datas)` |
| 组合权重意图 | `tradelearn.backtest.targets.target_weights(strategy, weights)`；或原生`order_target_percent` |
| Runtime/Context | 内部`tradelearn.backtest.event_runtime`；用户无需另启循环 |
| 实时Store | `bt.stores.NautilusStore`（当前源码，正式032旧导入分列），见[实时指南](live.md) |
| 日志 | Store集成配置，见[实时指南](live.md#可选-sqs-监控)；普通`bt.Strategy`透明装配为0.3.2之后源码，旧SDK基类仍保留兼容 |
| 评价/报告 | 普通Analyzer继续；独立Reporter消费已计算结果 |
| MLflow | `tradelearn.integrations.mlflow.log(strategies, research_result=...)`，运行后显式记录 |
| 简单网格 | `tradelearn.integrations.search.grid_search`，不属于BT facade |

Engine根不再导出DataFeed、Params、Runtime、成本类、目标约束、grid_search、日志策略、TV/TDX/PTA或Sizer具体类。Feed使用`bt.feeds.PandasData`，Sizer使用`bt.sizers.FixedSize/PercentSizer/AllInSizer`。PTA退出；指标来源通过`tradelearn.indicators`选择，资格以[指标指南](indicators.md)为准。

策略基类不自动注入研究结果；使用自己的参数与状态显式提供组件/部署bundle，研究组件登记与执行沿[研究指南](research.md)。未实现的多tradeid账本明确拒绝。回测通过不代表NA/券商组合合格；实际支持和拒绝见实时指南。

摘要数值及单位、缺值、逐Bar年化profile见[组件速查](component-usage.md#stats-与报告)。摘要annual_return/volatility不再乘100，max_drawdown保负号；显示百分数由报告负责。

Engine的order_target_size/value按BT已成交仓位与comminfo.getsize计算；Lite组合target_weights独立考虑活动根订单剩余量。resampledata的旧Pandas全量聚合替代已退出。统一收口中的新源码采用一根未闭bar的有界完成协议，Seconds/Minutes默认边界按BT UTC日内锚，Days/Weeks/Months/Years按已验默认UTC日历边界，预载表从逐源bar协议及显式EOF形成；已对照数据/对象/元数据、共享指标暖机、原撮合/现金路径及7/11非整除参数。它当前只接历史bar feed，不能据此宣称live/自定义交易日历/非默认flags或replay完成；这些仍在本轮C2实施范围。

订单通知在下一Broker周期交付Submitted/Accepted/终态快照，回调对象与返回的订单对象不同，使用ref关联。Submitted快照到达时原单可能已成交，回调撤单不能追溯取消已发生的匹配；安全open-order查询复制执行信息。末根Bar提交保留Submitted历史，因无下一周期而不伪造成交通知。执行数量保留方向（卖出为负）。Lite便利层仍委托共同准入/事实，不额外启动执行循环。

批量多Feed按真实时间并集推进，各Feed长度和指标状态仅在它自己的Bar到达时前进；策略len/datetime表示共同事件时钟。主Feed结束后仍处理其他Feed，估值可沿用已知最新价，不补造OHLCV。相对valid的BT换算以目标Feed当前日期为基准，与共同实际提交时钟分别记录；需要绝对期限时显式传入datetime。Engine有效期在目标Feed实际推进时判断，不由无关主Feed提前过期；Market须等目标Feed的BT公开日期严格晚于创建日期，原始子秒/重复bar仍逐条推进。writer生命周期尚未实现，addwriter也明确拒绝。

订单 `created.dt/executed.dt` 使用 BT 日期序数；独立结果 `orders` 表保留 UTC 时间，底层无行情对象不造创建价格。转换实现位于中立 `backtest.clock`，Engine 公开日期助手继续导出。

共享 `Indicator` 用内部首 Line 协议接来源指标，移除其 `to_series/wrap_indicator` 公共方法；普通 Line 的前缀接口保留。自定义指标在构造完成时注册，消费者按依赖构造顺序取得当前事件值，避免持久指标读到前一事件。

`Cerebro.plot()` 尚未实现 BT 图形返回契约，调用明确拒绝；已有结果通过独立 Reporter 导出。

统一收口中的未发行源码已补历史bar子秒时钟：全整秒输入保留既有秒轴；含子秒时将本次运行所有Feed归一到精确int64纳秒轴，公开日期继续BT ordinal，不改持久协议或NA事件时钟。500ms、微秒日期/窗口、混合精度多Feed、有效期量化和重复可交付时点已与固定BT对照。默认跨日session end保留23:59:59.999990及显式EOF；EOF执行tick与新bar估值分开送入同一个TL Rust撮合周期，不改行情OHLCV或另造循环。该能力需要本次匹配的候选TL native，不属于正式032；Ticks、live变换、其他session/calendar及replay仍未因此合格。

broker.setcommission使用BT完整参数入口并规范化percabs；实际原生回测目前合格的是默认股票百分比费用。固定/保证金/乘数/杠杆/利息/分标的费用与任意Python override在准入前明确拒绝，不能接受后忽略。CommInfoBase/CommissionInfo的独立公式仍保固定BT计算语义；内部旧Rust经济profile继续独立保留，不伪装成BT执行资格。


## 后续BT兼容修复（尚未发布，不能外推0.3.0）

已实现的后续源码按固定BT1.9.78.123实际调用核对：`bt.ind`与`bt.indicators`指向现有共同Indicator模块；自定义/Percent/AllIn Sizer沿Strategy/NABroker及显式估算comminfo绑定，估算费用不是券商真实手续费；StopTrailLimit保原price/plimit固定价差，Rust水位改按completed close。这些能力已随正式0.3.1发布，并完成最终公网Linux/Mac定向组合资格；后续新增源码仍单独验收。

`bt.analyzers.DrawDown(fund=None)`的value模式采用有界增量观测；`get_analysis()`为支持属性访问的字典，含`len/drawdown/moneydown/max.len/max.drawdown/max.moneydown`，drawdown为BT正百分比，moneydown保账户货币单位。它与Observer复用公共观测回撤计算，公共Stats/Reporter的有符号比例单位不变。NA路径必须显式权威account_provider，缺值/布尔值/非法初始峰值不补零；该结果是未调整资金流的观测回撤，不证明TWR绩效。fund=True需要合格fundvalue；本轮未发行回测源码已提供无外部现金流profile的固定份额与同一原生权益单位值，含`broker.set_fundmode/set_fundstartval`及`fundmode/fundvalue/fundshares`标准BT视图。尚未合格的资金流/实时fund来源不由TL另造账户账本。后续源码的Returns已补观测权益对数收益、真实日/周/月/年/分钟桶及常量状态计数，并通过固定BT和实时Runtime定向对照；它在stop定稿，尚未包含在0.3.1。0.3.2后续源码还补原生时间BarType周期元数据、Sharpe固定BT的默认周期/无风险换算/方差/年化选项及用户next前账户采样，采用当前桶和常量矩状态。legacyannual、完整fund/时区/会话等仍未闭合；不以明确拒绝计实现。这些尚未包含在公共0.3.1。

Position空仓price按BT归零；`len(Position)`保固定BT的整数尺寸约束，浮点尺寸调用len会按BT抛TypeError，持仓存在性使用`bool(position)`，数量读取`.size`，不截断小数数量。

后续0.3.2候选同时对齐Analyzer/Observer的构造参数和组件start先于业务start。Observer集合名称使用BT的`obsname`，普通`name`保留给用户参数；Analyzer名称使用`_name`，传入Data作构造参数不替换所属Strategy的数据上下文。增量Engine的datetime/date/time采用与预载相同的BT读视图；`on_bar`快照使用所选Data的真实源时间，缺源时间不拿cursor冒充。源码/产物/平台资格分列，当前正式0.3.1不自动获得这些改动。

`v0.3.2`标签之后的本地后续修复另核默认`Line.get()`返回：Engine预载/增量的数据、日期与指标窗口对照固定BT返回`array.array("d")`，Lite仍返回NumPy窗口；只转换返回值，缓冲区和原始时钟不变。该修复尚未进入0.3.2或任何后续正式产物；qbuffer/exactbars、窗口不足和完整Line代数资格不能据此外推。

同属标签之后本地修复的增量暖机，复用共同BT逐Data门槛：未齐数据先`prenext`，门槛处按固定BT触发`nextstart`；`len(strategy)`读取现有Runtime市场步计数，而非主Data长度。调用者明确的同时间原子组已与固定BT对照；普通逐个到达仍保既有输入顺序，不伪称自动排序回放。全局`strategy.datetime`、完整时钟Line与恢复仍未闭合，这些后续源码也不包含在0.3.2正式标签中。

随后本地主时钟读接口修复将增量`strategy.datetime`接到共同Runtime的有界日期视图，不再借用Data0最后一根时间。该视图只有日期，不注册为行情Data、不补OHLC；每个源事件的原始ns和执行采样边界不变。显式完整历史请求仍受prefix预算限制，超限在接收下一根行情前失败。固定BT的当前日期/lag/get已对照；完整日期Line代数、多数据迟到情况下Analyzer分桶与完整恢复仍需继续。这也是0.3.2标签之后的未发行源码。

`v0.3.2`之后的本地Store修复允许普通`bt.Strategy`通过`engine.run(monitor=...)`启用远程日志，通过`monitor=True`只启用同一原生宿主的本地日志。策略仍只写`params/self.p`及标准回调，不接收monitor_config、宿主或SDK基类。Store按运行生成内部observer class属性，原用户类不变；继承的旧SDK factories按身份去重，不启动第二Sender。普通策略默认不记录；旧`TradeLogStrategy`默认本地记录保留兼容。`monitor=True`当前限通用原生宿主，私有driver尚未接通此本地开关；拒绝不计完成。该修复仅源码/离线验证，未包含在0.3.2发行，完整保护接回与IC迁移仍独立验收。

033候选e14f490之后的本地有效期修复单列：Engine的`valid`按固定BT保留绝对ordinal数字；数字0为无到期，不等于`Order.DAY`（timedelta零）。date/datetime/timedelta公开`order.valid`转换成BT日期数字，DAY对照固定BT的23:59:59.009999。回测Market忽略valid，NA路径同样不构造Market GTD，已有`tif="ioc"`仍保留IOC；None/数字0的Limit走无到期GTC。Lite的相对秒契约不变。非Market明确deadline仍受固定NA GTD资格门约束，两个真实失败不绕过。这些后续字节尚不在033候选wheel中。

同批后续源码的Engine `Order`日期常量视图提供固定BT的`Order.DAY == timedelta(0)`，不会修改Lite/core的字符串TIF常量；Engine的原生TIF控制用已有broker kwargs，例如`tif="ioc"`，不从非BT的`Order.GTC/IOC`读取。回调仍是同一共同Order投影，Engine的isinstance/issubclass视图识别该对象，没有另一个订单账本或交易循环。该视图未进入033候选。

后续canonical还补自定义Observer声明`lines=("measured",)`后，在`__init__`之前取得`self.lines/self.l`及同名线，`next`中可写`self.lines.measured[0]`并读lag/get窗口。调度复用现有市场步及策略暖机阶段，默认`prenext/nextstart`委托`next`；流式输出受原capacity限制。多Data输出使用同一Strategy步时钟，预载仅借用既有BatchClock数组，增量沿已有有界日期时钟，不注册假行情Data。固定BT默认预载的单/多Data、暖机、输出窗口已对照；Observer自有minperiod、内部Indicator归属、_stclock及完整集合/内置单位仍待逐项闭合。这些是未发行源码，不能用来宣称033wheel已包含。

上述valid/Order.DAY/Observer与原生关闭生命周期修复现收拢为034源码候选，尚未正式发行。033tag因实物native owner-GC反例已保留但取消上传；新产物必须使用034版本并重新通过相应安装/native门，不复用033发行资格。

后续canonical定向修复Observer内部指标的归属：仍在现有共享图计算与推进，Observer自建SMA不再推迟业务Strategy的`next`；业务自身SMA暖机保持。固定BT `runonce=False`事件模式与TL预载/增量对照4例及嵌套构造上下文恢复已通过；核心定向共84通过。固定BT默认runonce的嵌套Observer指标曾出现IndexError，未用它冒充有效golden；Observer自有阶段/minperiod及完整once/event等价仍未完成。此项晚于034冻结，只是后续源码，未进入034候选wheel。

同一后续源码还以固定BT事件模式对照显式`Cerebro(runonce=False)`的Observer独立`prenext/nextstart/next`、标准`addminperiod`的重叠首样本规则，以及多Data默认Data0时钟和`_stclock=True`的策略时钟；预载和有界增量均复用现有图/时间视图。117项组件/核心定向通过，未进入034冻结wheel；默认runonce嵌套指标、完整Observer集合/内置单位/恢复仍分列，未将有限覆盖写成完整兼容。

统一收口中的未发行源码已补`SharpeRatio(legacyannual=True)`：使用固定BT年度观察profile的首个业务后broker值作初年锚点、Data0年份与年度常量矩，忽略该旧分支不使用的factor/fund等选项；零方差仍按BT抛ZeroDivisionError。33项固定BT/有界流式定向通过。它不宣称支持完整fund/会话profile。独立验收新增Trade费用/分批退出/翻仓及Observer集合反例后，源码补累计原生成交事实投影、翻仓旧Trade关闭/新Trade打开和整数索引/getnames/getitems；83项相关定向通过。StopLimit固定NA订单类无commissions()时，实时与恢复共同读取完整同币种原生fill事件费用；不完整/void/混币保持未知，45项真实NA回放及负例通过。此处均不是034旧候选wheel能力；本轮不再分段正式发布，待有限C1–C8统一验收。

本轮后续源码还修正已有fromdate/todate的UTC-naive比较约定，保留源索引；重复Observer按BT保留全部实例和重复名，索引/同名查找取首项、属性取末项，Analyzer自动后缀保持。原生PendingUpdate/PendingCancel与拒改/拒撤进入同一Order投影，原生type/id/time/reason保存在`order.info.native_event`，拒改不会变成整单Rejected；pending期间复用已绑定的原生只读cache handle支持标准position查询，不借用Strategy getter或复制账户账本。独立反例/实际范围见本轮TARGET-AUDIT与CONSOLIDATED-C2-C4-C6-REPAIRS-V7；源码536关联项通过仍不等于最终新wheel或整轮完成。


### 统一收口中的六个内置Observer字段（源码，未发行）

已公开的 `Value`、`Broker`、`BuySell`、`Trades`、`DrawDown`、`TimeReturn` 后续源码改用固定BT标准线字段：分别为 `value`、`cash/value`、`buy/sell`、`pnlplus/pnlminus`、`drawdown/maxdrawdown`、`timereturn`。DrawDown是BT正百分比；TimeReturn是当前日/周/月/年或显式子日桶的简单收益，首桶以启动账户值为基准，并非每次回调都重置。它们不改变Stats/Reporter的公共比例单位或现金流绩效profile。

`fund=None`读取broker模式，显式True/False按BT选择单位值或账户权益；Broker在fund模式的cash线保留NaN，不伪造单位现金。BuySell标记为对应Data实际已通知执行的价格均值，`barplot/bardist`按该bar高低价偏移；Trades在关闭通知时写毛/净盈亏，`pnlcomm`默认True。原有 `values` 内的成交/交易计数只保留为历史诊断便利值，不替代这些标准价格/盈亏线。实时采样结果与线窗口有界，事件投影仅持常量累计/末值，不复制全量成交历史。

普通用户策略可以使用BT的 `notify_fund(cash,value,fundvalue,shares)`；同一Runtime在业务next前按真实broker采样并通知，Observer/Analyzer使用原时钟阶段。无外部现金流回测profile、六内置字段、fund模式/显式选择、runonce/event、指标暖机、barplot/毛净、时间桶及有界流式已有定向BT对照；真实NA订单回报到BuySell/Trades标记也已验。实时fund通知仍需P05权威fund字段来源；当前NABroker缺该来源时不造fundvalue或份额，不以拒绝算完整live fund支持。资金流、完整会话/回放与最终wheel另验，本节不是已发布032或旧R4wheel能力。


本轮C1后续源码还修正`Strategy.addminperiod`：固定BT把该调用传给策略自身线，其`_periodset`仍按Data/Indicator分别计算prenext/nextstart/next资格，不能把它当全局等待N根bar的控制。Engine已按此修复，直接调用不再强制延后所有feed；Indicator自有addminperiod及Lite Strategy便利暖机保持。36组固定BT实际init/start、无指标/指标2/3、batch/有界stream逐值通过。原生命周期、暖机下单、runstop测试改用真实Indicator声明minimum，保留原断言；不靠调整时间或删门槛掩盖差异。这是源码层，不是已发行032或旧R4候选。


P02当前在途源码已将默认历史`replaydata`接入同一Runtime：原始观察/撮合时点保留，Data与Strategy长度按逻辑bar前进，当前bar/OHLCV/lag/get原位替换；TA/TV/TDX的私有有界bar基线回退用于同bar修订，不全前缀重算。默认历史普通/稀疏/跨日及Seconds/Minutes/Days/Weeks/Months/Years有限profile，EMA时钟/lag/get、原始行情Market/Limit/Stop成交与回报、用户Indicator/Observer阶段已有175项定向对照；另6项checkpoint归属/长前缀修订反例通过。固定BT在目标时间框架排到raw源之前的clone有私有wall-clock filter check；本实现仅捕获该历史准备过程的检查时点用于EOS重置，不替换任何源/策略/执行时间，过去/未来检查时点都有固定BT对照。977项相关Runtime/Engine/原生控制与Trade回归通过。非默认flags、重复/迟到数据、完整自定义会话、全指标checkpoint目录和live replay仍未闭，最终wheel另验。此处为在途源码状态，不能当完整回放资格或现有wheel能力。


P04当前源码已实现既有`Cerebro.add_timer`的bar驱动状态、标准Timer对象/strats转发与`notify_timer`。普通通知在broker回报后、业务/指标本bar推进前；历史cheat走原有bar协调器的pre-match接缝，仍由同一Rust内核撮合，不建立后台线程/第二订单循环。稀疏repeat每次真实输入至多发一次、返回实际计划when，不补发漏过的每个间隔；日期筛选保留固定BT1.9.78.123的mask跳日行为，而非替换成另一日历政策。已验普通/offset/repeat、week/month carry/allow、固定UTC偏移、默认SESSION_START/END、暖机/上一阶段指标和策略时钟、Market/Limit下单，以及有界Runtime普通通知。真实官方NA2rc7 Store回放收3个普通timer通知，零订单；完整自定义session/DST/calendar、用户运行中注册及原生宿主cheat的精确语义仍未闭，不能由本节外推。源码尚未进入最终wheel；现役032不改变。


P05现金/基金增量（未正式发行）：`broker.setcash/set_cash`和标准`broker.add_cash`进入同一个Rust Portfolio。运行中setcash保留BT即时现金/缓存value视图，安全命令边界提交原生；add_cash在下一真实open撮合后按BT估值阶段应用，因此不能给此前的撮合补足资金。基金份额只消费精确尚未完成的原生应用回执，使用上一单位值；重复/冲突/未知回执不重复增加份额。单位值每真实bar采样一次，不建立第二现金账本。stock-like BT提交预算也在同一内核处理，区分提交时资金拒绝与后续实际撮合Margin；没有可信启动价格时不补未来价格或0参考。

真实MacCP313候选R2已有现金/三分析器/份额及coordinator、多feed、replay对照通过，原始回归1064通过/1启动兼容失败；该失败在精确Python源码修复+同R2native下通过，修复尚未进入R2wheel。对应native/Rust测试与分层字节见本轮P05-CASH-CANDIDATE-INSTALL，原红全部保留。按标的subset估值、实时外部fundfacts与最终统一wheel仍另验；本段不表示现役032或R4已获得这些改动。

2026-11-13 用户最终入口覆盖：公开 `from tradelearn.engine.stores import NautilusStore` / `bt.stores.NautilusStore` 惰性引用同一内部实现；源码删除公开 Store.run。普通 `cerebro.run()` 自动选绑定的 NA broker/store，同一宿主承担日志和退出；run 的既有 BT 参数先应用到 Cerebro。公开 monitor 仅 endpoint 与 credentials 的 access_key_id/secret_access_key。可信归档绑定未取得时保留本地日志并明确报错，不启远程 Sender、不凭 SQS ACK 删除原件。当前已验源码，不代表正式032已具备该入口或二字段远程已完成。

P03历史源码增量（未进入当前公共入口wheel）：既有resample五个flags和bar级MicroSeconds、PandasData tz/tzinput/sessionstart/sessionend、calendar提供者与Cerebro.addcalendar已实现所列历史profile。全局日历在feed启动前应用，晚登记也刷新原目标对象且保留Lines引用；clone会话结束委托原源。命名或pandas_market_calendars实例按真实UTC会话表与early close/holiday取值，研究可选依赖固定5.6.0、缓存至多368行，不把源时点替换为业务时间。44日历/边界、14实际Market/费用与6早收市timer对照通过；此前非默认106与时区/session/timer63复用。固定BT+pandas3的命名wrapper类型错误及Tokyo日线EOF私有EOS溢出均保留实证，不能据其他正例宣称该私有路径合格。live变换、非默认/迟到/session replay、原生cheat仍待，最终统一wheel另验。官方提供者接口见[Calendar usage](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html)。

P02后续本地候选已补有序子日非默认flags与 `adjbartime=True` 的逻辑交付回合：旧bar修订可在原始下一条行情到达后、该新bar公开前交付，原始行情时间/价格不改；clone源与其派生回合共享同一调度器，撮合仅由同一Rust账户执行。Market准入与有效期使用该feed公开BT日期，COC引用订单创建时close；报告消费每交付回合实际原生账户采样，不能按重复原始日期重建另一账本。当前只是本地Mac候选资格，非正式034；迟到/重复、调整privatecheck、日历/session/live及完整有限后端容量仍待，不外推全部replay支持。精确范围与原失败见本轮 `C2-P02-ORDERED-ADJUSTED-CANDIDATE.json`。

后续报告/阶段候选进一步将全部历史replay的报告权益接到逐交付原生账户采样，与是否启用 `stdstats` 无关。BT `set_coc(True)` 的订单仍在业务回调提交，但账户应用发生在下一broker回合，使用创建bar的close；该创建bar业务后的Observer不提前扣费。公开订单执行日期保BT创建日期，账户应用和原始报价来源时点分别保留。此为后续本地候选实测，非原aad53候选或正式032字节；精确分层见本轮 `C2-P02-REPORT-PHASE-REPAIR.json`。

迟到/重复与clone排队的后续源码资格：`takelate`按固定BT公开逻辑日期处理，接受的迟到修订增加0.000001 Gregorian day的逻辑标签，原始观察ns/OHLC与实际可得顺序保持。clone可暂退到前一可见bar，排队修订由同一批事件时钟交付；broker turn与source row分别识别，held quote可被后续BT broker回合使用，不能把它误作重复回合。自定义Indicator的输出长度按其BT时钟独立前进，有限持久后端只保相邻两个bar基线，避免撤回/held输入重复推进状态。当前是3bed本地wheel+4文件源码delta资格，非该wheel已经包含后续修复或全replay完成；日历/其他高时间框架/live与完整有限后端目录仍另验。

迟到clone的首个逻辑bar可短暂没有可见值（长度0）；此时不虚构前一bar，也不把空feed作为原生报价。用户可重名Data；公开名称只是别名，内部按显式Data对象分别绑定指标与变化标记。`takelate=False`在调整时间profile中仍可能触发固定BT的历史检查与排队回合，不能把拒绝某行解释为完全没有后续回合。相关修复当前仅本地5文件源码delta（见本轮 `C2-P02-LATE-IDENTITY-REPAIR.json`），正式wheel另验。


### 稀疏数据源的 Indicator 事件模式（0.3.4候选）

`cerebro.run(runonce=False)`遵循固定BT事件模式：每个策略回合按注册顺序调用已就绪Indicator，即使其数据源本回合没有新bar；已有逻辑bar的输出原地修订。默认`runonce=True`的历史批量模式仅推进新到达的数据源，命令式`Indicator.next`副作用次数可能不同，这也是固定BT本身两种模式的区别。共同EMA等状态化后端以该逻辑bar的checkpoint重算当前修订，不把持有次源当新bar累积两次。用户不增加参数、第二Runtime或撮合循环。

Engine成交价格/数量/费用保留BT浮点输入精度；Lite和直接内核保留其既有六位数值profile，两种profile不能靠精确余额相等代替分别资格。COC公开`order.executed.dt`保留创建close的BT日期，原生fills/订单帧另保实际应用turn；Reporter不重写原生时钟。时区解释位于共同核心，Engine与Runtime引用同一实现，内部UTC/ns与源事件均不改。
