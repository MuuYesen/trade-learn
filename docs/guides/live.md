> 当前正式组合为TL0.3.2/Store0.2.1a4/官方NA2.0.0rc7。独立只读Collector接口另由已发行Store0.2.1a5提供，不改变TL0.3.2 extra的a4锁；精确平台、SHA和真实验收见TL整合正式回执。下文日期化旧候选仅证明其历史，0.3.2之后源码不冒充已发行行为。

# 实盘协议

> **待实现目标（2026-11-11）**：收敛共同策略/流水线与运行产物、历史资格/截面/调仓恢复、数量与在途订单处理；不新增执行引擎或自动扩展venue资格。当前唯一NautilusStore和真实SQS/IAM合同继续有效，讨论中的endpoint+token不是可用接口；本页候选/发行/真实验收状态分别保留。见[统一改造状态](../concepts/api-boundary.md#pending-unification-20261111)。

0.3.0通过内部 `tradelearn.backtest.event_runtime.StrategyRuntime` 协调默认研究、事件回放和外部驱动 paper/live 的策略生命周期；Engine根导出已退出。接入方负责事件调度、行情、权威订单/账户事实和执行保护；Runtime 不启动后台循环。完整调用与限制见 [Runtime](../concepts/runtime.md#正式增量策略runtime)。

## NautilusStore：唯一执行装配入口

通过TL使用cTrader的推荐发行目标命令为 `pip install "trade-learn[nautilus,tl-store]"`。TL的单向optional依赖选择真实发行包tl-store（正式TL0.3.2固定0.2.1a4，本轮未发行034源码固定已发行0.2.1a8，内置cTrader），不要求用户另装adapter。未来适配器按需选择，不默认全部安装。独立NA用户可以只装tl-store及官方NA，不反向依赖TL或商业监控。当前组合仅验CPython3.13 Mac arm64/Linux x86_64；旧公开TL0.2.9没有新extra和统一Store能力，本地候选可装不表示公网PyPI已可用。

当前源码唯一公开Store类是 `tradelearn.engine.stores.NautilusStore`（同一内部实现，正式032历史导入路径不变）。官方适配器仍使用下方的 `NautilusStore(config, data_factories=..., exec_factories=...)`；TL-Store的cTrader使用 `NautilusStore(adapter="ctrader", config=adapter_config)`。随后都沿Cerebro.setbroker/adddata/addstrategy/run，不增加转发Store、第二Runtime或额外事件循环。构造/getdata/getbroker不连接、不读取凭据、不创建持久状态。旧一期DefaultStore及其包装、全局register_adapter入口和NautilusAdapterStore类已退出当前候选；历史验收只证明其原版本。

独立适配器由已安装的 `tradelearn.store_adapters` entrypoint按名称发现，或通过 `NautilusStore.register_adapter(name, factory)` 显式注册可信工厂。注册表只保存工厂，不保存账户实例；未安装、重名、未知/冲突配置明确拒绝。选择独立adapter不能同时另传官方factories、账户或策略组装参数；这些参数由adapter config唯一拥有。官方路径继续核对固定NA版本清单及实际官方factory类型身份，不接受未显式选择的第三方factory。

当前装配契约为API2：工厂接收config，返回 `api_version=2` 的装配映射，而非另一个Store实例。通用NA装配字段为 `config`（官方LiveNodeConfig）、`data_factories`、`exec_factories`、`account_id`（从adapter配置派生）、可选 `strategy_config/state_dir/account_provider/validate_feed`。所有键、NA类型、单账户、factory与client路由键、冲突身份及feed语义在启动前校验。独立factory必须实现官方NA Python factory接口；官方内置Rust factory按其实际公开类型验证，不冒称它继承Python factory基类。API1和未知字段明确拒绝。

已有私有原生profile可通过同一API2装配映射交付其既有driver，以及明确的trader/account/strategy/state身份。driver使用现有NautilusBroker，提供getdata/validate/run/runtime，远程监控时另提供monitor_settings；唯一Store统一管理Cerebro驱动选择、单次运行、通知、stop和Sender退出。driver仍自行执行既有原生准入、账户路由、Node/Runtime drain与dispose，不再包一层Store。此扩展不是新Store类或旧API兼容别名；自营profile与部署不进入公开TL-Store。

官方/独立路径共同禁止外部polling Runtime叠加、混合Store或外来feed。实例、broker、会话和持久身份隔离，失败保留主异常并完成退出。具体账户与订单能力按来源验收，工厂存在不等于实盘资格。

### 官方适配器例子

以下是 **2026-11-08 本地源码新增、尚未发布** 的接口；已发行的 0.2.9 不据此获得 Store。安装目标为 Python 3.13/3.14 上的 `pip install "trade-learn[nautilus]"`；使用本轮源码时须安装对应本地构建包。官方依赖固定 `nautilus_trader==2.0.0rc7`，平台须有相应 wheel。

Store 仅组装配置、订阅、现有 ExecutionHost 与 NautilusBroker。`engine.run()` 在进入原有 live 轮询前选择 NA 驱动；由 NA 回调线程创建唯一 StrategyRuntime，订单仍直接进入现有执行桥。创建 Store、`getbroker()`、`getdata()` 均不连接、不创建 Node/Recorder/意图文件，也不交易。

下面采用官方 Binance Testnet 适配器。配置对象、factory、`BarType` 和字段已在官方 rc7 wheel 上离线核对；真实连接与交易没有在本轮验收。运行前由用户准备自己的 Testnet 配置，代码中的环境变量属于官方适配器连接参数。

```python
import os
from tradelearn import engine as bt
from tradelearn.engine.stores import NautilusStore
from nautilus_trader.common import Environment
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import AccountId, TraderId
from nautilus_trader.adapters.binance import (
    BinanceDataClientConfig, BinanceDataClientFactory, BinanceEnvironment,
    BinanceExecutionClientConfig, BinanceExecutionClientFactory,
    BinanceInstrumentProviderConfig, BinanceSpotMarketDataMode,
)

class MyStrategy(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.average = bt.talib.SMA(self.data.close, timeperiod=self.p.period)

    def next(self):
        ...

provider = BinanceInstrumentProviderConfig(
    load_all=False, load_ids=["BTCUSDT.BINANCE"],
)
connection = dict(
    environment=BinanceEnvironment.TESTNET,
    instrument_provider=provider,
    api_key=os.environ["BINANCE_TESTNET_API_KEY"],
    api_secret=os.environ["BINANCE_TESTNET_API_SECRET"],
)
config = LiveNodeConfig(
    trader_id=TraderId("TL-001"),
    environment=Environment.SANDBOX,
    data_clients={"BINANCE": BinanceDataClientConfig(
        **connection, spot_market_data_mode=BinanceSpotMarketDataMode.Json,
    )},
    exec_clients={"BINANCE": BinanceExecutionClientConfig(
        **connection, account_id=AccountId("BINANCE-001"),
    )},
)
engine = bt.Cerebro()
store = NautilusStore(
    config,
    data_factories={"BINANCE": BinanceDataClientFactory()},
    exec_factories={"BINANCE": BinanceExecutionClientFactory()},
)
engine.setbroker(store.getbroker())
engine.adddata(store.getdata("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL"))
engine.addstrategy(MyStrategy)
engine.run()
```

普通 `bt.Strategy` 不需要日志基类。上例未传 monitor，默认不创建日志器；可选日志接线见下节，用户无需手工创建 Runtime、Node 或 Sender。最终简洁 cTrader 用户例的唯一源码为 [ctrader_monitor.py](../../examples/live/ctrader_monitor.py)；其客户端已接固定受信服务契约；真实服务部署和最终安装资格仍待，不能称已可用于生产远程发送。

### 参数归属与限制

| 入口 | 已实现参数与行为 |
|---|---|
| `NautilusStore(config=None, *, adapter=None, data_factories=None, exec_factories=None, strategy_config=None, state_dir=None, account_provider=None)` | `config` 为官方 `LiveNodeConfig`；factory 的键与对应 client config 的键完全匹配，并官方直接配置路径限定于固定NA版本清单及实际factory类型；独立适配器须显式选择已安装/注册的adapter，并校验API2、NA接口、配置及路由。此检查不是二进制来源认证。官方路径每Store恰好一个暴露account_id的exec config；独立路径由已选择的adapter从其权威配置派生，冲突时拒绝。连接、重连、路由、provider 等参数由官方配置拥有，不在 Store 重复定义。 |
| `strategy_config` | 官方 `StrategyConfig`，省略时为 `StrategyConfig(strategy_id=StrategyId("TL-001"))`。NA 执行身份/OMS 等放这里，业务参数仍放 `addstrategy(MyStrategy, *args, **parameters)`，沿用 TL `params` 与构造参数机制。 |
| `getdata(bar_type=None, *, name=None, **parameters)` | 接收原生 `BarType` 或其字符串；标的、周期、价格类型、聚合来源由其唯一表达。默认 feed 名为 instrument ID。未发行源码接收BT已有的 `tz/sessionstart/sessionend/calendar` 视图元数据；它们不重解释原生UTC输入。每标的仍只有一个原始Bar订阅；标准resample/replay可注册同源派生视图，不新增Tick回调、隐式补空秒或历史回填。 |
| `getbroker()` | 返回该实例唯一的 `NautilusBroker`；启动时直接绑定 ExecutionHost，无 Store 逐笔转发。启动前读取现金/持仓显式失败。 |
| `account_provider(cache, portfolio)` | 可选，沿用执行桥已有权威账户投影；涉及现金/权益读取时需要提供合格投影，缺失即拒绝，不推测余额等于权益。使用资格由具体来源验收。 |
| `state_dir` | 本地状态根目录；按 trader/account/strategy 身份摘要隔离意图与日志。意图复用现有 `IntentJournal` 的独占锁与恢复检查，不是第二订单账本。重启须保留状态并完成原生订单核对，不能清空目录解除未知订单阻断。 |
| `NautilusStore(..., buffer_capacity=512)` | 传递已有 Runtime 滚动容量；策略参数和 `addsizer()` 沿用原入口。 |
| `stop()` / `engine.runstop()` | 请求 NA handle 停止；`run()` 负责等待退出、停止同一 Runtime、drain 持久化任务、dispose Node 并关闭本地日志。异常清理保留主异常。 |
| `get_notifications()` | 消费本实例状态队列，返回 `(message, args, kwargs)`：`STARTING`、Runtime 成功启动后的 `RUNNING`、`STOPPED` 或 `FAILED`。这是 Store 生命周期状态，不是券商连接/重连成功证明。 |

`Cerebro.addstore(store)` 可显式登记；正常 `setbroker`/`adddata` 已自动发现所属 Store，不必额外调用。此前的 `stores` 列表和 `addstore()` 仅保存对象，没有连接能力；旧回测与 polling 入口继续沿原路径运行。

每 Store 单次运行，每账户各用独立 Store/Cerebro。重复运行应新建实例，保留同一持久身份；不使用全局 singleton。混合其他 Store、普通 feed、live poller、paper 驱动、多策略或优化均在创建 Node 前拒绝。流式analyzer、显式observer与SignalStrategy已接入现有Runtime；已有普通原生timer、会话/calendar/DST和历史cheat timer已在源码接通同一Runtime；public Store LiveNode不提供pre-venue-match bar-open阶段，不能据官方BacktestEngine的SimulationModule可用就称该实盘阶段可用。依赖回测Stats的后分析器继续区分运行后统计。未实现的addwriter生命周期明确拒绝，非BT writer查询便捷入口已退出。`run()` 须在没有运行中 asyncio loop 的线程调用；没有另造后台事件循环或异步 facade。NA 路径不生成回测 Stats，不将原生执行当作模拟撮合。

未知 Store、feed、native run 参数显式报错；官方适配器自己校验其字段。不支持的订单类型/控制、缺失账户/标的与恢复冲突继续由现有桥拒绝。特定 venue、账户经济数据及订单能力须逐项验收，不能由 factory 存在推定可交易。

### Backtrader 简洁参数接缝（2026-11-10，本地候选）

唯一入口仍为 `NautilusStore`，不恢复 DefaultStore 或其他同义 Store。
已安装/显式注册的 adapter 可接收 `NautilusStore(adapter=..., **连接参数)`；
这些参数由实际 adapter 严格校验，不能同时给 `config` 或原生装配参数。
官方 adapter 仍使用上方经核对的官方配置，不承诺所有 adapter 共用同一套凭据参数。

adapter 可将 `getdata(dataname=..., timeframe=bt.TimeFrame.Seconds,
compression=1, price="mid")` 编译成同一个原生 BarType；不支持该能力的 adapter
明确要求原生配置，不能静默猜标的或周期。cTrader 对应字段及环境、账户、OAuth
约束见 TL-Store 的组件技术说明；旧公开 TL0.2.9及旧安装组合不支持这一候选入口。

装配协议继续为 API2，当前 interface1 增加两个可选惰性回调 `compile_feed` 和
`close_adapter`，分别只在登记 feed、原生退出清理时调用；不是交易事件转发。
adapter 必须核对接口能力。构造和 feed/broker 登记不启动 Node、连接、读取凭据文件
或后台刷新。显式原生 `CTraderStoreConfig` 保留，与简洁字段互斥，二者进入同一工厂/
会话/Runtime/Broker装配与退出链。`run` 对清理异常保留原主异常。

0.3.0 extra锁定`tl-store==0.2.1a3`，正式命令为
`pip install "trade-learn[nautilus,tl-store]==0.3.0"`，组合须用CPython3.13；
a3已提供Mac arm64/Linux x86_64（glibc2.34+）真实wheel。新增平台及ABI仍独立实施；旧公开0.2.9不具备此入口。

### 实时功能映射现状（2026-11-10候选，尚未完成全量覆盖）

下表核对 TL 当前公开源码与固定官方 NA2.0.0rc7 的已安装公开类型/签名；
是实现盘点，不把 NA 类型存在当作所有 adapter/产品支持。完成标准引用项目系统规范，
本页只登记实际 API 与缺口。以下“cTrader限制”指当前 TL-Store 实现，不推断券商协议本身缺能力。

| TL公开功能/参数 | NA公开映射 | 当前通用桥 | 当前cTrader组件/验证 |
|---|---|---|---|
| Cerebro/adddata/addstrategy/setbroker/run | LiveNode/Strategy/客户端Factory | 已映射，唯一驱动 | 简洁入口与高级入口同一组Factory/会话；Mac安装通过，Linux新ABI待验 |
| Strategy.__init__/start/prenext/next/stop、params/self.p、指标 | 唯一StrategyRuntime由NA回调推进 | 已映射 | 源码/离线生命周期，未据此声明现役新接口已验 |
| addminperiod、OHLCV lines、history_panel | 已交付bar及TL有限可见窗口 | 已映射当前数据，未增加历史请求 | 不做隐式回填；标的类别由券商metadata解析 |
| getdata(dataname,timeframe,compression,price) | InstrumentId/BarType、官方内部time bar | adapter能力编译接缝已映射 | Seconds/Minutes/Days/Weeks/Months/Years与bid/ask/mid；MicroSeconds仅整毫秒可表达。固定NA实际周/月/年边界回放通过；空秒不补，溢出和不足毫秒拒绝 |
| broker.getcash/getvalue/getposition | Cache/Portfolio及显式account_provider | 已映射单账户；多hedged仓须显式路由 | a8候选默认读取同轮cTrader原始free_margin/equity并保来源钟；其他来源须权威provider，不能用balance代equity；新版本组合仍待 |
| buy/sell、submit_order(size,price,exectype) | OrderFactory.market/limit | Market/Limit已有；Stop/StopLimit、Trailing与原生关联组已补源码；最终安装产物仍须按各自清单核验 | **当前仅Market IOC**，并须通过原费证与账户核验；不能宣传Limit已可交易 |
| close | 原有反向减仓意图及NA position_id/reduce_only | 已补typed close：自动核同账户/strategy的唯一仓，指定position_id选择hedged仓；reduce_only不可关闭，尺寸不能超过该仓；沿原Journal持久核路由 | typed close已实现；不得把未限定反向Market称券商close |
| order_target_size/value/percent、target_weights | TL意图计算→同一submit入口 | 公开计算入口保留；依赖资金/仓位与订单类型资格 | 不能绕过缺资金或原仓位/尺寸保护 |
| Stop(price) | OrderFactory.stop_market(trigger_price) | 已补stop_market及触发价恢复核验（本地候选），尚无venue交易验收 | adapter也有独立实现缺口，不等于OpenAPI不支持 |
| StopLimit(price,pricelimit) | OrderFactory.stop_limit(trigger_price,price) | 已补stop_limit双价格及恢复核验（本地候选），尚无venue交易验收 | 同上，需扩协议/报告/撤改单闭包后验收 |
| StopTrail(trailamount/trailpercent) | trailing_stop_market/trailing_offset_type | Mac本地安装候选已映射初始水位/绝对价差/BPS及原始准入恢复；不在TL另追价，Linux新组合待验 | adapter/产品资格待核；不得默认启用NA仿真 |
| StopTrailLimit与限价偏移 | trailing_stop_limit/limit_offset | **0.3.0桥未实现**。BT实际以原price/plimit差形成limitoffset，触发价和限价同步移动；先前称固定执行限价是错误描述。下一版已补绝对回撤/零差比例的真实NA工厂映射及共同核反例，但最终安装/原生更新事件资格尚未完成；非零固定差+比例回撤在固定NA共享offset单位下不等价，不自造追价循环 | 不因有trailing类型就认定当前venue支持 |
| buy_bracket/sell_bracket、parent/transmit/oco | 一次NA关联OrderList，OTO与Bracket子单OUO；独立OCO为原生OCO | Mac本地安装候选缓存未transmit意图于既有Broker投影，完整组校验后一次原生提交；没有TL子单控制循环 | 固定NA模拟器整组拒绝/正常与部分成交回放已验；cTrader实际factory声明不支持组单，提交前拒绝。真实venue待验 |
| cancel(order) | Strategy.cancel_order | 已映射，原生身份核验 | 现有native cancel接口；不得重发原提交 |
| 修改订单 | NA modify_order | TL通用Broker协议声明modify，当前engine.NautilusBroker无公开实现 | 当前adapter也未实现修改；不捏造Strategy.modify API |
| tif/valid | NA TimeInForce/expire_time | day/gtc/ioc/fok已映射；未指定保持GTC。date/datetime/timedelta/相对秒的GTD候选编译与恢复字段保留；固定NA模拟器两项过期边界资格仍未通过，当前新准入在broker/host/group提交前明确拒绝GTD，Market GTD同样拒绝 | 当前只接IOC；不能静默把其他TIF改为IOC |
| size/price精度、账户路由、reduce_only/position_id | Instrument.make_qty/make_price、精确账户/仓位 | 已映射，不允许隐式四舍五入/路由覆盖 | 原centi-unit、账户/strategy/合约归属保护保留 |
| notify_order/部分成交/终态/重复回报 | NA订单事件→TL投影 | 当前实现，跨类型/晚到终态修订仍须逐项闭包 | 现有150项协议基线按原来源复用，不覆盖新增类型 |
| notify_trade | NA权威position事件及官方关闭快照→TL Trade通知 | 后续未发行源码分列原生net realized_pnl、完整同币种佣金及gross投影；只在open/close发BT通知，partial change更新原生事实；翻仓不误读新周期cache，不重算账户现金 | 固定NA真实离线部分平仓/翻仓及缺费/身份负例已验；缺失/混币/冲突保持未知，逆向/调整/完整恢复和真实venue未据此合格 |
| notify_cashvalue | NA权威资金投影→TL既有回调 | 已补：原生Bar推进前，经现有Runtime通知一次权威cash/equity快照；未覆盖callback不额外读取provider，缺失/非有限/布尔值拒绝 | 缺资金不填零 |
| 重启/未决/UNKNOWN | 原IntentJournal→native cache/回报核对 | 已有Market/Limit恢复；Stop/StopLimit/Trailing核原始准入值；组单全部持久准备后一次原生提交，重启核对关联元数据（本地源码验证） | 未知结果不重发、原状态不清；OAuth锁/原子刷新/重启读取保留 |
| Cerebro流式analyzer | 现有Runtime组件回调 | 已补start/on_bar/on_order/on_broker_event/on_trade/on_end/stop及get_analysis；复用TL现有组件辅助函数 | 不生成伪Backtest Stats；需要完整回测Stats的后分析器仍属回测路径 |
| Cerebro observer | Runtime市场/权威订单与position回调 | 显式observer已补；BuySell计本run去重Native OrderFilled，Trades计本run已关闭position，非第二账本 | Value/DrawDown/TimeReturn需权威provider；无provider不自动添回测stdstats，显式资金observer缺源预检拒绝 |
| Cerebro.add_signal/signal_strategy | 现有SignalStrategy与唯一Runtime | 已补配置规范化，保留accumulate/concurrent与原signal计算，不改engine.strats/signals | 原生回放核同Broker、唯一Runtime及typed close，未据此声明所有venue支持 |
| writer、timer/calendar | 独立报告/回测组件；timer/calendar未合格 | 非BT getwriter查询便捷入口已移出Cerebro；不宣称完整writer生命周期 | timer/calendar明确拒绝 |
| 模拟资金/手续费/滑点/撮合、优化、回测Stats | 回测专属 | 回测保留，不转券商功能 | 无NA基础回测单独验收 |
| nextstart、标准add_timer/notify_timer | 实时nextstart已由共同Runtime按首次暖机完成调度；默认委托next | 普通/historical-cheat与native普通timer源码已实现；会话/DST/calendar/runtime注册分组已验，最终统一安装仍待 | 默认/自定义nextstart均归共同business.next且不双计 |

固定版本证据入口：TL `tradelearn/backtest/strategy.py`、`engine/strategy.py`、
`tradelearn_runtime/{broker,execution,store}.py`，NA安装包 `common.OrderFactory`公开签名，
TL-Store `tl_store/ctrader/execution.py` 的实际提交/拒绝分支。
[官方订单说明](https://nautilustrader.io/docs/nightly/concepts/orders/)仅作模型参考，
其nightly能力不能代替固定2.0.0rc7及具体adapter/产品的验证。

当前较早v6候选为TL0.2.9+store.20261110.5 / tl-store0.2.1a2；后续typed close/路由持久核验、GTD字段、资金通知和流式组件已进入未发布TL0.2.9+store.20261110.6本地安装候选。独立验收目录核实际site-packages；新变化52通过、2项固定NA模拟器GTD边界明确xfail（不是通过）。旧Mac130复用未重跑。Linux新native、实盘venue与IC同run自然验收仍未完成。
原生组单交原生client/venue唯一管理，不与 `manage_contingent_orders=True` 策略联动叠加；
当前桥明确拒绝该叠加模式，未启用NA仿真。Bracket退出使用NA原有OUO以处理部分退出时的
保护数量联动。已发送OCO成员、跨instrument组、混合position路由等不能静默重新关联。
原有IntentJournal仅增加组关系元数据，全部成员持久化后才可发唯一组命令；任一准备失败
不发部分订单，未知结果阻断且不自动重发。恢复核对原始NA初始化/组关系；不能从新缓存
缺单推断可重下。`OrderUpdated`数量按原生事实修订TL投影及待成交量，重复同事件异值拒绝。
触发/挂起状态保留真实native_status，不将其当独立Accepted确认，四段计时证据规则不变。

### 可选 SQS 监控

**最终入口（当前源码候选，未进入正式032）**：用 `from tradelearn.engine.stores import NautilusStore` 或 `bt.stores.NautilusStore` 组装 `getdata/getbroker`。普通 `bt.Strategy` 与 `bt.talib.SMA` 不接收私有运行对象或 SDK 基类；只调用 `cerebro.run()`。源码删除公开 Store.run，内部仍为同一官方 NA 宿主、Runtime 和退出流程。`monitor=True` 仅开通用原生宿主的本地日志；私有 driver 本地开关仍未实现，不计支持。

监控用户配置只公开下列两字段。endpoint 是本人已注册受限发送队列的完整 URL；access_key_id/secret_access_key 是该队列的现有受限长期 IAM 发送钥对。示例字符串明确表示本人参数，不是可用生产凭据：

```python
cerebro.run(monitor={
    "endpoint": "https://sqs.ap-southeast-1.amazonaws.com/123456789012/your-registered-queue.fifo",
    "credentials": {
        "access_key_id": "YOUR_RESTRICTED_SENDER_ACCESS_KEY_ID",
        "secret_access_key": "YOUR_RESTRICTED_SENDER_SECRET_ACCESS_KEY",
    },
})
```

用户不手填 receipt_origin/archive_binding_id，不传 session_token 或平台 token；内部已有 STS 兼容并未全局删除。归档绑定必须沿本人 IAM 和精确队列对应的受信服务注册核验，不能猜 URL 或相信未认证响应。**当前源码已接受信发现客户端**：固定 Sentinel TLS 服务的 monitor audience/精确队列挑战及签名证明，校验服务注册返回后启 Sender；WEB 服务已正式部署，既有注册 IAM→真实 STS/当前 PG 精确绑定与重放拒绝已验；同最终目标 wheel 的 Sender/归档原件证明 roundtrip 仍待。未取得可信绑定会明确提示 `TRUSTED_MONITOR_BINDING_UNAVAILABLE`，同一宿主仅保留本地日志，远程发送关闭。不能将本地保护通过冒充二字段远程可用。

内部既有源段/outbox、归档回执和完整原件校验继续保留。仅 SQS ACK 不允许删除原件；缺归档证明保留日志，不改变策略交易逻辑。凭据只留内存，停止时关闭记录器、等待发送器退出，再释放客户端；不记录凭据正文。当前定向验收覆盖普通策略、唯一 logger、真实官方 NA 离线回放、sender 退出和无可信绑定的本地保留；实际服务绑定/真实发送和统一最终 wheel 仍另验。

### 对标与本地验收

参考 [Backtrader IB Store 文档](https://www.backtrader.com/docu/live/ib/ib/) 的连接配置、feed/broker 组装与周期参数分工，以及 [Store 接口源码](https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/store.py) 的生命周期/通知入口。这里独立实现实例隔离，不复制 GPL 实现或 singleton，不替换 TL 参数机制。直接创建第二套 live Runtime、Store 逐笔转发两种方案会重复所有权或增加热路径，均未采用。

实现入口：`tradelearn_runtime/`、`tradelearn_runtime/store.py`、`leafquant_monitor/bootstrap.py`；回归入口：`tests/unit/engine/test_nautilus_adapter_driver.py`、`test_nautilus_store.py` 及 `tests/test_monitor_bootstrap.py`。验收覆盖惰性构造、配置冲突、账户隔离、无 super 的参数/日志装配、官方 NA 离线 Bar 回放、唯一 Runtime/Broker、runstop 及各阶段失败清理。此次只验证本地源码及离线行为，不代表新版 wheel 已发行、真实券商已接通或远程监控已可用。

## 回测 broker 与实盘 broker 的区别

| 项目 | Rust 回测 broker | paper / live broker |
|---|---|---|
| 职责 | 本地撮合、订单推进、portfolio 记账 | 连接外部交易系统，回传订单状态和成交 |
| 状态来源 | Rust 内核内部状态 | 外部 broker / 交易柜台 |
| 成交语义 | 按 bar 和撮合模式确定 | 由真实市场和 broker 回报决定 |
| 策略假设 | `next()` 内发单，后续事件更新状态 | 同样发单，但不能假设立即成交 |
| 接口边界 | 内部 Rust 回测 broker | 私有适配器实现 broker-neutral 协议 |

因此，为了加快回测而优化 RustBroker，不会自动改变实盘 broker；但策略层必须只依赖共同的“意图 + 事件”语义。

trade-learn 不公开 `tradelearn.live` 包。具体券商适配器由独立扩展或部署项目提供。增量策略使用 `tradelearn.engine` 的公开 API；Lite 仍用于批量研究，不能因共享生命周期就推定 `Lite.I` 已支持完整实盘。

## 中性订单协议

实盘适配器应该接收中性订单请求，返回委托确认，并通过事件回流成交、撤单、拒单和状态。这里描述的是框架内部协议，不是用户策略应直接导入的公开 API；具体字段定义见 `docs/internals/contracts.md`。

| 类型 | 关键字段 | 用途 |
|---|---|---|
| `OrderRequest` | `symbol`, `side`, `qty`, `order_type`, `limit_price`, `stop_price`, `tif`, `client_oid` | 策略发出的 broker-neutral 下单请求 |
| `OrderAck` | `client_oid`, `broker_oid`, `accepted_ts` | broker 接受请求后的确认 |
| `Fill` | `broker_oid`, `symbol`, `qty`, `price`, `commission`, `ts` | 成交事件 |
| `PositionSnapshot` | `symbol`, `qty`, `avg_price`, `ts` | 持仓快照 |
| `AccountSnapshot` | `cash`, `equity`, `ts` | 账户快照 |
| `OrderStatusUpdate` | `broker_oid`, `status_str`, `ts`, `replay` | 订单状态更新 |

这些类型只描述跨运行时共同语义，不携带 Backtrader bracket 字段，也不携带某个券商私有状态机。

实盘适配器应同时支持单笔 `order_status(...)` 和批量 `order_statuses(...)`。执行器需要等待委托反馈时，优先批量查询，减少来源查询或 REST proxy 的请求次数。

## BrokerEventPump

内部事件泵用来把外部 broker 的轮询结果标准化成框架事件。普通用户不需要直接使用它；私有适配器作者如需对接 broker 事件，应参考内部契约文档，而不是在策略代码里导入 `tradelearn.core`。

## 策略侧约束

为了让同一套策略可以从回测迁移到 paper / live，建议遵守：

1. 在 `next()` 里表达交易意图：`buy`、`sell`、`close`、`order_target_percent`、`target_weights`。
2. 不假设立即成交，也不要假设发单后持仓立即变化。
3. 通过 `notify_order`、`notify_trade`、Stats 或 broker 事件观察成交结果。
4. 多资产目标权重策略应表达“目标组合”，不要在策略里手写 broker 状态同步。

## 外部适配器与 Nautilus 依赖

券商与执行宿主适配器应作为独立扩展实现：

- 输入：`OrderRequest` / 策略意图。
- 输出：`Fill` / `OrderStatusUpdate` / `PositionSnapshot` / `AccountSnapshot`。
- 框架内：只依赖 `tradelearn.core` 契约和事件泵。

这样可以保持 `tradelearn.backtest` 的 Rust 性能优化和实盘 broker 的连接逻辑互不污染。

[NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 是公开的外部交易引擎。选择它作为执行宿主时，集成层必须安装并导入 `nautilus_trader` 及所需适配器；业务策略可以继续使用 TL API。公共 TL 本身不强制依赖 Nautilus。

0.2.9将通用桥接和可选日志纳入同一发行包；安装 `trade-learn[nautilus]==0.2.9` 才装配固定官方框架，桥接不强制启用商业监控。基础包不导入Na，未装依赖或Python版本不支持时，实际桥接入口明确拒绝。本地自研适配器不随公共包交付，须独立核对是否存在于目标上游版本。接入方应声明并验证支持的版本、平台、订单能力、恢复语义和报告顺序。

## 当前源码状态

| 能力 | 状态与限制 |
|---|---|
| 原生回测、Engine / Lite 研究 | 已有实现；原生回测需匹配的 Rust 扩展 |
| 正式 StrategyRuntime | 0.2.9 覆盖默认研究与增量事件协调；不启动内部 loop |
| Broker 中性契约、内部事件泵 | 接入基础设施；内部类型不是策略 facade |
| 通用只读 observers | capability 1；不含持久传输或商业 SDK |
| Nautilus通用桥 | 包内 `tradelearn_runtime.execution`；可选官方rc7，账户/路由需显式配置，具体venue另验 |
| NautilusStore | 本地新增未发布；`NautilusStore` 按上述 API 组装唯一 NA 驱动，默认不连接，run 时启动 |
| 具体券商接入 | 官方内置或独立扩展；不交付私有cTrader、账户配置或生产策略 |
| 发行与实盘验收 | 单独核验；源码存在不表示已发布或已通过真实券商验收 |

### 本轮源码回归补充（2026-11-12）

NA2.0.0rc7与现有TL-Store原生扩展、当前Python源码组合：93 passed / 6 skipped / 2 xfailed。缺少同Python3.13 TL Rust扩展的6项须由最终wheel安装复验；两项GTD边界仍未合格。此证据不等于新wheel或线上venue验收。

默认及显式`bt.sizers.FixedSize`在原生装配时转换为共享固定数量计算，保stake/tranches，不读取或伪造cash/commission。当前原生Store对其他Sizer启动前明确拒绝；回测仍使用标准BT Sizer协议。需要资金的原生Sizer须先提供可核账户与费用输入后独立资格验收。

### 后续源码的原生专属挂单控制（未发行）

`buy/sell(..., post_only=True)` 沿已有 broker kwargs 接入，必须为布尔值，仅对带执行限价的 Limit/StopLimit/StopTrailLimit 生效。固定NA离线撮合与持久恢复证明已通过，尚未进入新wheel或公共版本；当前0.3.1不能据此使用。OHLCV回测不模拟venue的maker准入政策，显式True会拒绝，不忽略；cTrader当前能力边界及未知/断线恢复门保持。后续源码补入 `display_qty`、`trigger_type`、`emulation_trigger`、`exec_algorithm_id` 与伴随的 `exec_algorithm_params`，仍沿相同BT broker kwargs。display必须为正、精度无损且不超过总量；固定NA的非Market工厂接受该字段（包括StopMarket），但不外推为所有venue都支持冰山。trigger采用固定NA `TriggerType` 对象或名称，算法ID采用字符串或原生ID；固定NA2rc7的算法参数实际要求 `dict[str,str]`，数值不静默转字符串。真实QuoteTick+官方OrderEmulator、配置过的原单转发算法和两子单算法已有源码回放；算法必须预先装入同一官方Node/BacktestEngine，不由策略另外创建执行器。

NA算法派生订单沿官方 `exec_spawn_id` 与缓存聚合API投影到已受理BT父单，累计成交价/完整同币种费用只读原生子单事实，原生主单减量不改变BT请求数量或产生虚假ACK；外来归属、重复分配使总量超过原请求、数量冲突会失败，缺费/混币仍未知。只持有投影，不另存成交账本或创建调度循环。固定NA默认算法取消父单并不保证子单也取消：子单仍活动时保留未完成状态与数量，只有原生全部确认后才报告Canceled。实际算法的子单撤销政策由其原生插件负责；不能把转发/分拆测试外推成任意TWAP、任意venue长期资格。已有SQLite intent保存新字段并复核类型/归属，恢复可从官方已减量主单与子单重建投影；同缓存启动重放已有定向验收，完整进程/断线与最终wheel仍保留验收门。 官方OrderEmulator释放可能把Limit/Stop类转换为Market/Limit；恢复仅在原始初始化、OrderEmulated、转换初始化、OrderReleased的真实同归属/数量/TIF/路由链完整时读取原准入类型、价格与控制proof，不把任意类型变化当合法释放；缺原证明仍拒绝。

这组仅为后续源码，不属于R4安装wheel或已发布032；正式版本、自然venue资格与最终统一发行另核。其他原生专属字段仍按已列缺口处理，不无条件透传。

### 034候选：原生对象关闭与资格门

033的20项CI和候选安装回归曾通过，但新增实际wheel反例发现关闭Store后的NativeHost/SDK日志器循环引用会由工作线程GC析构固定NA的unsendable策略。因此033正式上传已取消、tag保留，不能把它当公开发行；034修复尚待新wheel与exactCI资格。

034源码在业务stop后退役共同broker的原生gateway，完成workers drain和Node dispose后只保留Python诊断；关闭日志器不再持有原生策略的绑定时钟。保留异常时只释放已结束的框架原生帧局部引用，原异常身份、消息、堆栈位置和未决intent事实不变。关闭后读取原生账户/持仓会明确失败，不从诊断对象制造行情或执行事实。用户策略仍是普通BT接口，不接收Host或清理胶水。

新门包含日志开/关、失败退出、真实NA Node、保留business/logger/error及另一线程强制GC；native unraisable现在作为pytest失败。SDK源码仅strategy.py关闭引用与tradelearn.py有限034/API1边界变化，其余16文件与032冻结相同。私有应用driver及自然交易仍由消费者独立验收，不能把源码通过当已部署。


### 未发行源码：权威基金观测与通知批次（2026-11-13）

现有可信 `account_provider(cache, portfolio)` 返回现金/权益的同一账户观测时，可选携带 `fundmode`（严格bool）、`fundvalue` 与 `fundshares`（有限实数，份额不为负）。三个字段须同时存在；缺失不从现金或权益推造单位值/份额，零及负单位值保留。它们必须来自适配器已核的同账户、同币种、同观测边界的权威来源，不是策略手填的收益参数。普通无基金来源的策略继续使用账户权益；显式基金分析或 `notify_fund` 不把未知当零。现有NautilusBroker提供标准BT `get_fundmode/get_fundvalue/get_fundshares` 与同名属性，通知读取一次provider快照；没有新增用户构造参数、份额账本或现金流推算。

该源码桥已用实际固定NA回放与明确合成provider验收；当前cTrader简单配置与NA自然账户事实**没有提供基金份额/NAV来源**，因此该自然能力仍未验，不能由离线合成值推断支持。正式032字节不变，最终统一候选安装另验。

研究Engine多单通知沿BT broker批次表达：同一admission阶段先按提交次序发送全部Submitted，再按原次序发送Accepted；终态/控制事件边界及原生填单顺序不变。此为通知投影，不改变原生撮合、费用或账户事实；不外推NA异步venue通知为同步批次。


### 未发行源码：实时派生Bar视图（2026-11-13）

Store返回的原生Bar订阅可沿普通 `cerebro.resampledata(data, ...)` 或 `cerebro.replaydata(data, ...)` 注册派生视图；先 `adddata(data)` 再注册沿BT clone语义保留raw。唯一NA订阅及原生行情缓存不被派生OHLC替换，同一回调驱动既有有界聚合器和同一Runtime。replay仅替换当前逻辑Bar，持久指标恢复相邻逻辑Bar基线，不能把每个修订累计成新Bar。clone省略名称时沿BT使用空名，显式公共Data名称可以相同；`getdatabyname`选择最后注册的同名Data，内部订单路由按Data身份区分；未知开仓路由不能猜一个clone承接Trade。

当前源码实际NA离线资格为默认Seconds/Minutes的2/7压缩、稀疏/跨午夜、raw＋clone/单视图、有界SMA及订单/Trade数据归属。真实原生订单消费原始源行情，而非重撮合派生Bar；期望价应按该原生行情验证。物理SQLite准入任务按真实Node异步边界运行，没绕过意图落盘。停止/断线/Node结束不是源EOF，不自动冲出未完成Bar。闭合的源交付时点与公开Bar标签分别保留，不能用未来标签补一个不存在的来源事件。

这不是全部实时配置已合格：非默认flags、迟到/重复、calendar/higher-frame已分组实现并按精确源码独立验收；真实有限源EOF生命周期只在有来源完成事实时适用；driver自有派生和串接变换尚未实现。上述源码尚未最终wheel/正式发行，正式032和现役消费者不因此改变。


未发行源码的会话接缝已按同一历史源转换函数投影到实时feed：普通SESSION_START/END timer、NY跨DST、晚注册全局calendar及显式来源覆盖，raw和daily clone已有实际NA/BT对照。`data.date2num/num2date` 与datetime视图保持标准BT语义，原生ns不改。独立验收与最终安装另列；这不外推为configured calendar replay或原生cheat已经完成。原生clock在source同刻触发晚于on_bar；官方模拟器的SimulationModule.pre_process则确有当前输入钩子，模拟器组装缺口与真实venue无法倒放的边界须分开，不将拒绝或缺桥写成上游全面不支持。

### 2026-11-14 配置日历 replay 与 Linux 接入候选

当前源码的 Days/Weeks/Months/Years replay 配置复用同一有界日历状态；source 或晚注册的全局 calendar、原生 session/DST 沿相同 feed 上下文转换。调整后的旧逻辑修订先交付，当前 raw bar 在该源事件的最终阶段才可见，普通策略长度取可见 Data 的逻辑长度。BT 的 feed `p.timeframe/p.compression` 保留源构造参数，实际派生周期在 `_timeframe/_compression`，不再混写。

该组已完成固定 BT 及真实 NA 离线回放的源码资格，仍未进入统一正式 wheel。原生 live late/duplicate/撤回、明确源 EOF 和 native cheat 等剩余条目单列；停止 Node 不作为 EOF，不补造未完成 bar。

IC 所需的普通 `bt.Strategy`、`bt.stores.NautilusStore`、唯一 `cerebro.run(monitor=...)` 已有 Linux CP313 单平台 build-only 候选并完成真实安装/离线 NA 资格。其精确文件与 SHA 由集成回执交消费者；不提供尚未正式发行的 `pip install trade-learn==0.3.4` 命令。当前正式版本与候选能力分列，真实本人 IAM/Sender/归档回执联合仍另验。

实时到达子路径现支持官方 NA 明确准入的重复/迟到源事件与逻辑 replay 标签分离：原始 `RuntimeContext.ts_event_ns` 保持券商/源事件时钟，较新的 BT 逻辑标签不替代它。clone 只保存有界待交付修订和前一可见状态；撤回/恢复按 BT 的 Data/Strategy/Indicator 各自长度规则推进。固定状态指标在同一输入长度重算时恢复 checkpoint，不因另一个 feed 的回合重复消费 raw 值。

这项资格不代表默认 NA 会接收乱序输入，也不代表 idle/check、真实源 EOF 或全部迟到订单经济已完成。旧历史 wall-clock clone/check 红例仍保留；节点停止不能被解释为源 EOF。对 NA 账户，cash、equity、notional/exposure、unrealized PnL 是不同原生事实；尚未完成的 scoped valuation 不用 exposure 填作 margin equity，缺 NAV/share 事实时不派生基金份额。

#### 原生 idle 检查源码状态

当前候选在既有 NA Actor Clock 上注册每个派生视图至多一个单次完成检查；没有新源 bar 时，只完成已观察的聚合，不构造新行情。默认子日 resample、启用边界时间调整的 replay 已有同观察时钟 BT/NA 数据与指标阶段对照；clone 的未来标签仍等 raw master，停止仅取消本 Actor 的内部检查，不视作 EOF。完成观察时钟与原始 source ns 分列，不能用检查时钟改写源行情时间。该源码尚未纳入统一正式 wheel。

订单经济资格分开保留：idle 完成的 resample 上提交 Market 后，原生宿主可能在下一 raw 行情成交，而历史 BackBroker 要等被订购的 derived view 再推进。当前定向反例仍红，不能用数据/指标轨迹相同宣称这一成交时序也相同；真实场所成交也不能由第二撮合器改写。最终候选资格须明确各执行宿主与数据视图适用边界。

#### 查询与执行宿主的资格边界

当前源码的 `broker.getvalue/get_value(datas,mkt,lever)` 已补单币、无杠杆stocklike CASH的单/多/重复Data及global市场视图；MARGIN单Data仅投影原生有符号资产notional，不能称账户权益。显式空list/tuple沿固定BT忽略mkt，None才选择global市场组件。MARGIN多资产资本视图、FX/hedged及自然NAV/units缺来源时仍未知，未以拒绝计支持。

原生Live执行不承诺重现历史BackBroker的模拟成交时序。BackBroker自身按orderedData推进才允许Market撮合；固定BT的live broker公开buy/transmit可以在Data未推进时即时发往endpoint。TL的NA路径同样保唯一NA执行权威，不再用另一BackBroker延迟、重算或改写真实成交。历史回测仍按对应BT模拟契约逐项验收，原始跨宿主差异记录保留。


### 原生撤销成交与恢复（本轮源码候选）

固定官方 NA 的 `OrderFillVoided` 经原订单缓存投影到同一 BT Order。数量和累计费用只读原生事实；原生 `VOIDED` 显式记录并映射 BT Canceled，原生重开保留原 CID 与待成交余量，不重建现金或补造平仓损益。算法子单按原生 spawn 根 ID 汇总到原始父单，持久意图不为子单新建提交权限；修正身份按子单 ID 与 correction ID 隔离，保留旧终态证明。恢复只装载当前缓存并记住有界原始修正事件，旧回报不会在后续真实成交后再次重开。

这组尚未计入正式 TL 安装产物。`PositionAdjusted.apply_adjustment` 的模型能力与真实生产路径分开：固定 Python ExecutionClient 的标准订单回报入口不能接该非订单事件；本次实际 CfdSwapModule 只变更账户状态，没有发该位置回调，不据此自建调整账本或声称所有 venue 已支持。


### 默认账户事实刷新（本轮源码候选）

普通策略无需 SDK 基类来维持默认 cTrader 账户事实。带原生观察时间的 provider 在既有 NA Clock 上触发 `QueryAccount`，实际报表仍由同一账户执行客户端、Session 和原生账户缓存产生。SDK 启用时请求共用该调度，不增加第二轮询器；无实时观察时间的自定义 provider 与历史/离线驱动保持原生命周期。

每次只允许一个在途查询，以新原生账户事件确认响应；无响应时由自有原生 deadline alert 明确失败。原有金额/币种/身份、观察窗口与过期检查不放宽，过期事实不能充作当前余额；停止时取消自有检查和未执行的回调。该源修复尚未计入正式 TL wheel，也不代表自然券商资金刷新或监控归档联合已通过。


### 组件读取同一账户观察（本轮源码候选）

原生 broker 的组件启动与每次 Analyzer/Observer 采样各有一个只读观察作用域：fund 探测和估值共享同一次 provider 返回，离开作用域即丢弃，不跨 bar 缓存、不代替原生账户事实。默认 `fund=None` 仍在启动时读取真实 mode；DrawDown 只将后续实际 bar 的值计入回撤，不把这个配置读取算入收益样本。原生与历史宿主的采样期继续分列。


### Engine 回测数值精度（本轮源码候选）

普通 Engine 的成交数量、输入价格与费用按既定 BT 浮点语义保留；不能无条件截到 6 位而改变仓位、费用或把正数量变成零。模式在同一个 TL Rust broker 绑定点内部选定，包含普通、COC 和其他既有 runner；用户不增加参数，Python 不重算原生成交或账户。直接内核/Lite 的既有数值 profile 与真实 NA venue 的报价/下单精度分别定义，不能借此绕过券商 precision 检查。新 native 的正式发行与平台资格仍待，旧 wheel 字节不覆盖。
