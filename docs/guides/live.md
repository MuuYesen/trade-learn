> 正式公共发行组合：TL 0.3.0 / TL-Store 0.2.1a3 / 官方NA 2.0.0rc7；实际实现与native统一为`tl_store.ctrader` / `tl_store.ctrader._native`。TL标签v0.3.0源af47b786与Store私有源8364acf的实际OIDC上传均成功。最终公网安装与逐wheel SHA以TL整合IC-FORMAL-RELEASE-CONTRACT.json为准，部署/venue自然运行独立验收；下方日期化旧候选仅证明各自历史。

# 实盘协议

> **待实现目标（2026-11-11）**：收敛共同策略/流水线与运行产物、历史资格/截面/调仓恢复、数量与在途订单处理；不新增执行引擎或自动扩展venue资格。当前唯一NautilusStore和真实SQS/IAM合同继续有效，讨论中的endpoint+token不是可用接口；本页候选/发行/真实验收状态分别保留。见[统一改造状态](../concepts/api-boundary.md#pending-unification-20261111)。

0.3.0通过内部 `tradelearn.backtest.event_runtime.StrategyRuntime` 协调默认研究、事件回放和外部驱动 paper/live 的策略生命周期；Engine根导出已退出。接入方负责事件调度、行情、权威订单/账户事实和执行保护；Runtime 不启动后台循环。完整调用与限制见 [Runtime](../concepts/runtime.md#正式增量策略runtime)。

## NautilusStore：唯一执行装配入口

通过TL使用cTrader的推荐发行目标命令为 `pip install "trade-learn[nautilus,tl-store]"`。TL的单向optional依赖选择真实发行包tl-store（当前0.2.1a1候选，内置cTrader），不要求用户另装adapter。未来适配器按需选择，不默认全部安装。独立NA用户可以只装tl-store及官方NA，不反向依赖TL或商业监控。当前组合仅验CPython3.13 Mac arm64/Linux x86_64；旧公开TL0.2.9没有新extra和统一Store能力，本地候选可装不表示公网PyPI已可用。

唯一公开Store类是 `tradelearn_runtime.NautilusStore`。官方适配器仍使用下方的 `NautilusStore(config, data_factories=..., exec_factories=...)`；TL-Store的cTrader使用 `NautilusStore(adapter="ctrader", config=adapter_config)`。随后都沿Cerebro.setbroker/adddata/addstrategy/run，不增加转发Store、第二Runtime或额外事件循环。构造/getdata/getbroker不连接、不读取凭据、不创建持久状态。旧一期DefaultStore及其包装、全局register_adapter入口和NautilusAdapterStore类已退出当前候选；历史验收只证明其原版本。

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
from tradelearn_runtime import NautilusStore
from tradelearn.strategy_logging.tradelearn import TradeLogStrategy
from nautilus_trader.common import Environment
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import AccountId, TraderId
from nautilus_trader.adapters.binance import (
    BinanceDataClientConfig, BinanceDataClientFactory, BinanceEnvironment,
    BinanceExecutionClientConfig, BinanceExecutionClientFactory,
    BinanceInstrumentProviderConfig, BinanceSpotMarketDataMode,
)

class MyStrategy(TradeLogStrategy):
    def __init__(self):
        ...

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

`TradeLogStrategy` 在运行时复用同一个原生宿主日志器，工具包生成运行 ID、环境、账户/策略身份及本地日志上下文；用户无需 YAML 或手工创建 Runtime、Node、Sender。该例只记录本地日志，不启动远程 Sender。改用 `bt.Strategy` 则不创建日志器。

### 参数归属与限制

| 入口 | 已实现参数与行为 |
|---|---|
| `NautilusStore(config=None, *, adapter=None, data_factories=None, exec_factories=None, strategy_config=None, state_dir=None, account_provider=None)` | `config` 为官方 `LiveNodeConfig`；factory 的键与对应 client config 的键完全匹配，并官方直接配置路径限定于固定NA版本清单及实际factory类型；独立适配器须显式选择已安装/注册的adapter，并校验API2、NA接口、配置及路由。此检查不是二进制来源认证。官方路径每Store恰好一个暴露account_id的exec config；独立路径由已选择的adapter从其权威配置派生，冲突时拒绝。连接、重连、路由、provider 等参数由官方配置拥有，不在 Store 重复定义。 |
| `strategy_config` | 官方 `StrategyConfig`，省略时为 `StrategyConfig(strategy_id=StrategyId("TL-001"))`。NA 执行身份/OMS 等放这里，业务参数仍放 `addstrategy(MyStrategy, *args, **parameters)`，沿用 TL `params` 与构造参数机制。 |
| `getdata(bar_type=None, *, name=None, **parameters)` | 接收原生 `BarType` 或其字符串；标的、周期、价格类型、聚合来源由其唯一表达。默认 feed 名为 instrument ID。当前每标的限一个 Bar 订阅，不新增 Tick 回调，不做隐式重采样、补空秒或历史回填。 |
| `getbroker()` | 返回该实例唯一的 `NautilusBroker`；启动时直接绑定 ExecutionHost，无 Store 逐笔转发。启动前读取现金/持仓显式失败。 |
| `account_provider(cache, portfolio)` | 可选，沿用执行桥已有权威账户投影；涉及现金/权益读取时需要提供合格投影，缺失即拒绝，不推测余额等于权益。使用资格由具体来源验收。 |
| `state_dir` | 本地状态根目录；按 trader/account/strategy 身份摘要隔离意图与日志。意图复用现有 `IntentJournal` 的独占锁与恢复检查，不是第二订单账本。重启须保留状态并完成原生订单核对，不能清空目录解除未知订单阻断。 |
| `NautilusStore(..., buffer_capacity=512)` | 传递已有 Runtime 滚动容量；策略参数和 `addsizer()` 沿用原入口。 |
| `stop()` / `engine.runstop()` | 请求 NA handle 停止；`run()` 负责等待退出、停止同一 Runtime、drain 持久化任务、dispose Node 并关闭本地日志。异常清理保留主异常。 |
| `get_notifications()` | 消费本实例状态队列，返回 `(message, args, kwargs)`：`STARTING`、Runtime 成功启动后的 `RUNNING`、`STOPPED` 或 `FAILED`。这是 Store 生命周期状态，不是券商连接/重连成功证明。 |

`Cerebro.addstore(store)` 可显式登记；正常 `setbroker`/`adddata` 已自动发现所属 Store，不必额外调用。此前的 `stores` 列表和 `addstore()` 仅保存对象，没有连接能力；旧回测与 polling 入口继续沿原路径运行。

每 Store 单次运行，每账户各用独立 Store/Cerebro。重复运行应新建实例，保留同一持久身份；不使用全局 singleton。混合其他 Store、普通 feed、live poller、paper 驱动、多策略或优化均在创建 Node 前拒绝。流式analyzer、显式observer与SignalStrategy已接入现有Runtime；依赖回测Stats的后分析器、TL尚未实现的timer/calendar调度仍明确拒绝。未实现的addwriter生命周期明确拒绝，非BT writer查询便捷入口已退出。`run()` 须在没有运行中 asyncio loop 的线程调用；没有另造后台事件循环或异步 facade。NA 路径不生成回测 Stats，不将原生执行当作模拟撮合。

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
| broker.getcash/getvalue/getposition | Cache/Portfolio及显式account_provider | 已映射单账户；多hedged仓须显式路由 | 资金投影须权威provider，不能用balance代equity；缺失仍拒绝 |
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
| notify_trade | NA权威position事件→TL Trade通知 | 当前映射；不另造财务执行事实 | 须结合具体产品/费用证据验收，非模拟成交 |
| notify_cashvalue | NA权威资金投影→TL既有回调 | 已补：原生Bar推进前，经现有Runtime通知一次权威cash/equity快照；未覆盖callback不额外读取provider，缺失/非有限/布尔值拒绝 | 缺资金不填零 |
| 重启/未决/UNKNOWN | 原IntentJournal→native cache/回报核对 | 已有Market/Limit恢复；Stop/StopLimit/Trailing核原始准入值；组单全部持久准备后一次原生提交，重启核对关联元数据（本地源码验证） | 未知结果不重发、原状态不清；OAuth锁/原子刷新/重启读取保留 |
| Cerebro流式analyzer | 现有Runtime组件回调 | 已补start/on_bar/on_order/on_broker_event/on_trade/on_end/stop及get_analysis；复用TL现有组件辅助函数 | 不生成伪Backtest Stats；需要完整回测Stats的后分析器仍属回测路径 |
| Cerebro observer | Runtime市场/权威订单与position回调 | 显式observer已补；BuySell计本run去重Native OrderFilled，Trades计本run已关闭position，非第二账本 | Value/DrawDown/TimeReturn需权威provider；无provider不自动添回测stdstats，显式资金observer缺源预检拒绝 |
| Cerebro.add_signal/signal_strategy | 现有SignalStrategy与唯一Runtime | 已补配置规范化，保留accumulate/concurrent与原signal计算，不改engine.strats/signals | 原生回放核同Broker、唯一Runtime及typed close，未据此声明所有venue支持 |
| writer、timer/calendar | 独立报告/回测组件；timer/calendar未合格 | 非BT getwriter查询便捷入口已移出Cerebro；不宣称完整writer生命周期 | timer/calendar明确拒绝 |
| 模拟资金/手续费/滑点/撮合、优化、回测Stats | 回测专属 | 回测保留，不转券商功能 | 无NA基础回测单独验收 |
| nextstart、标准add_timer/notify_timer | 实时nextstart已由共同Runtime按首次暖机完成调度；默认委托next | timer调度尚未实现，add_timer明确拒绝 | 默认/自定义nextstart均归共同business.next且不双计 |

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

本地源码已支持 `store.run(engine, monitor=...)` 直接使用受限 SQS 发送配置。`endpoint` 是完整队列 URL；`credentials` 是该队列的 AWS 发送凭据，字段沿现有发送器使用 `access_key_id`、`secret_access_key`、可选 `session_token`。区域从队列 URL 读取，不使用默认 AWS 身份链，不访问 Sentinel 领取配置，也不要求平台 token。

```python
# 接入配置由服务方提供；实际秘密只放在用户私有配置中。
store.run(engine, monitor={
    "endpoint": sqs_queue_url,
    "credentials": restricted_sender_credentials,
    "receipt_origin": trusted_receipt_origin,
    "archive_binding_id": archive_binding_id,
})
```

`receipt_origin`、`archive_binding_id` 沿既有归档证明协议提供，不是交易账户参数，不能猜测或省略后放松原件清理条件。使用 `TradeLogStrategy` 时，Store 自动生成运行 ID、环境和策略/账户上下文，原生宿主与 TL observer 共用一个日志器；后台 Sender 复用原有文件 outbox、源段格式、重试和回执校验。仅 SQS ACK 不允许删除原件，缺归档证明保留本地日志。用户不用手工创建 Runtime、Node、Sender。

连接规格的构造只做语法校验；启用时先检查 SDK 依赖，再创建本地记录和后台发送。凭据仅保留在内存，停止时先关闭记录器、等待发送线程退出，再释放发送客户端；日志不包含发送凭据。IAM 资格只能由实际 AWS 请求确认，本轮以合成凭据/队列客户端验证本地链路，没有调用真实 AWS 或修改 IAM。发送失败保留重试状态和日志，不控制策略交易逻辑。

此前讨论的普通平台 `token` 不作为 SQS 身份，`store.run(engine, monitor={"endpoint": ..., "token": ...})` 会明确拒绝；本轮没有新增平台 HTTP 接收服务或 Bearer 自动引导。

数据说明：仅上传运行日志，不上传策略源码、券商密码、交易API密钥或OAuth凭据。

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

`buy/sell(..., post_only=True)` 沿已有 broker kwargs 接入，必须为布尔值，仅对带执行限价的 Limit/StopLimit/StopTrailLimit 生效。固定NA离线撮合与持久恢复证明已通过，尚未进入新wheel或公共版本；当前0.3.1不能据此使用。OHLCV回测不模拟venue的maker准入政策，显式True会拒绝，不忽略；cTrader当前能力边界及未知/断线恢复门保持。其他原生专属字段仍按已列缺口处理，不无条件透传。
