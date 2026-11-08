# 实盘协议

0.2.9通过公开 `engine.StrategyRuntime` 协调默认研究、事件回放和外部驱动 paper/live 的策略生命周期。接入方负责事件调度、行情、权威订单/账户事实和执行保护；Runtime 不启动后台循环。完整调用与限制见 [Runtime](../concepts/runtime.md#正式增量策略runtime)。

## NautilusStore：Cerebro 组装入口

以下是 **2026-11-08 本地源码新增、尚未发布** 的接口；已发行的 0.2.9 不据此获得 Store。安装目标为 Python 3.13/3.14 上的 `pip install "trade-learn[nautilus]"`；使用本轮源码时须安装对应本地构建包。官方依赖固定 `nautilus_trader==2.0.0rc7`，平台须有相应 wheel。

Store 仅组装配置、订阅、现有 ExecutionHost 与 NautilusBroker。`engine.run()` 在进入原有 live 轮询前选择 NA 驱动；由 NA 回调线程创建唯一 StrategyRuntime，订单仍直接进入现有执行桥。创建 Store、`getbroker()`、`getdata()` 均不连接、不创建 Node/Recorder/意图文件，也不交易。

下面采用官方 Binance Testnet 适配器。配置对象、factory、`BarType` 和字段已在官方 rc7 wheel 上离线核对；真实连接与交易没有在本轮验收。运行前由用户准备自己的 Testnet 配置，代码中的环境变量属于官方适配器连接参数。

```python
import os
from tradelearn import engine as bt
from nautilus_trader.common import Environment
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import AccountId, TraderId
from nautilus_trader.adapters.binance import (
    BinanceDataClientConfig, BinanceDataClientFactory, BinanceEnvironment,
    BinanceExecutionClientConfig, BinanceExecutionClientFactory,
    BinanceInstrumentProviderConfig, BinanceSpotMarketDataMode,
)

class MyStrategy(bt.TradeLogStrategy):
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
store = bt.stores.NautilusStore(
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
| `NautilusStore(config, *, data_factories, exec_factories, strategy_config=None, state_dir=".tradelearn/nautilus", account_provider=None)` | `config` 为官方 `LiveNodeConfig`；factory 的键与对应 client config 的键完全匹配。当前每 Store 恰好一个暴露 `account_id` 的 exec config，账户直接从该配置读取。连接、重连、路由、provider 等参数由官方配置拥有，不在 Store 重复定义。 |
| `strategy_config` | 官方 `StrategyConfig`，省略时为 `StrategyConfig(strategy_id=StrategyId("TL-001"))`。NA 执行身份/OMS 等放这里，业务参数仍放 `addstrategy(MyStrategy, *args, **parameters)`，沿用 TL `params` 与构造参数机制。 |
| `getdata(bar_type, *, name=None)` | 接收原生 `BarType` 或其字符串；标的、周期、价格类型、聚合来源由其唯一表达。默认 feed 名为 instrument ID。当前每标的限一个 Bar 订阅，不新增 Tick 回调，不做隐式重采样、补空秒或历史回填。 |
| `getbroker()` | 返回该实例唯一的 `NautilusBroker`；启动时直接绑定 ExecutionHost，无 Store 逐笔转发。启动前读取现金/持仓显式失败。 |
| `account_provider(cache, portfolio)` | 可选，沿用执行桥已有权威账户投影；涉及现金/权益读取时需要提供合格投影，缺失即拒绝，不推测余额等于权益。使用资格由具体来源验收。 |
| `state_dir` | 本地状态根目录；按 trader/account/strategy 身份摘要隔离意图与日志。意图复用现有 `IntentJournal` 的独占锁与恢复检查，不是第二订单账本。重启须保留状态并完成原生订单核对，不能清空目录解除未知订单阻断。 |
| `Cerebro(buffer_capacity=512)` | 传递已有 Runtime 滚动容量；策略参数和 `addsizer()` 沿用原入口。 |
| `stop()` / `engine.runstop()` | 请求 NA handle 停止；`run()` 负责等待退出、停止同一 Runtime、drain 持久化任务、dispose Node 并关闭本地日志。异常清理保留主异常。 |
| `get_notifications()` | 消费本实例状态队列，返回 `(message, args, kwargs)`：`STARTING`、Runtime 成功启动后的 `RUNNING`、`STOPPED` 或 `FAILED`。这是 Store 生命周期状态，不是券商连接/重连成功证明。 |

`Cerebro.addstore(store)` 可显式登记；正常 `setbroker`/`adddata` 已自动发现所属 Store，不必额外调用。此前的 `stores` 列表和 `addstore()` 仅保存对象，没有连接能力；旧回测与 polling 入口继续沿原路径运行。

每 Store 单次运行，每账户各用独立 Store/Cerebro。重复运行应新建实例，保留同一持久身份；不使用全局 singleton。混合其他 Store、普通 feed、live poller、paper 驱动、多策略或优化，以及尚未接入 NA 生命周期的 Cerebro analyzer/observer/writer/timer/calendar 均在创建 Node 前拒绝。`run()` 须在没有运行中 asyncio loop 的线程调用；没有另造后台事件循环或异步 facade。NA 路径不生成回测 Stats，不将原生执行当作模拟撮合。

未知 Store、feed、native run 参数显式报错；官方适配器自己校验其字段。不支持的订单类型/控制、缺失账户/标的与恢复冲突继续由现有桥拒绝。特定 venue、账户经济数据及订单能力须逐项验收，不能由 factory 存在推定可交易。

### 监控复用接缝

`run(monitor={"endpoint": ..., "token": ...})` **尚未实现**，现明确抛出 `NotImplementedError`。现有 Sender 仍依赖 SQS/AWS 受限凭据与已有授权配置；尚无已验证的 Bearer 自动引导，不能以 endpoint/token 替代发送权限。Store 复用原生宿主日志器和 Runtime observer 接缝，后续认证引导应在该装配边界接入，不在交易热路径或策略回调里另造 Sender，也不修改后端授权。

数据说明：仅上传运行日志，不上传策略源码、券商密码、交易 API 密钥或 OAuth 凭据。

### 对标与本地验收

参考 [Backtrader IB Store 文档](https://www.backtrader.com/docu/live/ib/ib/) 的连接配置、feed/broker 组装与周期参数分工，以及 [Store 接口源码](https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/store.py) 的生命周期/通知入口。这里独立实现实例隔离，不复制 GPL 实现或 singleton，不替换 TL 参数机制。直接创建第二套 live Runtime、Store 逐笔转发两种方案会重复所有权或增加热路径，均未采用。

实现入口：`tradelearn/engine/stores/`、`tradelearn_runtime/store.py`；回归入口：`tests/unit/engine/test_nautilus_store.py`。验收覆盖惰性构造、配置冲突、账户隔离、无 super 的参数/日志装配、官方 NA 离线 Bar 回放、唯一 Runtime/Broker、runstop 及各阶段失败清理。此次只验证本地源码及离线行为，不代表新版 wheel 已发行、真实券商已接通或远程监控已可用。

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
| NautilusStore | 本地新增未发布；`bt.stores.NautilusStore` 按上述 API 组装唯一 NA 驱动，默认不连接，run 时启动 |
| 具体券商接入 | 官方内置或独立扩展；不交付私有cTrader、账户配置或生产策略 |
| 发行与实盘验收 | 单独核验；源码存在不表示已发布或已通过真实券商验收 |
