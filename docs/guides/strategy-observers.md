# Strategy observers

本页定义0.2.9本地候选的通用策略观察API（`OBSERVER_API_VERSION = 1`）。它是Python domain API，不是日志wire协议，不提供商业SDK、持久化、发送器或账户金融计算。发布与外部SDK联合验收是独立步骤。

## 唯一运行协调

`engine.StrategyRuntime`负责策略构造、setup/init/start、数据与指标推进、warmup、业务通知和stop。默认Cerebro/Backtest/Lite研究路径与增量事件路径均委托同一协调器；Rust撮合、预载数组和纯批量指标一次预计算仍是优化后端，不是另一个策略生命周期。批量研究profile允许初始化时编译完整指标数组，业务回调只见当前prefix。增量profile可接外部执行器；Lite.I继续属于批量研究，不据此宣称Lite完整实盘。

匹配后端及事件源决定实际输入次序。默认模拟可能在当前bar推进后同步交付匹配报告，外部执行器则独立交付异步报告。观察器记录真实发生顺序，不缓存/延迟/重排业务回调。比较两种运行的trace时，必须同时固定行情/时钟及执行报告输入和撮合假设；相同OHLCV不保证现实成交顺序或收益相同。

## 装配与过滤

```python
from tradelearn.engine import Strategy
from tradelearn.core.strategy_events import ObserverSpec, OBSERVER_API_VERSION

assert OBSERVER_API_VERSION == 1

def create_observer(strategy):
    # This callable is supplied by an external application/SDK.
    return ObserverSpec(
        observer=application_observer,
        events=frozenset({'lifecycle.start', 'business.next',
                          'intent.submit_order', 'business.notify_order'}),
        phases=('after', 'error'),
        context=validated_run_context,
        error_handler=application_error_handler,
    )

class ExternalBase(Strategy):
    strategy_observer_factories = (create_observer,)

class UserStrategy(ExternalBase):
    def __init__(self):
        self.sent = False  # Does not have to call the external base __init__.
    def next(self):
        if not self.sent:
            self.sent = True
            self.buy(size=1)
```

这是接缝示意，`application_observer`、context校验与持久实现由外部提供，不是某个SDK薄基类的实现。Runtime在构造并绑定策略后、初始化前按MRO从基类到派生类收集`strategy_observer_factories`，同一个factory仅装配一次；用户覆盖`__init__`或`notify_order`且不调用super，不会漏装，也不会让薄基类重跑业务callback。factory只返回callable或`ObserverSpec`，不负责推进数据/指标或调用生命周期。无效factory/配置在启动前显式抛错，不自动降级。

运行前/运行间隙可调用`strategy.add_observer(callback, events=None, phases=('after','error'), context=None, error_handler=None)`，返回不透明订阅句柄；`remove_observer(handle)`移除。Runtime也提供同名转接。events是精确domain名称集合，None订阅全部；phases可选before/after/error。不得在观察回调内新增/移除订阅或重入框架。无观察器或没有匹配过滤器时，不生成快照/完整日志。

## 事件与来源

不可变`StrategyEvent`包含`name/phase/details/context/error_type/error_message`。

- `lifecycle.setup/init/start/stop`：TL生命周期；`lifecycle.closed`表示清理结束（即使业务stop失败也仅一次）。
- `business.pre_next/prenext/next/notify_order/notify_trade/notify_cashvalue`：TL业务调用。notify中的状态只是业务收到的镜像，不是第二份权威成交。
- `intent.submit_order/cancel_order`：TL下单/撤单意图，before在执行接口调用前、after在返回后。after提交包含返回order快照以关联执行端已分配的标识；同步broker可能在intent.after之前发业务notify，观察器保留该原次序。

`details.args/kwargs`是只读值快照，可能含已有`runtime_context`。Order.info中的原client_oid等关联字段保留，不生成新ID。`runtime_context`的symbol、源时间、原事件快照来自实际已交付输入；sequence只用于本次进程调用顺序，重启会重置，绝不能用作持久event_id、历史revision或成交去重键。它不是hook观测墙钟时间。

run/environment/业务身份由外部显式context绑定；缺失保持缺失，不推测live、不重置已有source/strategy/account/framework身份。库不产生wire event_id/revision，不把本地订阅句柄当业务ID。实盘submitted/accepted/rejected/canceled/filled权威事实由执行适配器产生，外部SDK应按其既有身份/修订/去重契约接续；本API不自动保证商业端到端重放去重。

快照不暴露可写Strategy、Order、原生行情对象。内置提取现有字段及slots/扩展类型的受控只读属性，保留原raw/precision等源值，不猜单位、不换汇、不补零。对象可实现同步`__strategy_snapshot__() -> Mapping`显式提供domain字段，结果仍被递归冻结；不能返回可写原对象。无法完整快照或超过嵌套预算会形成可见诊断，不以半份隐式记录冒充完整证据。外部SDK负责最终wire映射、字段资格与缺失NA语义。

## 只读与错误策略

观察器及error handler必须同步；async函数注册拒绝，返回awaitable也记为错误，不创建事件loop或等待网络。SDK可以同步完成其本地durable接收，再由自己的发送器异步传输；本库不实现这条链。

观察期间Runtime步进、策略业务方法（包括用户override的next/start/stop/notify）、买卖撤单、指标写入/重算及其他框架变更入口受guard约束。类声明期的幂等guard只检查观察上下文，不再派一次业务，不改正常方法签名或订单行为。观察器仅拿冻结事件；trusted Python扩展仍可持有外部引用，本API不宣称能沙箱化任意反射或恶意属性赋值。

业务callback执行一次。callback失败时发error观察并重新抛原异常；观察器/快照/error handler失败不替换主异常、不回滚已执行订单、不阻止真实execution fact持久化。错误可通过`strategy.observation_failures()`读取：累计total、omitted计数和最近64条诊断；首次失败有RuntimeWarning，error handler可接外部告警政策。诊断环形保留不是源事件durable存储，累计/截断明确可见；真正本地持久失败处理、补传、告警和停止政策属于外部SDK/宿主，不应把本接口当可靠发送队列。

## 市场批次与异常结束

默认批量研究在同一timestamp把全部active bars合步后推进指标和一次next；逐事件Runtime在每次调用后推进，不能把相同OHLCV视为相同调度输入。两个feed都已暖机时，逐A/B交付会真实看见A3/B2，再看见A3/B3；观察器必须如实记录。调用方可显式使用`runtime.step_market_batch([('A', bar_a), ('B', bar_b)])`声明同时间戳原子批次；完整验证先于任何feed推进，批次只触发一次共同业务调度。Runtime不隐式等待别的feed，标准live映射继续消费原到达顺序。严格对照还必须给相同执行回报顺序及匹配假设，不重排回报来使trace相同。

初始化完成/部分启动后由Runtime负责清理，stop和closed恰一次；构造失败尚未取得策略实例时不强行调用stop。业务/启动失败A与清理失败B同时发生时保留A对象和原traceback，B仅作为受保护附加诊断；正常结束时stop失败仍向调用方抛出。清理中禁止新业务订单，但必要撤单及已到达权威执行事实不因此丢弃。observer及error handler失败不替换上述主异常。

本页为通用接口候选，不是TradeLogStrategy TL版成品。既有Nautilus SDK入口由其所有者维护；SDK方后续用capability 1和冻结TL候选包进行适配/联合验证。

## 外部依赖候选识别

实际Python distribution名为`trade-learn`（wheel文件前缀`trade_learn`），import名为`tradelearn`。本地候选是`trade-learn==0.2.9`与`OBSERVER_API_VERSION == 1`；旧0.2.8没有本页完整默认调度/观察合同。尚未发布到公共包源，不应仅写普通公共源安装命令便声称取得候选。

2026-10-27本机可安装候选为`trade_learn-0.2.9-cp312-cp312-macosx_11_0_arm64.whl`，SHA-256 `ca243bfdb06d2ba81119121a60a9e35bbedacba466f583b9dd2e6d7f77d0abc9`。仅适用macOS arm64 / CPython3.12，不是Windows或Linux wheel。调用方从已核交付清单取得文件，在隔离环境本地安装并核Python/Rust版本一致；SDK适配方同时冻结能力版本和完整候选源码摘要，实际TL SDK薄类仍待其实现及联合确认。此哈希标识本次候选，不取代后续发行版本管理。
