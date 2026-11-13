# 更新日志 (Changelog)

本项目的所有重大变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本 (Semantic Versioning)](https://semver.org/spec/v2.0.0.html)。

## [0.3.4] - 2026-11-13

- Close native Store ownership explicitly after drain/dispose: retain Python diagnostics, retire the broker route, and release closed SDK logger clock references. This prevents a disposed unsendable NA Strategy from being dropped by another thread's cyclic GC. Native-owner tests cover real Node disposal, logging on/off, failure identity and retained business/logger objects.
- Match fixed BT validity: numeric0 differs from DAY; dates/durations expose ordinal-day order.valid; Market ignores valid; native Market IOC stays IOC. Engine Order.DAY is timedelta0 over the shared core order projection, while Lite keeps native TIF string constants. Non-Market GTD remains unqualified on fixed NA.
- Bind custom Observer output lines/aliases to the existing bounded Runtime clock; dispatch default BT warmup/nextstart phases and retain each multi-data strategy step without a fake market subscription. Observer-owned indicators/minperiod and full built-in units remain incomplete.
- Version/API1 SDK boundary includes034; Store stays pinned to formal a5. This version does not establish every BT/NA feature or live venue support.

## [0.3.3] - 2026-11-13 (held; never published to PyPI)

- Tag v0.3.3/e14 was held after actual installed-wheel native owner-GC failure; Release43336847550 was cancelled before publishing. Existing031/032/a5 bytes stay immutable.
- Ordinary BT Strategy logging uses Store-owned observer assembly; local-only logging is `store.run(engine, monitor=True)`, remote uses the existing SQS mapping. No SDK superclass or private strategy constructor arguments are required on the common native host. Private application driver migration remains separate.
- Match fixed BT Engine `Line.get()` array return, multi-data warmup/strategy count, bounded strategy datetime reads, and data datetime lag/arithmetic in ordinal-day units; source/event clocks stay unchanged.
- Returns/Sharpe calendar sampling follows Strategy datetime, including late secondary sources, without changing the observed-value compatibility profile or public cashflow metrics.
- Pin standalone Store0.2.1a5 and qualify the SDK finite version/API1 boundary. Existing032/a4 and standalone Storea5 artifacts remain immutable. GTD pinned-NA defects, complete BT Line/Observer/fund semantics and venue limits remain explicit incomplete capabilities.

## [0.3.2] - 2026-11-13

本节记录本版实现与兼容边界；正式发行及平台安装资格以对应版本产物和收据为准。0.3.1与Store0.2.1a4既有正式字节及旧SDK冻结保持。

- Analyzer/Observer继承参数与构造前owner/data绑定；Analyzer的name参数保留给用户，集合名称使用_name，重复实例加序号，集合索引/迭代走共同实现。
- Returns按真实日历时间桶常量计数，输出BT观测权益对数收益字段；Sharpe按当前桶和在线矩状态处理固定BT的周期/无风险利率/样本方差选项，不借Stats历史重算，不改变公共TWR金融口径。legacyannual、完整fund/时区/会话等仍未全部支持。
- 共同Runtime在用户next前采样观察账户值，对齐DrawDown/Sharpe原始BT通知阶段；未新增运行循环。
- 固定NA的BarType周期声明传到惰性订阅及Runtime feed，不猜采样频率；非时间Bar保未知。Engine日期读视图在预载/增量路径共用，内部Unix/ns不变，缺OI保NaN；预载可选Line受prefix限制。
- 既有broker kwargs新增严格布尔post_only，对NA限价族映射，恢复保存和核对不可变标志；固定NA模拟器正反例通过，OHLCV后端不模拟该venue准入策略。cTrader支持范围不外推。
- SDK有限白名单增加0.3.2/API1，其余17文件不变，未知版本/API继续拒绝。完整BT/NA行为、GTD与其他已列缺口仍继续，不以这版候选称整轮完成。

## [0.3.1] - 2026-11-13

正式公共PyPI已发行；Linux/Mac最终公网定向组合及Mac/Linux KQ native23零skip通过，其他功能/平台仍按精确资格范围。

- Engine自定义/百分比Sizer绑定原Strategy和唯一NABroker；新增BT风格委托估算佣金策略，实际成交费用仍以NA事实为准。
- StopTrailLimit保留BT初始price-plimit差值，触发价与限价同步移动；回测核在完成bar的close更新水位。NA桥接金额及零原始价差百分比形式；非零固定价差百分比仍受固定NA单一offset单位限制。
- DrawDown Analyzer在线增量输出BT百分比/货币/嵌套max字段，公共Stats/Reporter金融口径不变；实时Observer采样历史按既有feed容量有界。
- 平仓notify_trade.price保留开仓成本价，成交退出价仍由原订单/Fill事实提供；无新订单账本。
- 补齐ind/indicators模块别名、Position空仓价格及直接len行为；完整BT接口对账仍在继续，Returns/Sharpe与GTD缺口不算已完成。

## [0.3.0] - 2026-11-12

本节记录0.3.0的实现与兼容边界。正式发行与平台资格以版本产物和发行收据为准；已发布0.2.9保留原字节，部署另行验收。

- Engine与Lite共用事件/时钟、持久Indicator、订单和账户事实。Engine保持已核Backtrader外观；明确拒绝未实现参数，不承诺完整BT移植。Lite退出向量化I/未来初始化，七个示例采用有界状态，保原EMA初值与简单TR口径。
- 指标公开来源收敛到固定资格集合TA-Lib/TV/TDX；PTA与重复ta入口退出。KunQuant提供82个Alpha101和158特征的显式TS/STREAM产物；编译、数值、恢复与平台资格分别记录，不用图生成代替数值合格。
- 研究保留Pandas/Dataset/DuckDB、成熟标签与purge、显式组件调用、全部候选/trial、模型manifest和可信冷恢复；Riskfolio仅已核历史协方差、long-only最小方差，换手/成本等未支持参数拒绝。
- gplearn按真实上游边界选择Python3.10的0.4.2与Python3.11+的0.4.3；旧版与现代sklearn仅在自有estimator实例上复用公开验证器，保全部输入检查/候选与可pickle状态，不改第三方类或公式。
- 共同Reporter消费已有事实；修复Pandas3重采样精度、混合时间事实的Excel导出和窄屏报告可读性。运行后MLflow记录及显式模型产物保留，默认本地SQLite，不隐式重算或训练。
- CLI仅new/lab/doctor和help/version；new生成策略、Notebook与_agents/skills；lab只启动并清理自有Jupyter。TL自建MCP、tradelearn mlflow及lab附带服务参数/编排退出，MLflow原生命令仍可显式使用。
- 安装组为default/research/nautilus/tl-store，原lab并入research；默认无研究/UI/云发送依赖，doctor自动识别已装组件。官方NA2rc7保持原包和唯一执行底座。
- TL-Store独立命名统一到tl_store.ctrader及tl_store.ctrader._native，不保永久旧别名；新Store0.2.1a3与TL0.3.0的本机组合和外部发行资格分列。观察SDK明确只接TL0.2.9/0.3.0及observer API1，未知版本/API拒绝。
- tvkit0.18.1有限研究候选已实作验证，none复权、完整历史/Charting契约仍不等价；当前Provider不静默替换，旧通信依赖按实际消费者保留，不新增实盘数据循环。
- Linux wheel最终打包移除ZIP目录条目并重建仅文件RECORD，保持Python/native/许可文件原字节，兼容严格的安装闭包验收。
- 移除公开AutoML薄包装，探索比较迁至examples/exploration/classifier_comparison.py；保各实例训练/预测诊断和连续分数ROC AUC。固定Oracle、metrics覆盖和exact-tag/exact-commit发行门继续保留。

## [0.2.9] - 2026-11-03

### 新增与调整
- 修复 CI 所需依赖及公开数据导入，保留冻结 Oracle 与指标覆盖率门槛；修复行情周期、重复 Bar、TA-Lib 暖机和示例训练输入的已验失败。
- 公开券商无关的 `engine.StrategyRuntime`：统一默认 Cerebro/Backtest（含 Lite 批量 profile）、事件回放和外部驱动 paper/live 的策略构造、生命周期、指标推进与业务通知。Rust 撮合和批量预计算保留；Lite 增量 profile 仍明确拒绝。
- 提供通用只读 strategy observer capability 1，以及显式同时间戳市场批次与异常清理契约；观察接口本身不承担日志投递；包内日志组件按需装配，生产适配器与账户配置独立。
- 默认纳入通用TL↔Nautilus桥、可选策略日志与native Recorder；`[nautilus]` 精确依赖官方2.0.0rc7，这条执行路径标为实验，不等于所有内置venue均通过交易验收。基础导入不要求Na，日志/远程发送显式启用。
- 账户/路由/持仓归属与持久未知提交保护独立于监控装配；固定官方19项内置清单从发行物生成，不包含私有cTrader。
- 修复已知证书/HTTP/IDNA与PyO3依赖漏洞，保全原生依赖原始许可材料；新增精确tag/版本/同commit CI发布守卫。
- 同步多语言首页及公共安装、执行/日志接入说明。API见[Runtime](docs/concepts/runtime.md)、[接入指南](docs/guides/live.md)及[observers](docs/guides/strategy-observers.md)；具体发行产物与日期验收见[记录](docs/reviews/2026-11-03-runtime-release.md)。

## [0.2.7] - 2026-10-09

### 新增
- 通用 `PositionSnapshot` 增加可选 `market_value` 和 `mark_price`，保留原有四参数构造方式；发布 wheel 增加接口验收。

### 修复
- **文档契约审计**：稀疏行情按主时钟到期；普通止损限价只使用触发后价格并跨 bar 保持激活；卖空与后续买入统一检查保证金占用。
- **数据与统计**：修复秒/毫秒/微秒重采样、TDX 分钟/月线歧义、重复行情漏检；缺失价格不再自动补成收益，NaN 传播与年化输入校验按契约执行。
- **研究示例**：ML 使用独立时间留出与标签边界清理；修复 Alpha002/003 未定义相关系数及 Alpha066 除零结果，统一 Alpha101/191 加权归约与非有限窗口规则，历史输出可能因正确性修正改变。
- **验证**：恢复 Rust 测试与 CI、冻结离线 oracle 校验、隔离导入副作用，并修复基准工具退出检测和合成数据。
- **订单历史**：记录真实委托创建时间和成交时间，附带委托价格与 parent/OCO 引用，保留 tag/info；异步数据源按各自游标记录成交，避免引用未来日期。
- **括号订单**：`buy_bracket` / `sell_bracket` 支持以 `stopexec=None` 或 `limitexec=None` 关闭对应侧，返回值保留 `None` 占位，不生成额外市场订单。
- 保持 `exact` 默认撮合模式和已验证的 Rust 订单生命周期。PR #6 中重复的平行实现未替换现有内核。

## [0.2.6] - 2026-09-23

### 新增
- **订单能力**：接入跟踪止损 / 跟踪止损限价、显式 OCO、有效期、括号订单及订单 `tag` / `info` 统计元数据，沿用现有 broker、runner 与 Rust 撮合架构。

### 修复
- **跟踪撮合**：水位按先撮合后更新的顺序单向推进；跟踪限价单遵守触发后的限价约束，跨 bar 保留触发状态，避免引用触发前的限价机会。
- **订单生命周期**：exact 与 smart 均原子处理显式 OCO；各 runner 在撮合前检查 UTC 有效期，并同步撤单、过期和关联子单的终态通知。保留 smart 原有隐式保护性退出行为。
- **兼容性**：保持原有 `Order` 数值常量，同步 Python 与 Rust 包版本为 `0.2.6`。

## [0.2.5] - 2026-07-20

### 优化
- **Alpha101 因子计算**: 为部分滚动计算路径增加 numba 加速，降低沪深300指数增强实盘权重生成的原始因子计算耗时。
- **实盘 API 边界**: 收敛 `engine` / `lite` 公开接口边界，减少策略代码对内部 `backtest` 模块的直接依赖。

### 修复
- **版本一致性**: 同步 Python 包版本、Rust 扩展版本与 wheel 构建测试断言，避免安装后 `tradelearn.__version__` 与发布版本不一致。

## [0.2.4] - 2026-05-05

### 修复
- **输出冗余**: 移除了 `Backtest.run()` 内部硬编码的统计摘要打印，由用户决定显示逻辑，解决了 Jupyter 下输出重复的问题。
- **环境容错**: 优化了 `smart_tqdm` 在 Jupyter 下的加载逻辑，当 `ipywidgets` 不可用时自动降级到文本模式，解决了 `model not found` 的报错。
- **流同步**: 统一了 Jupyter 环境下的标准输出流，彻底解决了输出背景变粉的问题。

## [0.2.3] - 2026-05-05

### 修复
- **进程生命周期管理**: 修复了 `tradelearn lab` 在按下 `Ctrl+C` 后无法彻底关闭 JupyterLab 和 MCP 服务子进程的问题，增加了强制清理逻辑防止端口冲突。
- **Jupyter 审美优化**: 修复了 Jupyter 环境下进度条显示粉色背景和重复渲染的问题，自动切换至 `tqdm.notebook` 蓝色原生进度条。

## [0.2.2] - 2026-05-05

### 新增
- **智能控制台反馈系统**:
    - 实现了 `SmartPbar` (tqdm 子类) 和 `ConsoleState` 来管理终端美学。
    - 添加了“智能间距”逻辑，自动确保执行阶段（数据获取 -> 因果推断 -> 回测运行 -> 报表生成）之间恰好有一个空行。
    - 集成了 `smart_print`，将手动输出与控制台状态跟踪器同步，防止多余或缺失的换行。
- **增强型回测指标**:
    - 引入了 **SQN (System Quality Number)**，用于评估系统的稳健性。
    - 添加了 **凯利准则 (Kelly Criterion)**，提供最优仓位规模建议。
    - 集成了 **市场暴露率 (Exposure Time %)**，追踪资金占用效率。
    - 扩展了回撤追踪，新增 **最大/平均回撤持续时间 (Drawdown Duration)**。
    - 标准化控制台输出为高密度单列排版，确保专业级的投研报告展示。

### 变更
- **终端 UI 精细化**:
    - 将所有旧有的 `print` 和 `logging` 进度指示器替换为同步的 `smart_tqdm` 进度条。
    - 屏蔽了 `tvDatafeed` 的冗余第三方警告，营造更纯净的研究环境。
    - 重构了 `Backtest.run` (Rust 内核)，支持毫秒级执行下的进度条可见性。
- **报告流水线**:
    - 优化了 `Reporter` 序列化逻辑，确保 JSON 输出能够优雅处理 `pandas.Timestamp` 和 `pandas.Timedelta` 对象。
    - 标准化了 `CausalSelector` 报告通过 `to_section()` 的嵌入方式。

### 修复
- 解决了由于流交织导致的进度条 0% 和 100% 状态出现在不同行的问题。
- 修复了 HTML 报告生成中，由于策略包含复杂时间指标导致的 JSON 序列化错误。
- 修正了交易持续时间计算，确保在所有回测引擎中正确捕捉 `dtopen` 和 `dtclose`。

## [0.2.0] - 2026-05-04

### 新增
- **Trade-Learn 2.0 正式发布**。
- 采用 Rust 编写的高性能回测内核。
- 集成用于因子筛选的因果推断套件。
- 使用 Bokeh 构建的现代 HTML 交互式 Tear Sheet。
- 支持多资产截面回测。
