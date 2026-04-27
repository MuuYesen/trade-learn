# Project Overview

基于当前项目代码目录:

- `/Users/muyesen/MAIN/Project/Personal/trade-learn-release`

这份文档按**重构完成后的目标形态**整理,当前阶段(阶段 -1)尚未开工,大部分目录还是 1.x 状态。

## 1. 项目总览

trade-learn 2.0 是 Python 量化研究框架,定位定位详见本文愿景与路线图章节。

主要模块分工:

- `tradelearn/core/` — 契约(所有模块的公共类型)
- `tradelearn/data/` — 行情数据(opentdx / tvdatafeed)
- `tradelearn/indicators/` — `ta.*` / `ta.tdx.*` / `ta.tv.*` 三命名空间
- `tradelearn/factor/` — 因子评估(融合 alphalens)
- `tradelearn/metrics/` — 指标真源(融合 empyrical)
- `tradelearn/report/` — 策略报告(融合 pyfolio + quantstats)
- `tradelearn/backtest/` — Cerebro + Strategy + Analyzer + Rust 撮合核
- `tradelearn/ml/` — ML 能力(MLStrategy / Feature Store / causal)
- `tradelearn/mcp/` — MCP server(给 LLM 客户端暴露 docstring)
- `tradelearn/compat/backtrader/` — backtrader 兼容层
- `tradelearn/brokers/` — 实盘 Broker(1.1:QMTBroker)
- `tradelearn/lab/` — CLI + 项目模板
- `backtest-rs/` — Rust workspace(撮合核源码)

## 2. 分支与代码副本

### 主分支

- `master` — 保留 1.x 稳定版(不动)
- `v2` — 重构主干,所有新工作都在这上面

### 1.x 副本(reference)

阶段 0 Week 1 执行:

```
git checkout master
git checkout -b v2
cp -r tradelearn/ reference/
git mv reference/tradelearn reference/tradelearn_1x
git add reference/
git commit -m "chore: freeze 1.x as reference oracle"
```

`reference/tradelearn_1x/` 作为**活 oracle**:
- 所有金标数据由它生成
- CI 里跑一致性测试对照它
- 1.0 发版前保留,发版后可删

## 3. 依赖方向

```
             core/ (契约)
                 ↑
        ┌────────┼────────┬────────┐
        │        │        │        │
      data/  indicators/ metrics/  core 子包
        │        │        │
        └────────┼────────┘
                 ↓
             factor/ (调 metrics)
                 ↓
             report/ (调 metrics)
                 ↓
             backtest/ (调 metrics + report + core)
                 ↓
              ml/ (调 backtest)
                 ↓
             compat/backtrader/ (调 backtest)
                 ↓
              mcp/ (读取全部,只读)
                 ↓
              lab/ (CLI)
```

规则:

- **core 不依赖任何上层**
- **metrics 不依赖任何上层**(是指标真源)
- 上层可以依赖下层,反向不允许
- 所有循环依赖禁止(CI 强制检查)

## 4. 跨语言边界

- **Python ↔ Rust**:通过 PyO3 + Apache Arrow(zero-copy)
- Rust 构件:`backtest-rs/` 编译产物 `_core.*.so` 放到 `tradelearn/backtest/`
- 用户不直接接触 Rust,所有 API 在 Python 层

## 5. 参考源头(署名要求)

所有上游署名写入 `NOTICE` 文件:

| 来源 | 协议 | 融合/参考方式 |
|---|---|---|
| empyrical | Apache-2.0 | 代码融合进 `metrics/` |
| alphalens | Apache-2.0 | 代码融合进 `factor/` |
| pyfolio | Apache-2.0 | 代码融合进 `report/` |
| quantstats | Apache-2.0 | HTML 模板融合进 `report/` |
| [MyTT](https://github.com/mpquant/MyTT) | MIT | `ta.tdx.*` 算法源 |
| [DolphinDB wq101alpha](https://github.com/dolphindb/DolphinDBModules/blob/master/wq101alpha/README_CN.md) | — | alpha101/191 公式参考 |
| pyneCore | 查确认 | `ta.tv.*` 后端 |
| backtesting.py | AGPL-3.0 | **只参考不复制**(Clean-Room) |
| backtrader | GPL-3.0 | **API 兼容**(只读公开文档,不读源码) |
| causallearn | MIT | 代码融合进 `ml/causal.py` |
| mlflow | Apache-2.0 | pip 依赖 |
| pandas-ta-classic | MIT | pip 依赖 |

## 6. 数据流(研究期典型)

```
1. Query.history_ohlc(engine, symbol, start, end)
   │
   ├→ 命中缓存? → 读 ./data/{engine}/{symbol}_{range}.parquet
   └→ 未命中 → 走 opentdx / tvdatafeed → 写缓存
   │
   返回 Bars(MultiIndex(ts, symbol), OHLCV, 前复权默认)
   │
2. ta.* / ta.tdx.* / ta.tv.* 算指标
   │
3. class MyStrategy(Strategy):                  # 严格 backtrader 风格
       params = (('fast', 10), ...)
       def __init__(self):
           self.ma = ta.sma(self.data.close, period=self.p.fast)
       def next(self):
           if self.ma[0] > self.data.close[0]:  # [0] = 当前
               self.buy()
   │
4. cerebro = Cerebro()
   cerebro.adddata(bars)
   cerebro.addstrategy(MyStrategy)
   cerebro.broker.setcash(1_000_000)
   cerebro.addanalyzer(MLflowAnalyzer, experiment="my_exp")
   stats = cerebro.run()
   │
5. Rust 撮合核(_core.so)事件循环
   │
   → 产出 stats(含 Returns/Trades/Equity)
   │
6. MLflowAnalyzer 自动汇总并上报到
   https://mlflow.leafquant.com
   │
7. Reporter(stats).html("report.html")         # 用户查看
```

## 7. MCP 数据流(AI 代码协作)

```
用户在 JupyterLab Chat 面板问问题
   │
   ▼
Jupyter AI 判断需要查 trade-learn API
   │
   ▼
调 MCP tool: search_api / get_api_docs
   │
   ▼
MCP Server 读源码 docstring(实时)
   │
   ▼
返回结构化文档给 Jupyter AI
   │
   ▼
AI 生成正确代码 → 用户点 Insert to Notebook
   │
   ▼
Shift+Enter 执行(和普通 notebook 一样)
```

**MCP 只读 docstring,不执行任何 trade-learn 代码**。

## 8. 契约对象(core/)

所有模块只认以下 7 个公共对象:

| 契约 | 形态 | 生产者 | 消费者 |
|---|---|---|---|
| `Bars` | DataFrame MultiIndex(ts, symbol) OHLCV | data / user | indicators / backtest |
| `Factor` | DataFrame MultiIndex(ts, symbol) → value | indicators / ml | factor / ml |
| `Signal` | DataFrame MultiIndex(ts, symbol) → weight | ml / strategy | backtest |
| `Returns` | Series time → pnl | backtest | metrics / report |
| `StreamBar` | @dataclass(ts, symbol, OHLCV) | data | 流式 indicators(未来)|
| `Experiment` | MLflow run 抽象 | tracking 内部 | MLflowAnalyzer |
| `Broker` | Protocol(place/cancel/positions/...)| SimBroker / QMTBroker | backtest / brokers |

详见阶段 0 的 `docs/specs/CONTRACTS.md`。

## 9. 测试分层

```
tests/
├── unit/               # 纯单元,每个模块独立
├── integration/        # 跨模块集成
├── consistency/        # 新版 vs 1.x reference 对照(重构期关键)
├── golden/             # 金标基线数据 + 固化 expected
└── compat/
    └── backtrader/     # 10 个 backtrader 开源策略迁移验证
```

CI required check:

- `tests/unit/` 全通过
- `tests/consistency/` 全通过(决策层 0 差异,数值 rtol 分层)
- `interrogate` ≥ 90% docstring 覆盖率
- `pytest --doctest-modules`(docstring 示例必须可跑)

## 10. 外部服务

| 服务 | URL | 用途 |
|---|---|---|
| MLflow | https://mlflow.leafquant.com | 实验追踪(开发/生产共用) |
| PyPI | https://pypi.org/project/trade-learn/ | 包分发 |
| GitHub | (仓库待建) | 源码 + CI + Release |
| 文档站 | (域名待定,建议 docs.tradelearn.io) | mkdocs-material 托管 |

## 11. 与 1.x 的关系

- `reference/tradelearn_1x/` 冻结,不改动
- 无现有用户,**无需保持 API 兼容**
- `reference/` 在 CI 里跑金标对照(新版必须和它数值一致)
- 1.0 发版后 `reference/` 可删除或归档

## 12. 主要里程碑

见 [PROGRESS.md](./PROGRESS.md) 和 [PROJECT.md](./PROJECT.md) 关键时间节点章节。

时间基线:

- 当前:**阶段 -1**(规划完成,等开工授权)
- M0.5:`0.1-alpha`(地基 + 金标)
- M5.9:**`1.0`** 🎉
- M6.4:`1.1`(QMT 实盘)
# trade-learn 2.0 愿景与路线图

## 一、项目定位

> **trade-learn 是一个 Python 量化研究框架,让传统策略与 ML 策略用同一套 API 写作,集 empyrical、alphalens、pyfolio 功能于一身,提供从数据到实盘的闭环能力,以 Rust 事件型撮合核为底,以本土 + 国际双轨指标体系为面,以 JupyterLab + MLflow 开箱即用为体验,并能无缝承接 backtrader 存量策略。**

名字对标 `scikit-learn`——承诺"trade + learn"双重能力,目标是占据**"量化圈的 scikit-learn"**这一空白生态位(qlib 太重、backtrader 无 ML、backtesting.py 做不了)。

### 八个短语 = 八个模块 = 八个验收维度

| 短语 | 含义 | 对应模块 | 交付物 |
|---|---|---|---|
| Python 量化研究框架 | 定位与语言 | 整体工程化 | pyproject / CI / 文档站 |
| **传统 + ML 策略统一 API** | **"learn" 兑现** | **`ml/` + `MLStrategy`** | **Feature Store + Model Registry** |
| 集 empyrical/alphalens/pyfolio 于一身 | 评估能力 | `metrics/` + `factor/` + `report/` | 融合后的自主内核 |
| 从数据到实盘的闭环 | 端到端能力 | Broker / DataFeed Protocol | 抽象层 + QMT adapter |
| Rust 事件型撮合核为底 | 性能 + 实盘通路 | `backtest/` + Rust 核 | `_core.so` |
| 本土 + 国际双轨指标体系 | 差异化细节 | `indicators/` | `ta.*` / `ta.tdx.*` / `ta.tv.*` |
| JupyterLab + MLflow 开箱即用 | 研究体验 | `lab/` + `backtest/analyzers/` | `tradelearn lab` + `MLflowAnalyzer` |
| 无缝承接 backtrader 存量策略 | 生态接入 | `compat/backtrader/` | 兼容层 |

## 二、典型用户操作流

```
$ pip install trade-learn[all]
$ tradelearn new my_research && cd my_research
$ tradelearn lab                    ← 一条命令起全栈

  ✅ JupyterLab    http://127.0.0.1:8888
  🔗 MLflow        http://your-mlflow-server:5000  (from MLFLOW_TRACKING_URI)
  ✅ MCP Server    (connected to Jupyter AI)

浏览器自动打开 → 左侧 Chat 面板 → 对话
   "帮我拉 GOOG 数据,写个 RSI 策略,挂 MLflowAnalyzer"
         │
         ▼ Jupyter AI 经 MCP 查 trade-learn docstring
         │
   生成正确代码 → 一键 Insert to Notebook
   # class SmaCross(Strategy): ...  纯策略
   # cerebro.addanalyzer(MLflowAnalyzer, experiment=...)  严格 backtrader 风格
         │
   Shift+Enter → Rust 核跑回测 → Analyzer 自动归档 MLflow
         │
   切到 MLflow UI 看历史对比 / 继续在 Chat 里问"最近 5 次哪组参数最好"
```

**核心体验**:一条命令起环境,对话式写代码,单一 notebook 作工作区,MLflow 作实验库,MCP 作 AI 知识通路——**多端(Claude Desktop / Cursor)共享同一个 MCP server,一份文档多端受用**。

## 三、愿景

### 近期(1.0,2026 年内)
成为**"中文量化圈停更库的接棒者"**——接过 empyrical / alphalens / pyfolio 的能力,加上 Rust 内核、双轨指标与统一 ML API,发布一个可信、够用、可维护的量化研究框架。

### 中期(1.x)
成为**"量化圈的 scikit-learn"**——通过 `MLStrategy` + Feature Store + Model Registry,让 ML 策略像写传统策略一样简单;通过 `compat.backtrader` 承接存量策略;通过 QMT 实盘对接,补齐"研究→实盘"最后一公里。

### 长期(2.x+)
演化为**"可商业化的策略平台技术底座"**——Apache-2.0 协议保留商业自由度,内核质量与插件化能力足以支撑未来的 SaaS 或企业服务。

## 四、产品哲学

1. **约定大于配置**:默认值就能跑,不写配置文件
2. **对得上比跑得快更重要**:指标数值对齐券商软件 > 极致性能
3. **闭环 > 大而全**:端到端能用 > 单点世界第一
4. **为实盘预留,但不逼实盘**:架构留钩子,功能不追逐
5. **先简洁后扩展**:核心 API 小而稳,高级能力走插件
6. **尊重存量用户**:backtrader / backtesting.py 用户零摩擦迁入
7. **克制即专业**:定位外的事不做
8. **扩展性分层**:用户层(Strategy / Indicator / Analyzer / MLStrategy)继承或写函数即可扩展;框架层(Broker / DataFeed / Reporter / FeatureStore)通过 Protocol 扩展;核心层(Rust 撮合核 + metrics)刻意不开放,保护一致性基线——**需要时易扩展,但不给用户 shoot-yourself-in-the-foot 的机会**

## 五、能力边界

### trade-learn **是**
- 回测为主的一体化研究框架
- 本地优先的单机工具(1.0)
- 策略作者的工作台
- Python 第一,Rust 只做引擎

### trade-learn **不是**
- 完整的实盘交易系统(风控/监控交给 QMT 或用户)
- 多用户 SaaS 平台(至少 1.0 不是)
- 低延迟 HFT 系统(tick 级不是目标)
- 券商交易终端
- 通用 AutoML 工具

## 六、关键决策(已锁定)

| 决策点 | 选择 | 理由 |
|---|---|---|
| 开源协议 | **Apache-2.0** | 商业友好 + 多库融合 NOTICE 规范 + 专利保护 |
| 回测引擎语言 | **Rust** | 性能 + 未来实盘可靠性 |
| 回测模式 | **事件型** | 贴近实盘,通向 QMT |
| backtesting.py 参考 | Clean-Room + 设计笔记 + 冷冻期 | 规避 AGPL 传染 |
| backtrader 兼容 | 独立适配层 `compat.backtrader` | 不复制 GPL 代码,API 兼容 |
| MLflow 对接 | 仅客户端上传(远程服务器用户自行部署) | 框架只做 Analyzer 侧集成 |
| 指标主基座 | pandas-ta-classic | 通用 + 活跃维护 |
| 内盘指标 | tdx30(通达信口径) | 对齐券商软件 |
| 外盘指标 | pyneCore(TradingView 口径) | 对齐 TV 图表 |
| 数据源 | 剔除 yfinance;保留 TV + opentdx | 聚焦 + 稳定 |
| 本地开发 | 自带 JupyterLab 一键启动 | 零配置体验 |
| 实验追踪 | 自动上传 MLflow | 可追溯 |
| 实盘方案 | 1.1 对接 quant-qmt-proxy | 不拖 1.0 |
| 一致性保障 | 金标集 + CI required check | 重构零信任损失 |

## 七、核心设计原则

1. **契约先行**:7 个核心对象(Bars / Factor / Signal / Returns / StreamBar / Experiment / Broker)定义清楚再写代码
2. **去 vendor**:融合或 pip 依赖,不内嵌第三方库源码
3. **指标唯一真源**:`tradelearn.metrics` 是全项目唯一的指标计算入口
4. **一致性硬约束**:重构前后决策层 0 差异,数值层 < 0.01% 且可解释
5. **跨平台优先**:1.0 严守跨平台,实盘走 extras(Windows-only)
6. **每阶段可发版**:0.x 随时可发,不烂尾
7. **文档即源码**:docstring 是唯一源头,API Reference(mkdocs) + MCP 查询 + LLM 代码生成都从它出发,零重复零不一致
8. **严格 backtrader 调用逻辑**:用户 API **逐字对齐 backtrader**——`Cerebro() / adddata / addstrategy / broker.setcash / broker.setcommission / addanalyzer / run`;Strategy 用 `params = (...)` + `self.p.xxx` + `__init__` + `self.ma1[0]`(索引 0 = 当前);`notify_order / notify_trade` 回调。**Strategy 保持纯净**,追踪/分析/报告通过 `Cerebro.addanalyzer(...)` 挂载,零策略侵入。这是"无缝衔接 backtrader 存量策略"的底线。
9. **双参考原则**:
   - **内核(Rust 撮合核)参考 backtesting.py**——借鉴其成熟的事件循环 / 订单撮合 / 成交价规则,保证逻辑正确性;Clean-Room 独立实现规避 AGPL
   - **用户 API(Python 调用层)参考 backtrader**——Strategy / Cerebro / Analyzer 范式,无缝衔接既存策略生态;API 兼容独立实现规避 GPL
   - 两者都走 Apache-2.0 + NOTICE 诚实署名

## 八、路线图概览

| 阶段 | 版本 | 主题 | 周数 | 累计 |
|---|---|---|---|---|
| 0 | 0.1-alpha | 地基 + 金标基线 | 2 | M0.5 |
| 1 | 0.1 | metrics 融合(empyrical) | 2 | M1 |
| 2 | 0.2 | factor / report 融合(alphalens + pyfolio + quantstats)+ 数据替换 + 设计笔记 | 4 | M2 |
| — | — | 冷冻期(并行做别的)| 2 | M2.5 |
| 3 | 0.3 | Rust 事件撮合核 | 5 | M3.75 |
| 4 | 0.4 | 数据缓存 + Analyzer(含 MLflowAnalyzer)+ CLI | 2 | M4.25 |
| 5 | 0.5 | JupyterLab 一键环境 | 1 | M4.5 |
| 6 | 0.6 | **ML 能力(MLStrategy + Feature Store + Model Registry)** | **1.5** | M4.9 |
| 7 | 0.7 | **MCP Server(API 知识 + 代码脚手架,让 LLM 在 notebook 帮写代码)** | **1** | M5 |
| 8 | 0.8 | compat.backtrader 兼容层 | 1.5 | M5.4 |
| 9 | **1.0** | **文档 + 全量金标 + 发版** | **2** | **M5.9** |
| 10 | 1.1 | QMT 实盘对接(可选) | 2 | M6.4 |

**总周期**:约 6 个月(理想)/ 8–9 个月(现实,1.5× 摊还)

## 九、各阶段要做什么

### 阶段 0:地基 + 金标基线(2 周)
- 写 9 份 Spec(Architecture / Contracts / Metrics / Indicators / Backtest / Strategy / Report / Consistency / Migration)
- 确定 Apache-2.0 协议 + LICENSE + NOTICE
- 迁 `pyproject.toml` + uv + 依赖分组
- CI 矩阵(Linux/macOS/Windows × Py 3.10–3.12)
- 时区 / Seed / 日志 / 错误体系
- **统一进度条(tqdm)**:长回测、数据下载、grid search、Feature 计算全部可视进度,不黑屏等待
- 金标数据集(10 策略 × 10 数据集,用 1.x 固化 expected)
- pyneCore PoC

**交付**:`0.1-alpha`,设计文档冻结,金标基线就绪

### 阶段 1:metrics 融合(2 周)
- 梳理 empyrical + pyfolio + alphalens 35 个指标清单
- 分四个模块:`metrics/returns.py / risk.py / factor.py / trade.py`
- 统一签名:`func(returns, periods, nan_policy, ...)`
- 金标测试:对原库 `rtol=1e-10`
- 覆盖率 100% + CI 卡死

**交付**:`0.1`,metrics 模块上线,指标层一致性保障

### 阶段 2:factor / report 融合 + 依赖替换 + 设计笔记(4 周)

**Week 1 — 数据源**
- 移除 yfinance
- `旧 TDX provider → opentdx` 适配器
- `Bars` 契约实现 + 复权/时区规范
- 数据缓存基础版(parquet)

**Week 2 — 技术指标**
- `pandas_ta → pandas-ta-classic`,封装 `ta.*`
- tdx30 迁到 `ta.tdx.*` + 通达信金标对照
- 删除 vendor `query/tec/pandas_ta`

**Week 3 — 融合**
- `factor/` 融合 alphalens
- `report/html.py` 融合 pyfolio + quantstats
- `report/excel.py` xlsxwriter 多 sheet
- `report/explore.py` pygwalker 集成

**Week 4 — 清理 + 设计笔记**
- 删除所有 vendor 目录
- **backtesting.py 设计笔记**(matching / event-loop / portfolio 三篇)
- 笔记冻结,进入冷冻期

**交付**:`0.2`,vendor 清空,设计笔记就绪

### 冷冻期(2 周,并行)
- Cargo workspace + PyO3 + maturin 脚手架
- 金标集扩充(通达信 + TV 导出对照)
- pyneCore 深度 PoC

### 阶段 3:Rust 事件撮合核(5 周)

**Week 1 — 脚手架 + 契约**
- 事件类型(Bar/Order/Fill/Cancel/Reject)
- **Broker trait + DataFeed trait(为实盘预留)**
- 事件队列(BTreeMap,决策确定性)

**Week 2 — 撮合核**
- Market / Limit / Stop / Stop-Limit 四种订单
- Slippage / Commission 模型
- **只看设计笔记,不打开 backtesting.py 源码**

**Week 3 — Portfolio 记账**
- equity / margin / position / pnl
- 多资产组合支持
- `trade_on_close` 语义
- **多周期(Multi-Timeframe)支持**:backtrader 风格多 data feed
  ```python
  cerebro.adddata(data_1d, name='daily')
  cerebro.adddata(data_5m, name='5min')      # 多 timeframe
  # Strategy 里 self.datas[0] / self.datas[1] 访问
  ```

**Week 4 — PyO3 绑定**
- Arrow 数据接口
- **Batched callback 机制**(性能关键)
- Python `Strategy` / `Cerebro` / `Analyzer` 基类(**严格逐字对齐 backtrader**:`params`/`self.p`/`__init__`/`next`/`notify_order`/`notify_trade`/`self.data.close[0]`)
- `SimBroker`(Broker 协议的回测实现)

**Week 5 — 对照 + 打包**
- 全量金标对照:trades 0 差异,PnL `rtol=1e-4`
- Benchmark:单品种 10 年 < 50ms、500 股组合 < 5s
- 跨平台 wheel(cibuildwheel)
- MIGRATION.md 记录差异

**交付**:`0.3`,Rust 事件撮合核,决策层一致性保障

### 阶段 4:数据缓存 + Analyzer + CLI(2 周)
- 数据缓存完善(TTL / 指纹 / offline mode)
- **Analyzer 扩展点(严格 backtrader 风格)**:
  - `Analyzer` 基类 + `Cerebro.addanalyzer(AnalyzerCls, **kwargs)` API
  - **逐字对齐 backtrader** 的 `cerebro.addanalyzer(...)`
  - Analyzer 在 Cerebro.run() 前后钩入生命周期
- **`MLflowAnalyzer`**:封装所有 MLflow 细节,用户零感知
  ```python
  cerebro = Cerebro()
  cerebro.adddata(data)
  cerebro.addstrategy(SmaCross)
  cerebro.broker.setcash(1_000_000)
  cerebro.broker.setcommission(0.002)
  cerebro.addanalyzer(MLflowAnalyzer, experiment="sma_goog")
  cerebro.run()   # 自动汇总 Strategy params + broker 参数 + stats + reports 上报
  ```
  - 自动记录:Strategy `params`(fast/slow)+ broker 参数(cash/commission)+ stats + trades/equity parquet + report.html/xlsx + strategy.py 源码
  - `MLFLOW_TRACKING_URI` 未设置或连接失败 → warn,不中断回测
  - URI 优先级:Analyzer kwargs > `MLFLOW_TRACKING_URI` 环境变量
- `grid_search` nested runs
- CLI 6 个核心命令:`lab / new / data(fetch/list/clear)/ run / mcp / doctor / --version`
- Config 系统(yaml + env var,可选)

**交付**:`0.4`,每次回测可追溯,CLI 可用

### 阶段 5:JupyterLab 环境(1 周)
- `tradelearn lab` CLI(typer)——**一条命令同时起 JupyterLab + MCP Server**,MLflow 指向用户配置的远程 server(`MLFLOW_TRACKING_URI`)
- 预装扩展(jupyterlab-git / pygwalker / ipywidgets / **jupyter-ai**)
- JupyterLab 默认展示 Chat 面板,Jupyter AI persona 预配"查代码优先走 MCP"
- `tradelearn new <name>` 项目骨架 + `tradelearn doctor` 环境诊断
- 3 个 starter notebook(探索 / 因子 / 回测)
- Windows 跨平台进程管理

**交付**:`0.5`,`pip install trade-learn[lab] && tradelearn lab` 一键启动全栈

### 阶段 6:ML 能力(1.5 周)

让"learn"真正兑现,把 ML 策略做成一等公民:

- `ml/strategy.py`:`MLStrategy` 基类,内置训练→预测→信号→下单流程
  ```python
  class XgbStrategy(MLStrategy):
      model = GradientBoostingRegressor()
      features = [ta.rsi, ta.macd, Alpha101()]
      target = Returns(horizon=5)
      threshold = 0.01
  ```
- `ml/features.py`:**Feature Store**,因子版本化 + 缓存复用
  ```python
  @feature(name="momentum_20d", version=1)
  def momentum(bars): ...
  ```
- `ml/registry.py`:与 MLflow Model Registry 集成,策略可直接 `model="my_model:production"`
- `ml/causal.py`:复活 causal 模块为一等公民 API(`CausalSelector`)
- 1 个完整 ML 策略示例(LightGBM + Alpha101 + 因果特征选择)
- 金标:ML 策略在回测下结果确定性(固定 seed 可复现)

**交付**:`0.6`,ML 策略与传统策略用同一套 API,名字"learn"兑现

### 阶段 7:MCP Server — docstring 暴露给 LLM(4 天)

**核心定位**:MCP 不是"再写一份 LLM 专用文档",而是**把已有的 docstring 以结构化方式暴露给 LLM**。docstring 是唯一源,mkdocs 生成给人看,MCP 查询给 LLM 用,零重复、零不一致。

- **docstring 工程化**(1 天,给 API Reference 和 MCP 同时服务):
  - 所有公开 API 补齐 docstring(Parameters / Returns / Examples / Notes)
  - doctest 可跑
  - CI 强制(`interrogate` / `pydocstyle`)
- `tradelearn/mcp/server.py`:MCP server 薄层(基于官方 Python MCP SDK)
- `tradelearn mcp` CLI 启动命令(stdio 模式,供外部客户端连)
- **Tools(3 个,只读查询)**:
  - `search_api(keyword)` — 关键词搜 API
  - `get_api_docs(module_path)` — 返回该符号的 docstring(直接读源码)
  - `list_mlflow_runs(experiment)` — 只读查 MLflow 历史实验
- **Prompts(6 个场景代码模板)**:
  - `scaffold_data_loading / scaffold_indicator_usage / scaffold_strategy / scaffold_backtest / scaffold_report / scaffold_backtrader_migration`
- Tool description 写"ALWAYS call before writing trade-learn code"引导 LLM 主动查询
- JupyterLab 预装 `jupyter-ai`,配 persona:查文档优先走 MCP

**设计哲学**:
- **文档即源码**——docstring 是唯一源头,API Reference(mkdocs)和 MCP 各取所需
- 无状态 MCP(只读查询,无远程执行)
- 代码/执行/结果**都在 notebook cell 里**,LLM 只帮写代码
- 同一个 MCP server 服务 Claude Desktop / Cursor / Jupyter AI

**明确不做**:重复造 cheatsheet、远程跑回测/算因子、内置 LLM provider、AI 自动决策

**交付**:`0.7`,docstring 高质量 + MCP 薄暴露层,AI 能在 notebook 里写对 trade-learn 代码

### 阶段 8:compat.backtrader 兼容层(1.5 周)
- `compat/backtrader/strategy.py`(Strategy / params / Lines 反转索引)
- `compat/backtrader/cerebro.py`(Cerebro 门面)
- `compat/backtrader/indicators.py`(常用 20 个指标映射)
- `compat/backtrader/feeds.py`(PandasData 映射)
- `notify_order / notify_trade` 映射
- 金标:10 个知名 backtrader 开源策略跑通对照

**交付**:`0.8`,一行 import 迁移存量策略

### 阶段 9:文档 + 发版(2 周)

**Week 1 — 文档站**
- mkdocs-material 搭建
- Quickstart(5 分钟跑通)
- Tutorials:因子研究 / 策略回测 / 组合回测 / **ML 策略** / MLflow / JupyterLab / 从 backtrader 迁移
- MIGRATION.md 完整版

**Week 2 — 发版**
- API Reference(mkdocstrings)
- Architecture 文档
- Benchmark 页(速度 + 指标一致性)
- 对比页:vs qlib / vnpy / backtrader / nautilus
- `demos/` 全量迁移
- 端到端金标测试全通过
- wheel 含 Rust 二进制,PyPI 发布
- NOTICE 最终审查

**交付**:`1.0.0` 🎉

### 阶段 10:QMT 实盘对接(1.1,2 周,可选)
- `brokers/qmt.py` 适配器(quant-qmt-proxy HTTP API)
- 实现 Broker Protocol
- 三种模式:`backtest` / `paper` / `live`
- 实盘二次确认 + 资金上限
- 断线重连 + 基础风控
- 实盘 MLflow run(每日一个)
- Windows-only CI

**交付**:`1.1`,实盘可用(Windows + QMT)

## 十、一致性保障体系

### 三道防线

**防线 1:金标数据集**
- 阶段 0 建立,用 1.x 固化 expected
- 10 策略 × 10 数据集 = 100 组对照
- 版本化冻结(`golden/v1.0/`)

**防线 2:分层对照**
| 层次 | 标准 |
|---|---|
| metrics 指标 | `rtol=1e-10` |
| `ta.*` 指标 | 对 pandas-ta-classic `rtol=1e-10` |
| `ta.tdx.*` / `ta.tv.*` | 对原平台导出 `rtol=1e-6` |
| 决策层(trades 时间/方向) | **0 差异** |
| Equity Curve | `rtol=1e-6` |
| 最终 stats | `rtol=1e-4` |

**防线 3:差异可解释**
- 所有差异写入 `MIGRATION.md`
- 不允许"不知道为什么"
- 边界情况提供 `compat="1.x"` 兼容模式

### CI 守护
- `tests/consistency/` 为 required check
- 金标变更走独立 PR,需 review

## 十一、最终架构

```
tradelearn/
├── core/              # 契约:Bars / Factor / Signal / Returns / StreamBar / Experiment / Broker
├── data/
│   ├── query.py       # tv / opentdx
│   └── cache.py       # parquet 缓存
├── indicators/        # ta.* / ta.tdx.* / ta.tv.*
├── factor/            # 融合 alphalens
├── metrics/           # 融合 empyrical ← 指标唯一真源
├── report/
│   ├── html.py        # 融合 pyfolio + quantstats
│   ├── excel.py       # xlsxwriter
│   └── explore.py     # pygwalker
├── backtest/
│   ├── __init__.py    # Cerebro / Strategy Python API(逐字对齐 backtrader)
│   ├── _core.*.so     # Rust 产物(Clean-Room 参考 backtesting.py)
│   ├── broker.py      # Broker Protocol + SimBroker
│   └── analyzers/     # Analyzer 扩展点(对标 backtrader addanalyzer)
│       └── mlflow.py  # MLflowAnalyzer(MLflow 上传全部细节在此)
├── ml/                # ML 能力("learn" 兑现)
│   ├── strategy.py    # MLStrategy 基类
│   ├── features.py    # Feature Store
│   ├── registry.py    # MLflow Model Registry 集成
│   └── causal.py      # 因果特征选择一等公民 API
├── mcp/               # MCP Server(docstring 暴露给 LLM)
│   ├── server.py      # 基于官方 Python MCP SDK
│   ├── tools/         # 3 个只读查询 tool(search / docs / mlflow)
│   └── prompts/       # 6 个场景代码模板
├── brokers/           # 1.1:QMTBroker
├── compat/
│   └── backtrader/    # 兼容层
├── lab/
│   ├── cli.py         # tradelearn lab / new / data / run / mcp / doctor
│   └── templates/
└── config.py          # yaml + env var

backtest-rs/           # Rust workspace
docs/
├── specs/             # 9 份设计文档
├── internal/          # Clean-Room 设计笔记
└── tutorials/
tests/
├── golden/            # 一致性基线
├── consistency/       # 分层对照
└── unit/
benchmarks/
```

## 十二、依赖分组(pyproject.toml)

```toml
[project]
license = { text = "Apache-2.0" }
dependencies = [
    # 科学计算
    "numpy>=1.26", "pandas", "scipy", "statsmodels", "scikit-learn",
    # 行情数据
    "opentdx", "tvdatafeed", "pyarrow",
    # 技术指标
    "pandas-ta-classic", "pynecore",
    # 报告(脚本 / notebook 通用)
    "xlsxwriter", "bokeh",
    # 因果(项目特色)
    "causallearn",
    # 实验追踪(MLflowAnalyzer 所需 client)
    "mlflow",
    # MCP server
    "mcp",
    # HTTP(实盘通路 + 通用)
    "httpx",
    # 工程化
    "tqdm", "typer", "pydantic", "pyyaml",
]

[project.optional-dependencies]
lab = [
    "jupyterlab", "jupyterlab-git", "ipywidgets", "jupyter-ai",
    "pygwalker",   # notebook 专用交互式探索,report.explore() 用
]
```

**判断标准**:"新用户第一次跑 Quickstart 需要它吗?需要 → 自带"。因此除了 JupyterLab(体积 ~300MB + 部分用户不需要 GUI)单独走 `[lab]`,其它全部自带——`pip install trade-learn` 开箱即用完整回测闭环。MLStrategy 默认用 scikit-learn 里的模型;想用 lightgbm/xgboost/pytorch 自行安装。

## 十三、明确不做(1.0 前)

- ❌ Sizer / Risk / Rebalancer 内置抽象
- ❌ Walk-Forward / Optimizer / Monte Carlo 内置
- ❌ 事件日历(财报日/停牌/涨跌停)
- ❌ 内置 LLM Provider(用户自带 Claude Desktop / Cursor / Jupyter AI)
- ❌ MCP 远程执行(跑回测/算因子——让 LLM 生成代码到 notebook 自己跑)
- ❌ AI 自动决策 / LLM 实时出信号 / 自然语言查数据
- ❌ 实盘监控 / 风控 / 报警
- ❌ Web 服务 / 多用户 / JupyterHub
- ❌ 自造 Pine 解释器
- ❌ GUI / 移动端
- ❌ tick 级 HFT
- ❌ 支持 < Python 3.10
- ❌ 遥测 / 商业钩子(接口保留,功能不实现)

## 十四、风险与对冲

| 风险 | 级别 | 对冲 |
|---|---|---|
| Rust 核结果与 1.x 不一致 | 高 | 金标集 + 每周对照 + MIGRATION |
| Clean-Room 冷冻期不遵守 | 高 | 明确纪律 + 冷冻期安排其他工作 |
| PyO3 callback 性能不达标 | 高 | Batched callback + 早期 benchmark |
| 跨平台 wheel 踩坑 | 高 | 阶段 3 专门留打包时间 |
| AGPL 合规(backtesting.py) | 中 | Apache-2.0 + Clean-Room + NOTICE |
| GPL 合规(backtrader 兼容) | 中 | 独立实现 API 兼容层 + NOTICE |
| pyneCore 成熟度 | 中 | 冷冻期 PoC,不行推 1.1 |
| opentdx 稳定性 | 中 | 以 opentdx 为唯一 TDX provider 口径 |
| 范围蔓延 | 高 | 阶段 7 后锁 scope,新需求进 1.1+ |
| 个人精力衰减 | 高 | 每阶段可独立发版,随时暂停不烂尾 |

## 十五、核心执行纪律

1. **契约不定,代码不写** — 阶段 0 设计文档是前提
2. **金标不过,PR 不合** — CI required check,无例外
3. **冷冻期不碰源码** — Clean-Room 铁律
4. **指标金标 rtol=1e-10** — 阶段 1 就立
5. **决策必须 0 差异** — 阶段 3 的硬约束
6. **差异必须可解释** — MIGRATION.md 记录,不允许黑盒
7. **文档与代码同步** — 每阶段必有 tutorial
8. **每阶段可发版** — 0.x 随时能停,不烂尾
9. **1.x 活着作 oracle** — 直到 1.0 发版
10. **1.0 后锁 scope** — 新需求进 1.1+

## 十六、用户画像

| 用户 | 使用场景 | 核心价值 |
|---|---|---|
| 个人量化研究员 | 业余研究 A 股 / 美股策略 | 开箱即用 + 本土指标对齐 |
| 小型私募研究员 | 团队内部策略研究 | MLflow 管理实验,策略可追溯 |
| backtrader 老用户 | 存量策略升级 | 零成本迁移 + 性能提升 |
| 金融科技课程 | 教学 / 毕设 | 完整闭环,可视化好 |
| ML 策略研究者 | 因子挖掘 + 建模 | 因果特征 + MLflow |
| 未来机构客户 | 策略平台 SaaS | 1.x 扩展 |

## 十七、关键时间节点

| 节点 | 版本 | 里程碑 | 一致性状态 |
|---|---|---|---|
| M0.5 | 0.1-alpha | 地基 + 金标基线 | 基线固化 |
| M1 | 0.1 | metrics 融合 | 指标层一致 |
| M2 | 0.2 | vendor 清空 + 设计笔记 | factor/report 一致 |
| M2.5 | — | 冷冻期结束 | — |
| M3.75 | 0.3 | **Rust 撮合核** | **决策层一致** |
| M4.25 | 0.4 | Analyzer(含 MLflowAnalyzer)+ CLI | — |
| M4.5 | 0.5 | JupyterLab | — |
| M4.9 | 0.6 | **ML 能力(MLStrategy + Feature Store)** | 名字 "learn" 兑现 |
| M5 | 0.7 | **MCP Server(LLM 客户端通路)** | 能力暴露就绪 |
| M5.4 | 0.8 | backtrader 兼容层 | 兼容对照通过 |
| **M5.9** | **1.0** | **正式发版** 🎉 | **端到端一致** |
| M6.4 | 1.1 | QMT 实盘 | — |

---

## 一句话

**trade-learn 2.0 = 一个让传统策略与 ML 策略共用 API、有自主内核、商业友好、对齐双轨指标、无缝承接 backtrader、可走向实盘的 Python 量化研究框架。** 6 个月出 1.0,接棒 Quantopian 停更生态,对标 scikit-learn 的产品哲学,占据"量化圈的 scikit-learn"这一空白生态位,成为中文量化圈值得长期投入的基础设施。
