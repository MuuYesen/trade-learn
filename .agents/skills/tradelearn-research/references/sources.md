# 方法出处

这份 Skill 的流程，综合自以下公开的 Agent Skill 和它们引用的研究文献。各条内容已用中文重新组织、统一口径，没有照抄原文；阈值统一为可调整的默认值，具体做法以本 Skill 为准。

## 参考的 Agent Skill（2026-11 检索）

| Skill | 主要吸收的内容 |
|---|---|
| [Jimmy7892/quant-research-skill](https://github.com/Jimmy7892/quant-research-skill) `quant-research`（MIT） | 先立题（机制、证伪条件、切分）、样本外只用一次、尝试计数；参数曲面、平滑、一个标准误内的稳定区域、取区域整体；有效观测数；滚动验证每折重做选择；拒绝清单；实盘前写好停用条件、实盘与模拟分布对照。**本轮补充**：区域体检（连通、贴边、双高原）、区域内样本内外相关为正先查泄漏、仓位层面等权合并、离散选择也作为网格轴、单格复跑核对网格映射、N 按"取最大"还是"做平均"分别取、PBO 测实际选择流程与噪声基线、滚动验证的预热与区域漂移、价格加噪与交易重排的局限、同根 bar 双触发按不利处理、高周期指标两种算法、暴露时间与最长空仓期、按持仓重叠算 Kish 有效笔数、回撤写明跨度与恢复时间、夏普收缩作停用参照、种子即尝试与集成、测试段被间接用掉的方式、用机制预测跨标的结果、点名"调到数据同意为止"、实盘禁止清单与退役复盘 |
| [tradermonty/claude-trading-skills](https://github.com/tradermonty/claude-trading-skills)（MIT）`backtest-expert` 等 | 找"最不容易坏"的策略；规则写到不需要主观判断；惩罚策略；逐年表现；样本外不到样本内一半的警戒线。**本轮补充**（`backtest-expert`、`edge-strategy-reviewer`、`exposure-coach`、`residual-edge-analyzer`、`manifoldbt-backtester`、`strategy-pivot-designer`、`edge-pipeline-orchestrator`、`trade-performance-coach`、`signal-postmortem`、`weekly-performance-digest`、`trader-memory-core`、`trade-hypothesis-ideator`、`data-quality-checker`、`macro-regime-detector`）：先过硬门槛再看评分、触发项全部列出、约束用上限不用权重、偏差清单必填项；对事先声明基准的残差 alpha；成交到往返交易等指标换算陷阱；参数与条件个数上限；机会数预估；行情标签事先定义；恒定平均仓位对照；复盘前核账的记录不变式；先过程后结果与"正常波动"类原因；信号事后分类的中性区间；R 倍数与 MAE/MFE；迭代停滞与改了再审的轮数上限；证据分级与同源不独立；输入新鲜度；单位与加总自检 |
| [algomaschine/algomaschine-kit](https://github.com/algomaschine/algomaschine-kit)（MIT）`backtest-audit`、`quant-idea-to-system`、`afml-validation`、方法说明页 | 运气、搜索、行情、错误四类解释加脆弱性的审计框架；随机择时与买入持有对照；截断重算、从持仓重建收益；"目前可信"的下一步只能是模拟盘。**本轮补充**：真实数据之前的流程自检（符号翻转零假设、植入效应、时点单元测试）、截断重算比仓位并从原始数据重跑、尝试台账永不清零与方案冻结、想法来源声明、核心项一票否决与审计阈值表、Reality Check 与 Hansen 再中心化、Lo 序列相关修正、预期误选数筛选、CPCV 路径、重叠标签的唯一性权重、按数据标记拆收益、时钟审计 |
| [quantskills/skill-backtesting-bias-avoidance](https://github.com/quantskills/skill-backtesting-bias-avoidance)（GPL-3.0） | 头条数字必须是样本外、扣成本、无前视；考虑序列相关的标准误；写"没有显著的净收益"而不写"策略无效"；盈亏平衡成本。**本轮补充**：HAC 与按持仓段两种口径取保守、已知真值的多路径合成数据校准流程、PBO 中间档、样本内外换手比、分级告警表 |
| [ml4t/skills](https://github.com/ml4t/skills)（Apache-2.0） | 前视的具体规则与"好得不像真的"信号；成本分层与 2.5 倍安全余量；IC、Rank IC、IC_IR 与重叠期修正。**本轮补充**：顺序硬门槛（CPCV 中位数、亏损路径占比、选择上界、留出段衰减）、因子门槛、稳健比例与断崖、按 IC 衰减选预测期、扩展窗与滚动窗对照与三层评估、特征筛查与折内选择、泄漏探测阈值、CPCV 执行与报告、分行情评估、三重障碍标签、PSI/KS 漂移、行情数据检测阈值、时点可交易池、容量公式、情景压力测试、HHI 与风格漂移、尾部与滚动夏普、上线阶梯与同一份代码、实验登记、中性化前的因果识别、平稳性双检验、PCA 噪声上界、主动管理基本定律 |
| [agiprolabs/claude-trading-skills](https://github.com/agiprolabs/claude-trading-skills)（MIT） | 滚动与扩展窗口、剔除与间隔、CPCV、DSR 与 PBO 的阈值。**本轮补充**（`walk-forward-validation`、`ohlcv-processing`、`correlation-analysis`、`market-microstructure-traditional`、`trade-accounting`、`risk-management`、`trade-journal`、`feature-engineering`、`signal-classification`）：滚动验证的结构下限与行情覆盖、多周期对齐与残缺 bar、截面有效赌注数与相关系数区间、执行缺口的机会成本与撤单样本、账户闭合核对、运行异常与逐笔规则执行核对、ML 泄漏与评价清单、Wilson 区间 |
| [wshobson/agents](https://github.com/wshobson/agents) `quantitative-trading`（MIT） | 偏差分类（前视、幸存者、过拟合、选择、成本）、事先登记、容量。**本轮补充**：训练、验证、测试三段切分（可选）、CVaR 与压力期条件相关、按持有期的亏损概率 |
| [quantskills/skill-backtest](https://github.com/quantskills/skill-backtest)（GPL-3.0） | A 股截面回测约定：T+1 开盘成交、涨跌停与停牌剔除、滚动持仓。**本轮补充**：截面回测的结构量级体检、等权可交易池与随机篮子基准 |

## QuantSkills 社区（[github.com/quantskills](https://github.com/quantskills)，2026-11 检索，GPL-3.0，`skill-ml-purged-cv` 为 MIT）

组织下 260 余个仓库，大部分是具体因子、数据接口和盯盘工具；下面这些方法论类的被吸收：

| Skill | 主要吸收的内容 |
|---|---|
| `skill-backtest-overfit` | 紧缩夏普、PBO、多重检验折扣、最短样本长度（MinTRL）；诚实申报尝试次数；用每期夏普 |
| `skill-ml-purged-cv`（MIT） | 本轮：测试段消耗登记（失败也算、改名不复用）、前向证据（预测先于标签登记）、验证链路 / 研究门槛 / 上线门槛分开判、按信息区间剔除与同日不拆分、不安全对照数重叠、每折训练量门槛与无效折不跳过、CPCV 路径分布、选择遗憾、选择流程与成本的对抗性测试、成本情景不计入尝试数、跨行情排名稳定、零预测基线、共同样本比较、逐特征可用时间与批量入库不是时点证据 |
| `skill-a-share-tradability-auditor` | A 股成交可行性四层约束、不复权判断涨跌停、"市场不会给你的收益"占比、先扣假收益再做统计检验。本轮：封死与开板的区分与三档敏感性、取最小值不相乘、判读阈值与处理顺序、重放规则与修正后复审、涨跌幅按日期查表 |
| `skill-factor-mason` | 因子先当嫌疑人、时点/样本/成本/样本外四道检查、中性化要有动机、失效归因、下一步实验最多三个 |
| `skill-factor-orthogonalize` | 逐日截面回归、不能向收益正交、正交前后并列。本轮：正交验收门槛与"变好也是警报"、小样本与控制变量护栏 |
| `skill-factor-ic-decay`、`skill-signal-stability-audit`、`skill-factor-decay` | IC 半衰期、样本护栏、秩自相关与头部重合度换算换手。本轮：滚动 IC 与强度描述口径、衰减模型选择与半衰期自助区间、衰减联动红旗、稳定性门槛与按实际调仓间隔的成本拖累 |
| `skill-walk-forward-validator` | 按标签跨度剔除并加间隔、分折汇总。本轮：因子滚动验证的硬门槛与软门槛分级 |
| `skill-factor-blend`、`skill-residual-guided-factor-selection` | 本轮：多因子合成（先过质量线、去冗余、与最佳单因子比）、选因子子集的一个标准误规则与多起点稳定性、样本外冻结与拒绝覆盖 |
| `skill-factor-drift-monitor`、`skill-market-regime-analysis` | 本轮：分级漂移检查与小样本降级、行情标签的防前视规则与条件 IC |
| `skill-a-share-pit-fundamental-vintage-builder` | 财报按首次披露版本、披露日顺延一个交易日、不用最新版本。本轮：同日多版本直接失败、代码扫描 |
| `skill-survivorship-universe-auditor` | 历史股票池重建、退市收益不能用 0 代替、稳定证券标识。本轮：退市偏差的成对量化 |
| `skill-intraday-data-quality-auditor`、`skill-futures-roll-auditor` | 行情不变量与缺口待核查；换月规则、信号用连续价、盈亏用真实合约。本轮：按时段分组查缺口、换月台账与通知日 |
| `skill-quant-research` | 用一句话写事件时序、安慰剂对照、无成本结果标为毛收益、复现信息。本轮：研究四阶段、哨兵行检查、按依赖结构选重抽样、压力测试补充项、报告闭合 |
| `skill-quant-research-experiment-registry` | 记录数据快照、代码与依赖版本；冲突数值不默默取舍。本轮：复现比对规则与复现状态、替代数据不算复现 |
| `skill-forecast-calibration-audit` | 本轮：概率预测的校准审计 |
| `skill-transaction-cost-analysis`、`skill-transaction-cost-calibration`、`skill-strategy-capacity-tca` | 执行缺口五项分解、用成交校准成本且不外推、容量曲线与参与率。本轮：符号约定与代理价偏差、按参与率报中位数与 95 分位及参考价覆盖、两种冲击模型对照与事实/估计分开 |
| `skill-factor-backtest` | 本轮：缺输入时停下、样例数据不是证据、永久停牌持仓清出 |
| `skill-quant-strategy-diagnostics`、`skill-strategy-lifecycle` | 近期与等长历史窗口对比、事实与假设分开；停用规则只用当时数据并报告机会成本。本轮：退化信号阈值、停用状态机回放与消融、基线复现、配对周期统计与 MDE |
| `skill-trade-review`、`skill-portfolio-pnl-attribution`、`skill-portfolio-attribution`、`skill-performance-attribution`、`skill-strategy-tearsheet-report`、`skill-risk-return-metrics` | 本轮：收益归因闭合（Brinson、Carino、因子面板对同一总账）、市场暴露与择时项、分组恒等式、解读带证据键 |
| 本轮从导航页补读：`agent-quant-failure-triage`、`skill-factor-drift-closure`、`skill-strategy-performance-report`、`skill-jev-history-replay`、`skill-backtest-assumption-check`、`skill-numerical-leak-check`、`skill-factor-debug`、`agent-alpha-portfolio-guardian` | 失效分层分诊与首个成立即停、逐层反事实与理想成交上界、诊断附证伪条件与复核触发；预警动作的对照组、告警事件账本、缺失标签双情景、原版与候选各自从现金推进、数据段"已查看"标记与回顾性修订；净值来源优先级与对账残差；口径参数哈希与可比性门槛、交易指纹证明改动生效、五轴差异账本与第一处分歧、检查工具的阳性与阴性对照、单因素消融的适用边界、配对符号检验；缺陷严重度与可信度门槛；未来扰动与敏感点切点设计；测试段产物的代码级禁读；守卫规则本身也要回测 |
| 各审计类 Skill 的共同约定 | 已证实问题与缺失证据分开报告，缺失证据不写成通过 |

**读过但没有收的**：具体因子与选股公式、数据商接口用法、下单与执行工具、仓位与杠杆方法（Kelly、波动率目标、分级降仓、账户熔断）、均值回复与协整的专项检验。部分来源的公式或示例有误、与本 Skill 冲突的也没有收（例如用年化夏普代入的 DSR 代码、把划分数当成 CPCV 路径数、滚动验证每折挑最优点、按所属期对齐宏观数据的示例、k 折概率校准、缺失日期补 0 收益、加权总分评分卡、不考虑标准误的固定实盘停用线）。

## 研究文献

- Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*.（概率夏普、MinTRL）
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). Pseudo-Mathematics and Financial Charlatanism. *Notices of the AMS*.（最短回测年数）
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.（剔除与间隔、CPCV、样本唯一性、元标签）
- Lo, A. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*.
- Harvey, C. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*.
- Hansen, P. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*.
- Politis, D. & Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*.
- Newey, W. & West, K. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*.
- Lo, A. & MacKinlay, C. (1988). Stock Market Prices Do Not Follow Random Walks. *Review of Financial Studies*.（方差比）
- McLean, D. & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*.（已发表效应的衰减）
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*.（IC 与主动管理基本定律）
- Brinson, G. & Fachler, N. (1985). Measuring Non-US Equity Portfolio Performance. *Journal of Portfolio Management*.
- Carino, D. (1999). Combining Attribution Effects over Time. *Journal of Performance Measurement*.
- Treynor, J. & Mazuy, K. (1966). Can Mutual Funds Outguess the Market? *Harvard Business Review*.
- Perold, A. (1988). The Implementation Shortfall: Paper versus Reality. *Journal of Portfolio Management*.
- Almgren, R. et al. (2005). Direct Estimation of Equity Market Impact. *Risk*.（平方根冲击）
