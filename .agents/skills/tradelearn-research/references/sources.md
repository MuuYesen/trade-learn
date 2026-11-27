# 方法出处

这份 Skill 的流程，综合自以下公开的 Agent Skill 和它们引用的研究文献。各条内容已用中文重新组织、统一口径；具体做法以本 Skill 为准。

## 参考的 Agent Skill（2026-11 检索）

| Skill | 主要吸收的内容 |
|---|---|
| [Jimmy7892/quant-research-skill](https://github.com/Jimmy7892/quant-research-skill) `quant-research` | 先立题（机制、证伪条件、切分）、样本外只用一次、尝试计数；参数曲面、平滑、一个标准误内的稳定区域、取区域整体；有效观测数；滚动验证每折重做选择；拒绝清单；实盘前写好停用条件、实盘与模拟分布对照 |
| [tradermonty/claude-trading-skills](https://github.com/tradermonty/claude-trading-skills) `backtest-expert` | 找"最不容易坏"的策略；规则写到不需要主观判断；惩罚策略（滑点 1.5–2 倍、最差成交）；逐年表现；样本外不到样本内一半的警戒线 |
| [algomaschine/algomaschine-kit](https://github.com/algomaschine/algomaschine-kit) `backtest-audit`、`quant-idea-to-system` | 运气、搜索、行情、错误四类解释加脆弱性的审计框架；随机择时与买入持有对照；截断重算、从持仓重建收益；"目前可信"的下一步只能是模拟盘 |
| [quantskills/skill-backtesting-bias-avoidance](https://github.com/quantskills/skill-backtesting-bias-avoidance) | 头条数字必须是样本外、扣成本、无前视；考虑序列相关的标准误；写"没有显著的净收益"而不写"策略无效"；盈亏平衡成本 |
| [ml4t/skills](https://github.com/ml4t/skills) `backtest-overfitting`、`lookahead-bias`、`transaction-costs`、`information-coefficient` | 前视的具体规则与"好得不像真的"信号；成本分层与 2.5 倍安全余量、1/2/3 倍成本；IC、Rank IC、IC_IR 与重叠期修正 |
| [agiprolabs/claude-trading-skills](https://github.com/agiprolabs/claude-trading-skills) `walk-forward-validation` | 滚动与扩展窗口、剔除与间隔的取值、CPCV、DSR 与 PBO 的阈值 |
| [wshobson/agents](https://github.com/wshobson/agents) `backtesting-frameworks` | 偏差分类（前视、幸存者、过拟合、选择、成本）、事先登记、容量 |
| [quantskills/skill-backtest](https://github.com/quantskills/skill-backtest) | A 股截面回测约定：T+1 开盘成交、涨跌停与停牌剔除、滚动持仓 |

## QuantSkills 社区（[github.com/quantskills](https://github.com/quantskills)，2026-11 检索）

组织下 260 余个仓库，大部分是具体因子、数据接口和盯盘工具；下面这些方法论类的被吸收：

| Skill | 主要吸收的内容 |
|---|---|
| `skill-backtest-overfit` | 紧缩夏普、PBO、多重检验折扣、最短样本长度（MinTRL）；诚实申报尝试次数；用每期夏普 |
| `skill-a-share-tradability-auditor` | A 股成交可行性四层约束、不复权判断涨跌停、"市场不会给你的收益"占比、先扣假收益再做统计检验 |
| `skill-factor-mason` | 因子先当嫌疑人、时点/样本/成本/样本外四道检查、中性化要有动机、失效归因、下一步实验最多三个 |
| `skill-factor-orthogonalize` | 逐日截面回归、不能向收益正交、正交前后并列 |
| `skill-factor-ic-decay`、`skill-signal-stability-audit` | IC 半衰期、样本护栏、秩自相关与头部重合度换算换手 |
| `skill-walk-forward-validator` | 按标签跨度剔除并加间隔、分折汇总 |
| `skill-a-share-pit-fundamental-vintage-builder` | 财报按首次披露版本、披露日顺延一个交易日、不用最新版本 |
| `skill-survivorship-universe-auditor` | 历史股票池重建、退市收益不能用 0 代替、稳定证券标识 |
| `skill-intraday-data-quality-auditor`、`skill-futures-roll-auditor` | 行情不变量与缺口待核查；换月规则、信号用连续价、盈亏用真实合约 |
| `skill-quant-research` | 用一句话写事件时序、安慰剂对照、无成本结果标为毛收益、复现信息 |
| `skill-quant-research-experiment-registry` | 记录数据快照、代码与依赖版本；冲突数值不默默取舍 |
| `skill-transaction-cost-analysis`、`skill-transaction-cost-calibration`、`skill-strategy-capacity-tca` | 执行缺口五项分解、用成交校准成本且不外推、容量曲线与参与率 |
| `skill-quant-strategy-diagnostics`、`skill-strategy-lifecycle` | 近期与等长历史窗口对比、事实与假设分开；停用规则只用当时数据并报告机会成本 |
| 各审计类 Skill 的共同约定 | 已证实问题与缺失证据分开报告，缺失证据不写成通过 |

## 研究文献

- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.（剔除与间隔、CPCV）
- Lo, A. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*.
- Harvey, C. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*.
- Newey, W. & West, K. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*.（IC 与主动管理基本定律）
- Perold, A. (1988). The Implementation Shortfall: Paper versus Reality. *Journal of Portfolio Management*.
- Almgren, R. et al. (2005). Direct Estimation of Equity Market Impact. *Risk*.（平方根冲击）
