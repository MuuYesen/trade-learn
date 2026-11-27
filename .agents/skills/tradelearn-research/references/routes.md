# 七条研究路径：入口、证据、关卡

用户的一句话先归到下面一条路径（可以同时属于几条，按顺序各自完成）。每条写明：**运行入口**（实际调用的 TradeLearn 接口）、**最少证据**（不齐就不能给结论）、**关卡**（禁止的错误结论）、**缺证状态**（证据缺了结论最高到哪一级）。结论分级、方法状态（通过 / 不过 / 不适用 / 受阻 / 未检验）和阈值只在 `gates.md` 维护。

统计计算只用已安装 TradeLearn 的实现，一处算法；本 Skill 的脚本不另写一份。可运行的接线示例在 [route_examples.py](../scripts/route_examples.py)：比较、成本情景、搜索、IC 标准误、读回和同配置续跑，各函数先做能力核验，缺接口时返回"受阻"，`self_test(目录)` 在合成数据上走一遍写入与读回（只验证接线，不是研究证据）。

## 零、每次都做：版本、能力与口径

1. **版本**：记录正在执行的 `tradelearn.__version__` 与 `tradelearn.__file__`，版本字符串相同的源码构建还要记录 Git 提交或安装包指纹。若 `importlib.metadata.version("trade-learn")` 与实际导入来源不同，先确定本次要用的环境。Skill 优先从该发行标签或源码提交获取。
2. **能力核验**：把 [backtest_checks.py](../scripts/backtest_checks.py) 加到导入路径：
   - `capabilities()` 只看名字在不在，不调用；
   - `native_known_answers()` 用冻结输入核对已安装的 `sharpe_se`、`paired_sharpe_diff`、`probabilistic_sharpe`、`deflated_sharpe`、`minimum_track_record_length`，结果为 通过 / 不过 / 受阻。

   "不过"说明已安装实现与约定不符（例如配对差方向反了），停下报告，不用它出数。下面标"新增"的接口（`metrics` 的推断函数与 `hac_mean_se`、`research.validation`、`research.parameter_surface`）不在 0.3.6 发行版中，只在更新的源码里；以能力核验为准。缺失时不要换一个"看起来差不多"的接口，也不要自己拼地址请求服务。
3. **年化期数**：`periods_per_year(市场, bar)` 按市场时钟取（A 股日线 252、加密货币日线 365、加密 30 分钟 17520）。框架结果里的年化指标先看 `metric_metadata[...]["periods"]` 与 `period_source`，用 `metric_periods_ok()` 核对；不一致时用 `tradelearn.metrics.sharpe(returns, periods=...)` 在收益序列上重算，并在口径表里写明。新版日内样本未覆盖一周时，`insufficient_intraday_coverage` 仅是兼容回退。须按市场时钟显式给 `periods` 后重新计算。0.3.6 的 `stats.summary["sharpe"]` 按每根 bar 当一天、取 252（`period_source = default_per_bar_profile`），日内数据会严重失真。
4. **市场数据保全**：原始数据只读，不就地改写、不前值填平、不替换成样例数据；记内容 sha256、区间、根数（`trial_records` 的 `config.data` 已带这些）。

**降级规则**（旧版本或接口缺失时）：
- IID 标准误：`annual_sharpe_se` 是已知答案公式，旧版本上可用，报告写"IID，参考公式"。
- HAC、配对差、PSR/DSR/MinTRL、稳定区域、滚动验证、PBO/CPCV：已安装版本没有时，对应方法状态记"**受阻**"，结论按 `gates.md` 定上限。用户已有经过核验的产物（例如更新版本写出的 `validation.json`）时，可以读回引用，写明来源和版本。
- 不在本 Skill 里临时手写这些方法来替代。

**本地存证（默认）**：每次运行把已有产物写进一个新的本地研究目录，不覆盖旧目录：
- `tradelearn.report.artifacts.write_artifact_bundle(stats, 目录, timeframe=...)` 写出 `stats.json`、`metadata.csv`、`equity.csv`、`fills.csv`、`trades.csv`、`positions.csv`、`report.html` 等；
- 滚动验证、情景、比较的结果对象用自己的 `.write(目录)`，或用 `research.validation.write_validation(目录, ...)` 合并写进 `validation.json`（新增）；
- `trial_records` 和 `ResearchRun(...).finish()` 的 `to_dict()` 写成 JSON；
- 一份清单：数据 sha256、TradeLearn 版本、年化期数及来源、标准误口径、尝试数 N 的取法、成本情景、运行命令。

`save_local_evidence()` 一次写完后两样和 bundle。用户读数优先从 `Reporter(stats).metric_facts()` 或 `stats.json` 取未舍入的数值、单位、缺失原因和年化口径。已有的因子、归因等结构化领域结果交给 `Reporter.from_analysis(result)`，一次计算可再次导出。**MLflow 只在用户指定了跟踪目标时用**，调用时显式传 `uri`（例如用户给的本地 `sqlite:///research.db`）；省略 `uri` 会沿用环境变量里可能指向远端的地址。

## 一、读取/复现

**触发**：「上次那个结果再看看」「这个结论能复现吗」「对一下别人的回测」「沿用上次设置，只把手续费翻倍」。

- **运行入口**：`tradelearn.research.validation.load_bundle(目录, expect_version=...)` 读回 bundle 与 `validation.json`（新增；拒绝未知 schema 与混合版本），`verify_inputs(bundle 或 trial_record, data=..., config=...)` 列出与当前数据、配置不一致的字段（`route_examples.read_back`）。缺失时手工读 `stats.json`、`trial_records.json`、清单并比对 sha256。模型用 `tradelearn.ml.load_bundle(路径, trusted=True)`（核对依赖版本与代码 SHA256）。用户指定了 MLflow 目标时，按用户给的 `uri` 读取对应 Run。
- **续跑**：先读清单与原指标，确认要改的设置。对已记录的单次回测，用 `route_examples.rerun_backtest` 传入经核验的本地 Python 回测函数与原输入，只替换一个已记录设置，写入新的目录。用结构化产物做前后对照，交回产物路径、变更和可重跑命令。缺原策略或设置时标受阻。复用的样本按已查看处理。用法见 [实验记录](experiment-log.md#七从上次结果继续)。
- **最少证据**：数据 sha256 一致、版本一致或差异已列出、参数与成本配置一致；每个比对指标事先写好比对规则（`experiment-log.md` 第五节）。
- **关卡**：`verify_inputs` 返回非空时不得称"已复现"；不能换一份替代数据再称复现成功；结构化结果与文字报告冲突时以结构化结果为准，冲突标"未解决"。
- **缺证状态**：找不到原产物 → "受阻"，只能重跑并标"重新测量"，不是复现；部分指标缺比对规则 → "部分"。

## 二、测量/诊断

**触发**：「跑一次回测」「这个数字对吗」「为什么没成交」。

- **运行入口**：`tradelearn.lite.Backtest(data, Strategy, cash=..., commission=...)` 的 `run()`；需要 Engine 的订单规则时用 `tradelearn.engine.Cerebro(periods=...)`，并通过 `cerebro.broker.setcommission(commission=...)` 设置百分比费用。结果取 `stats.fills`、`stats.orders`、`stats.positions`、`stats.equity`、`stats.returns`、`stats.metric_metadata`。
- **本地数据**：用 `tradelearn.data.open_dataset(用户给的文件)` 读取 CSV/Parquet，调用 `read(symbols=..., start=..., end=...)` 获取显式时区的 UTC 行情，使用同一查询的 `profile(...)` 检查行数、时间边界和质量。单标的策略需从面板显式选择标的。已有缓存用 `BarsCache.read` 读已知快照，记录文件、查询、时区、行情指纹和实际输入指纹。字段映射或时区需要转换时按 [行情审计](data-audit.md) 处理，再进入回测。
- **信号与订单**：先从所装版本的 `tradelearn.indicators` 及其 `capabilities.json` 查 `talib`、`tdx`、`tv` 的可用实现。Lite 用 `Strategy.window`，Engine 可用 `tradelearn.backtest.history.window` 取历史窗口。按指标需要设置 `addminperiod`，记录预热长度、第一笔有效信号与下单时点。成交异常时沿信号 → `orders`（状态、时间、数量）→ `fills` → `positions` 核对。
- **最少证据**：
  - 成本已显式设置（默认手续费是 0）；
  - 信号次根成交；
  - [成交自检](self-check.md#八回测出数前的两项自检)：每个候选、每个评估区间的期初持仓、预期调仓次数、实际成交数；高单价标的确认小数数量或合适的交易单位确实能成交；
  - 年化期数与市场时钟一致（第零节第 3 条）；
  - 夏普附标准误并写明口径；期末持仓的未实现盈亏单独列出（0.3.6 的 `final_unrealized_pnl` 期末估值有已知问题，用权益增量核对）。
- **关卡**：零成交未查明不得进入评价；不得引用 periods 与市场时钟不一致的框架夏普；单次测量不得写"有效""可以交易"。
- **缺证状态**：只做了测量、没有样本外和比较 → 结论最高"证据不足"，并按 SKILL.md 列出完整流程缺什么。

## 三、比较与稳健性

**触发**：「A 和 B 哪个好」「优势靠得住吗」「成本翻倍还行吗」「换个标的呢」。

- **运行入口**：
  - **样本外分段**：`research.validation.walk_forward_folds(data, train=..., test=..., holdout_start=...)` 切时间段，`walk_forward(data, folds, evaluate=..., params=固定参数, warmup=...)` 对每个固定候选只跑测试窗并拼接（新增）。它每个测试窗独立运行一次，持仓不跨窗，预热数据只用于指标成形；研究设计要求连续持仓时，改为一次连续回测、按时间切片取测试段并保留期初持仓（`self-check.md` 第八节）。两种做法在报告里写明用了哪种。缺失时用 `research.time_split` / `split_bars` 手工分段。
  - **配对差**：`research.validation.compare_candidates(收益字典, baseline=..., periods=..., block_length=...)`（新增）按共同时间戳对齐后调用 `metrics.paired_sharpe_diff(left=基准, right=候选, ...)`，`estimate` 为**候选 − 基准**，并报告对齐丢弃的行。直接调用时同样 left=基准、right=候选；返回对象字段是 `estimate`、`standard_error`、`lower`、`upper`、`observations`、`valid_draws`。
  - **各自的绝对证据**：`metrics.sharpe` 与 `metrics.sharpe_se(returns, periods, method=...)`。
  - **成本情景**：`research.validation.run_scenarios(evaluate, {"base": ..., "2x": ...}, periods=...)`（新增），失败情景保留为行。
  - 整套见 `route_examples.compare_fixed`、`cost_scenarios`。
- **最少证据**：
  - 事先写定的时间切分与样本外段，样本外段的数据使用状态（未触碰 / 已查看，`experiment-log.md` 第二节）；用已查看过的历史做比较时，结论只能写"在已查看样本上"；
  - 预热长度、跨段持仓的处理方式；
  - 每个候选自己的净夏普区间（绝对证据）与候选减基准的配对差区间（相对证据），块长与标准误口径在立题时写定；
  - 成本 ×2 情景：用**已冻结的**成本假设乘倍数，情景不是搜索，不计入尝试数，也不能拿来挑方案；
  - 用户问"这组参数的优势稳不稳"时，在立题时写定的参数邻域里做扰动核对（例如每个参数 ±1 步），逐格报配对差。
- **关卡**：
  - 绝对收益与配对相对改善**分开判**：两组都亏损时"少亏得显著"不能写成"有正收益"；某组正收益显著也不能推出它比另一组好（`gates.md` 第五节）；
  - 不用两个候选各自的标准误判并列；
  - 事先固定的候选比较不需要全网格和 DSR，这两项记"不适用"（N = 候选数，报 PSR）；但用户问的是参数优势是否稳健时，邻域扰动是必需证据，不能一概标"不适用"，没做就是"未检验"；
  - 邻域稳定只说明参数稳健，不能替代样本外和成本情景。
- **缺证状态**：只有样本内或只有点估计 → "证据不足"；配对区间跨 0 → 写"差异不显著"，不排先后；问参数稳健却没做邻域扰动 → 最高"证据不足"。

## 四、搜索

**触发**：「帮我调参」「最优参数是多少」「试试这些组合」。

- **运行入口**：
  - 网格试验：`tradelearn.integrations.search.grid_search(Strategy, bars, 网格, cerebro_factory=...)`（单标的 DataFrame）、`Cerebro.optstrategy`、`tradelearn.optimize.OptunaSearch`；所有入口的 `trial_records` 记全部候选，含失败和取消的，是尝试台账 N 的来源。
  - **不要用 Lite 的 `Backtest.optimize()` 选参数**：它按 `return_pct` 只返回单个最优点，记录里的年化指标也受第零节第 3 条影响；需要时只把它的 `trial_records` 当台账读。
  - 滚动验证：`research.validation.walk_forward(data, folds, evaluate=..., candidates=网格, objective=..., select=...)`（新增），每折只在训练段评分和选择，候选失败则该折失败并保留。
  - 稳定区域：在 `select` 里调用 `research.parameter_surface.stable_region(训练评分, 有序轴, standard_error=...)`（新增），返回 `smoothed`、`best_cell`、`components`、`boundary_cells`、`coverage`、`unavailable_cells`。
  - 评分指标和标准误口径（`metrics.sharpe_se` 的 `method`）在立题时写定；缺失或非有限的格计入 `unavailable_cells`，不能默默当通过。
  - 它只诊断区域，不选分量、不合并仓位：要按区域整体持仓，需要在 `evaluate` 里显式实现并说明。
  - 示例 `route_examples.search_with_stable_region` 只把平滑最优格带到测试段，并把区域体检写进每折证据。
  - PBO 用 `research.validation.pbo(全部候选收益表, select=...)` 检查实际采用的选择规则。默认选择样本内最大夏普。自定义规则只能选收益表中真实运行过的候选。动态区域在样本外的真实合并收益缺失时，该方法记"受阻"。CPCV 用 `cpcv_splits` 切分并在 `cpcv_paths` 的路径上运行和评分。
- **最少证据**：尝试数 N（从 `trial_records` 数 completed + failed）、每折的区域体检（覆盖率、分量数、贴边格、缺失格）、拼接后的样本外结果、N ≥ 2 时的 DSR（`metrics.deflated_sharpe`，所需的试验间方差从全部候选收益估）。
- **关卡**：不报"最优参数是 X"；N = 1（事先指定、没有搜索）时报 PSR，不报 DSR；邻近参数稳定只说明**参数稳健**，不能代替成本、行情、样本外检验；有缺失格时区域结论标明覆盖率。
- **缺证状态**：没有滚动验证 → 最高"证据不足"；`stable_region`、`walk_forward`、`pbo` 能力为"缺失"时对应项"受阻"。

## 五、因子与模型

**触发**：「这个因子有没有用」「模型能预测吗」「选哪些特征」。

- **运行入口**：
  - 截面因子：`tradelearn.metrics.factor` 的 `ic`、`rank_ic`、`ic_ir`、`quantile_returns`、`turnover`、`autocorrelation`，或 `tradelearn.factor.FactorAnalyzer`。
  - IC 均值的标准误：`tradelearn.metrics.hac_mean_se(逐期 IC, bandwidth=...)`（新增，Bartlett 核，输出与 IC 同单位），传按规则时钟对齐的逐期 IC。它**只用于均值**；带解释变量的回归系数（例如对基准回归的 alpha）需要回归 HAC，不能用它验证。缺失时 IC 显著性一项"受阻"。示例见 `route_examples.ic_mean_with_hac`。
  - 时序标签：`tradelearn.research.temporal.forward_return(bars, horizon=...)`（输出 `forward_return` 与 `label_end`），再用 `purged_splits` 切分。它只切分，不训练、不评分。`temporal.assessment` 给分块 IC；`walk_forward` 的 `labels=` 按标签区间剔除跨界训练样本（新增）。
  - 预处理与中性化：用 `tradelearn.research.preprocess` 中只在训练段拟合的转换器（如 `Neutralizer`）。
  - 模型：按 [ML 样本外流程](ml.md) 将 `FeatureSet.transform` 的特征与 `forward_return` 的标签分别准备，按 `purged_splits` 切分。每折新建 Pipeline，仅在训练片段 `fit`，再对训练与测试片段分别 `transform`。在训练特征和标签上拟合模型，用 `ModelScorer(current=False, ...).predict` 对完整测试面板评分。用 `ResearchRun.register` / `call` / `finish` 记训练与验证步骤，`tradelearn.ml.save_bundle` 冻结，`predict_bundle` 推理。保存折号与预测时点，评分、组合目标与实际净收益各自留证。
- **最少证据**：因子时点（发布时间）核对、IC 均值与 HAC 标准误、分组单调性、扣成本多空、样本外 IC；模型另需"预测为 0"基线和训练段内拟合的预处理。
- **关卡**：未过四道检查不称 alpha（`factor.md`）；全样本拟合的标准化或特征选择 → 验证链路"无法判断"。
- **缺证状态**：没有发布时间元数据 → "无法判断"时点项；只做了样本内 IC → "证据不足"。

## 六、组合及执行

**触发**：「多标的怎么配」「成本和滑点怎么设」「容量够吗」「A 股能成交吗」。

- **运行入口**：`tradelearn.research.portfolio` 的 `TopK`、`EqualWeight`、`Constraints`、`Allocator`、`RiskfolioOptimizer`；Engine/Lite 百分比费用用第二节已核验的入口。成本情景同第三节 `run_scenarios`。Engine 固定和百分比滑点用 `cerebro.broker.configure_matching(slippage=...)`，模型为 `tradelearn.core.costs.FixedSlippage` 或 `PercentSlippage`，运行前设置。新版需匹配的原生扩展，自定义模型未支持；先用小样本验证成交价按预期变化，再做完整情景。无法确认时把滑点情景标「受阻」。A 股 T+1、涨跌停、停牌的成交约束在框架中未找到现成实现，按 `china-market.md` 自行检查并写明。
- **组合归因**：`Allocator.build(scores)` 产出目标权重，实际敞口从成交与持仓重建。提供逐期组合收益、对应期初权重、同频因子回报及有时点覆盖的因子载荷后，用 `tradelearn.factor.PerformanceAttribution.attribution()`，或其 `to_analysis()` 接 `Reporter.from_analysis` 出表。先检查日期、因子列与持有标的覆盖率，缺失输入标注为证据缺口。该接口会将缺失的因子回报或载荷补零，残差由总收益减共同收益定义。
- **最少证据**：权重合计、约束生效的核对；成本依据；A 股成交可行性四层检查；换手与容量估算。
- **关卡**：不给真实资金的仓位或杠杆建议；越过涨跌停成交、违反 T+1 未修正前不谈收益。
- **缺证状态**：没有容量或冲击模型 → 容量一项"受阻"，最高"证据不足"。

## 七、只读复盘

**触发**：「复盘一下实盘」「策略失效了吗」「滑点比回测差多少」。

- **运行入口**：当前发行版没有只读查询客户端（`capabilities()` 中 `sentinel` 为"缺失"；客户端实施已暂停），只能用用户从 Sentinel「日志」页下载的原文，或在实盘机器上只读本机日志。解析方法见 `live-review.md` 第三节；解码函数按已安装包实际提供的为准，没有就按原文 JSON 字段读。同一窗口、同一参数用第二节的入口重跑回测作对照。
- **最少证据**：核账闭合、确认按设计运行、版本窗口、数据截止时间。
- **关卡**：不发起交易、不改实盘、不索要凭据；账不平不做归因；半年实盘的好坏不当作确认或证伪（`live-review.md` 第二节）。
- **缺证状态**：拿不到原文 → "受阻"，请用户下载；自算的复盘数字标"近似"。
