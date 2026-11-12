# 研究指南

研究输入与输出保持 Pandas。`FeatureSet → Pipeline → ModelScorer → Allocator` 分开定义与执行，学习组件只在成熟的训练样本上 fit，运行时仅 transform/predict。`ResearchResult` 是已有结果的交付对象；上下文不会自动拦截任意 sklearn.fit。

本轮源码实现与正式发行分开：文件 Dataset、窗口与显式追踪、Riskfolio 接线和共同 Reporter 已有源码；真实平台/发行资格见本轮验收记录，不由本指南自动推定。唯一设计约束见[研究依赖规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#research-integrations-20261111)。

## 先定义，再执行

以下输入 `bars` 为 UTC `MultiIndex(timestamp, symbol)` OHLCV。每个预处理步骤独立指定 columns；标签不混入模型特征。

```python
import pandas as pd
from sklearn.linear_model import Ridge
from tradelearn import research
from tradelearn.research.temporal import forward_return, purged_splits

feature_set = research.FeatureSet({
    "momentum": lambda p: p.close.pct_change(3, fill_method=None),
})
pipeline = research.Pipeline([
    research.preprocess.Winsorizer(columns=["momentum"]),
    research.preprocess.StandardScaler(columns=["momentum"]),
])
model = Ridge(alpha=1.)
scorer = research.ModelScorer(model, features=["momentum"], current=False)
allocator = research.Allocator(select=research.TopK(2),
                               weight=research.EqualWeight(gross=.8))

with research.ResearchRun("study") as run:
    for name, component in (("features", feature_set), ("pipeline", pipeline),
                            ("model", model), ("scorer", scorer), ("allocator", allocator)):
        run.register(name, component)
    features = run.call("features", "transform", bars)
    labels = forward_return(bars, horizon=2)
    sections = purged_splits(labels, train_end="2026-06-01", validation_end="2026-09-01")
    train = features.loc[sections["train"]].dropna()
    valid = features.loc[sections["validation"]]
    train = run.call("pipeline", "fit_transform", train)
    valid = run.call("pipeline", "transform", valid)
    run.call("model", "fit", train[["momentum"]], labels.loc[train.index, "forward_return"])
    scores = run.call("scorer", "predict", valid)
    weights = run.call("allocator", "build", scores) if not scores.empty else None
    result = run.finish(model=model, scores=scores, weights=weights,
                        artifacts={"train_end": "2026-06-01", "validation_end": "2026-09-01"})
```

`register`只登记定义和参数，重复名称绑定不同对象会拒绝。`call`记录真实方法、组件参数、输入输出摘要及成功/失败，外部模型也覆盖。直接调用已带追踪的 TL 方法仍沿用现有行为；需要完整执行证据时使用显式 call。`finish`不会训练、评估或补算缺失指标。

## 时间、标签及原行情

`forward_return`记录真实`label_end`。`observations`按每标的实际观测后移，`seconds`要求精确`t+h`观测，缺失保持NA。`purged_splits`排除跨下一段边界的训练/验证标签；训练时标签必须已成熟。简单`time_split(data, split=...)`保持`train < split / test >= split`，本身不做purge或walk-forward。

对特征和原行情复用同一个split边界：`research.split_bars(bars, split=...)`保留全部后段行情，包括没有未来标签的末尾Bar。不能按每标的机械`head(-5)`清理，也不能因标签成熟裁掉回测尾部或重选股票池。历史股票池、修订、复权、交易日历及特征是否可得由数据来源提供证据。

`temporal.assessment`计算单标的沿时间的多因子IC/RankIC及分块统计；截面评价用`FactorAnalyzer`，多期限/多因子用既有`MultiPeriodFactorAnalyzer`/`MultiFactorAnalyzer`，不将单标的冒充横截面。预测标签与成本后的可执行收益分别核验。

## 有界窗口与共同推理

Lite：`self.window(size=4)`默认单源，`data=self.datas`多源。Engine：独立`tradelearn.backtest.history.window(self, size=4, data=self.datas)`，不添加策略便利方法。详见[数据窗口](data.md#共同有界窗口)。FeatureSet接受标准面板，或携带`attrs['symbol']`的单源时间索引窗口；身份缺失拒绝，不猜symbol。窗口保来源时区；研究标准化到UTC。

运行只执行已拟合的pipeline/scorer。`ModelScorer(current=True)`只使用最新截面，最新截面缺特征不会退回旧评分；空评分代表没有决策，应直接返回，不能调用close_missing造成清仓。实际清仓要显式零目标。

最短完整模板共用[_workflow.py](../../examples/research/_workflow.py)，分别接[Lite](../../examples/research/index_enhance_lite_pipeline.py)与[Engine](../../examples/research/index_enhance_engine_pipeline.py)：

```bash
python -m examples.research.index_enhance_lite_pipeline --data '/authorized/bars/*.parquet' --split 2026-06-01 --output /tmp/research-lite
python -m examples.research.index_enhance_engine_pipeline --data '/authorized/bars/*.parquet' --split 2026-06-01 --output /tmp/research-engine
```

模板不会联网取数。指定`--mlflow-uri sqlite:////tmp/tracking.db`才在运行后实际记录共同MLflow，返回真实run_id/status/error；默认不上传原行情或发布模型。输出目录中的bundle需为空，避免覆盖旧产物。输出`bundle/state.joblib`与`bundle/manifest.json`保留FeatureSet、Pipeline、模型、Scorer、Allocator、窗口、输入schema、依赖版本、标签/purge元数据和SHA256完整性信息，`bundle/code`包含显式指定的特征定义模块；`train-labels.parquet`保留真实训练label_end，`research.json`保留步骤和分区摘要。输入Bars摘要含fingerprint，原行情不自动复制到模型产物。`ml.registry.save_bundle/load_bundle`是显式本地产物接口，加载需`trusted=True`，先核摘要与依赖版本；只加载可信joblib/pickle，恢复依赖同版本Python和研究库；外部定义通过`save_bundle(..., code_paths={"package/module.py": source_path}, dependencies=(...))`显式配套，加载先核代码摘要并将可信code目录加入本会话导入路径。不自动扫描/打包整个仓库或秘密，不是无依赖可执行文件。训练/推理前缀一致性和两个运行时执行由测试核验；保存本地产物不意味着注册模型或部署。

## 可选研究组件

`Allocator/TopK/EqualWeight/Constraints/RiskfolioWeight`从research facade导出，唯一实现在portfolio。`RiskfolioWeight(objective='min_variance', covariance='historical', gross=.9)`通过`allocator.build(scores, returns=historical_returns)`接RiskfolioOptimizer；returns须在决策前可得、样本完整且有限。当前仅支持单截面、long-only最小方差；不把score当期望收益。缺数据、非法解、不可行归一化明确失败；previous_weights尚无成本/换手模型，会明确拒绝，不能据此声称已约束换手。

`SymbolicFeatureGenerator.fit(train, columns=[...], target='label')`训练后，`transform`不读label。`to_dict()`给出已选表达式、输入顺序/dtype、训练范围、填补值、每一代全部候选（在上游剪枝之前快照）；不是自动转Rust/无依赖表达式。精确恢复保存拟合对象并保gplearn依赖。gplearn 0.4.3通过每代种群append边界捕获表达式和fitness；真实依赖测试核全量候选与恢复一致性，后续上游版本需重新验收。

`CausalSelector`的correlation/PC/FCI保持显式监督fit与无label transform，图/假设和空选择保留；详见[因果筛选](causal.md)。结构邻接不等于因果效应或收益证明。

Optuna保留独立`OptunaSearch/OptunaBacktestSearch`和原生trial建议接口；时间验证、完整trial、失败与预算证据由调用方及搜索集成负责，Test不得参与选型。无自动Experiment DSL。

## 报告、记录和模型加载

数据探索及因子`report`委托共同`Reporter.from_analysis(completed_tables)`；`to_analysis()`可先获得可检查结果。HTML/Excel使用同一表格，图表在HTML保留，Excel明确图形表示限制，不在渲染中fit/回测/联网。

独立MLflow在完成后调用：`tradelearn.integrations.mlflow.log(result_or_engine_strategies, research_result=result, ...)`，不重跑或默认上传行情/模型。ModelRegistry继续独立，发布和部署显式进行；本地模板不隐式注册或加载远端Production模型。

可选组件完整示例见[components.py](../../examples/research/components.py)：同一成熟分区连接符号生成、显式因果筛选、每步columns的预处理、原生Optuna试验、模型/Scorer、Riskfolio以及单/多期限多因子时间序列评价；保完整候选/全部trial/失败状态，Test不参与选型。小预算只证明接口与时间边界，不代表策略经济有效性。真实依赖测试覆盖PC和FCI、符号恢复及组合求解。


2026-11-12 恢复接线：Factor/DataExplorer 的完成结果缓存按输入值/索引/列/dtype/来源属性、组件配置及内部计算版本失效；返回独立表格快照。HTML 图按需单独缓存，Excel-only 不构建 HTML 图。因子标量保单位、不可计算原因和样本覆盖；多期限/多因子保各自命名空间。独立 `history.window` 与权重查询优先内部 `_decision_datetime`，外部对象可兼容旧时钟方法；不把 BT 日期序数当共同决策时间。旧 ML 独立脚本仅通过显式 FeatureFeed 声明训练所选额外 Lines 接新 Engine，研究算法不变。优化入口见[现行搜索](optimization.md)。

模型代码恢复还会拒绝已加载的同名叶模块与产物代码摘要冲突，提示在新进程加载，避免 Python 模块缓存静默复用另一版本定义。相同代码摘要可复用；信任确认、依赖版本和代码摘要都不会被自动跳过。


固定符号后端按真实Python边界选择：Python3.10为gplearn0.4.2，Python3.11+为0.4.3（上游0.4.3要求Python>=3.11/sklearn>=1.8）。旧0.4.2与新版sklearn组合仅在TL拥有的该estimator实例上接入sklearn公开validate_data，保输入特征/数量/有限值校验及可pickle状态；不改第三方类/全局实现、不绕过验证。两个版本的候选捕获、transform无标签与恢复分别验收；manifest仍记录实际运行版本，不外推跨minor模型可载。
