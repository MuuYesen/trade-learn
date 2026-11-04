# 因果特征筛选

`CausalSelector` 在已有研究接口中选择候选特征。`method="correlation"` 是默认的绝对 Pearson 相关性基线；`pc` / `fci` 显式调用 causal-learn。在各算法及条件独立性检验假设下得到的结构是候选关系，不证明因果效应、收益或样本外有效性。

## 实际 API

```python
from tradelearn.ml import CausalSelector

selector = CausalSelector(target="label", features=["momentum", "range"],
                          method="pc", alpha=0.05, indep_test="fisherz")
selector.fit(training_data)        # 只含已成熟训练样本；包括 label
X_train = selector.transform(training_data)
X_validation = selector.transform(validation_features)  # 预测输入不需要 label
```

`target` 是构造参数；`fit(data)` / `fit_transform(data)` 不接 `target=`，也没有 `max_reach` 参数。`selected_features_` 为列名；PC/FCI的`scores_`是目标邻接的0/1指示，不是效应强度、方向或置信概率。`max_features`只限制数量；同分按列名排序，不表示因果重要性排序。相关性/backend重拟合会清除上次发现的图。

## 用现有 FeatureSet、Pipeline 显式接入监督标签

不需要另一套管线。`FeatureSet.transform`默认排除target；`Pipeline`顺序调用fit/transform，不自动路由监督标签，也不自动训练最后一个预测模型。先分离训练/验证角色，再在选择器fit时显式接上训练标签：

```python
from sklearn.ensemble import GradientBoostingRegressor
from tradelearn.research import FeatureSet, Pipeline
from tradelearn.research.preprocess import StandardScaler
from tradelearn.research.temporal import forward_return, purged_splits

# bars: 已验证的UTC MultiIndex(timestamp, symbol) OHLCV面板，唯一且有序。
# 此标签是逐symbol下一条观测的价格收益，不是可执行交易收益。
labels = forward_return(bars, horizon=1, unit="observations")
feature_set = FeatureSet(
    {"momentum": lambda p: p.close.pct_change(fill_method=None),
     "range": lambda p: (p.high - p.low) / p.close},
    target={"label": labels["forward_return"]},
)
dataset = feature_set.fit_transform(bars, include_target=True)
parts = purged_splits(labels, train_end="2024-07-01", validation_end="2024-10-01")
names = ["momentum", "range"]
train = dataset.loc[parts["train"]].dropna(subset=names + ["label"])
validation = dataset.loc[parts["validation"]].dropna(subset=names)
preprocess = Pipeline([StandardScaler(columns=names)])
X_train = preprocess.fit_transform(train[names])
selector = CausalSelector(target="label", features=names, method="pc")
selector.fit(X_train.join(train[["label"]]))
if not selector.selected_features_:
    raise ValueError("没有入选特征；保留该结果，不强凑特征训练")
model = GradientBoostingRegressor(random_state=7)
model.fit(selector.transform(X_train), train["label"])
validation_scores = model.predict(selector.transform(preprocess.transform(validation[names])))
```

`purged_splits`按每条label_end剔除跨下一段边界的标签；它不自动证明特征可得、股票池、修订或所有多资产信息区间都合格。预处理、选择器和模型只在训练段fit；验证/测试只transform/predict。以上是一次切分示例，不是自动walk-forward，也不授权反复查看Test选参数。固定日期需按实际输入与研究角色选择。

可直接阅读[现有单资产策略示例](../../scripts/examples/ml_strategy.py)：`--selection-method correlation|pc|fci`明确方法，默认correlation；其70%时间切分及一步purge仅针对该单资产下一bar标签。该例在训练段筛选/拟合，测试段由策略预测并进入回测，不将示例成绩称因果证明。

## 图与报告

`to_mermaid()`读取每条边的两个实际端点。TAIL→ARROW、反向、双箭头、无向及双CIRCLE分别显示；混合CIRCLE、STAR、NULL或复合端点用无定向连线加左右端点名称保留不确定性，不伪造统一方向。多边不折叠，未知端点明确拒绝。`plot()`继续使用causal-learn的GraphUtils/pydot，另需Graphviz；两种绘图路径不是同一renderer。

```python
from tradelearn.report import Reporter
Reporter(stats).html("report.html", sections=[selector.to_section("候选因果关系")])
```

报告sections支持Mapping、现有回调和带html/excel方法的对象。HTML字典取`html`，Excel字典取`excel`；`to_section()`仅提供HTML，不自动生成Excel图。图的标题/列名会转义，不能把结构图当作干预效应估计。

## 模型与探索工具边界

`ModelRegistry.load`可加载既有MLflow模型；ResearchRun追踪参数/结果，并不自动序列化整套已拟合模型、预处理器和选择器。持久化这些训练产物由调用者明确完成。

未发布源码已移除`tradelearn.ml.AutoML`及`tradelearn.ml.automl`。原随机半分的分类器排行榜移到[探索示例](../../examples/exploration/classifier_comparison.py)的`compare_classifiers`；它只返回排行榜，不是时序验证、预测模型或交易策略。底层`tradelearn.ml._lazy_predict`的显式train/test模型比较实现保留，不新增兼容壳。正式0.2.9历史产物不变。

[研究指南](research.md) · [ML API](../api/reference/ml.md)

### 探索评分与失败诊断（未发布源码）

底层 `LazyClassifier` / `LazyRegressor` 接收调用者已划分的训练和评估样本，
不替代上面的时序验证。`fit` 返回形式、评分列与可选预测表保持不变。
每次调用会清空上次模型和诊断：`failures_[model]` 记录构造、拟合或预测失败的
`stage`、`error_type`、`message`；这些模型不进入成功评分表。
`score_errors_[model][metric]` 记录单项评分不可用原因，评分为 NaN，
同一模型的其它评分和预测保留。`ignore_warnings` 只控制打印，不隐藏诊断。

分类 ROC AUC 目前只支持二分类：正类为已拟合 `classes_[1]`，
优先使用 `predict_proba` 的对应列，否则使用一维 `decision_function`。
不会使用硬预测标签替代连续分数；已提供的概率方法报错或格式错误时也不会
静默转用另一方法。评估集必须含两个已拟合类别；多分类、单类、未知标签、
缺少连续评分、非有限分数或评分异常均为 NaN 并有具体原因。
这没有引入多分类 OvR/OvO 或平均规则。分析表 NaN 不应跨 JSON 直接输出；
需要转换为 null 并携带原因。

```python
from tradelearn.ml._lazy_predict import LazyClassifier

comparison = LazyClassifier(classifiers=[classifier_type], predictions=True)
scores, predictions = comparison.fit(x_train, x_valid, y_train, y_valid)
print(comparison.failures_)
print(comparison.score_errors_)
```
