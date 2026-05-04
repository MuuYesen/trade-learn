# 因果推断指南 (Causal Inference)

在量化投研中，**相关性 (Correlation) 不等于因果性 (Causality)**。传统的机器学习策略往往会陷入“伪相关陷阱”，导致回测表现优异但实盘迅速失效（因子失效）。

`trade-learn` 创造性地将因果推断引入因子研究流水线，帮助开发者发现驱动收益的真正原因。

---

## 核心组件：`CausalSelector`

`CausalSelector` 是 `tradelearn.ml` 中的核心工具，它基于因果发现算法（如 PC, FCI）对因子特征进行筛选。

### 1. 为什么需要因果筛选？
传统的特征筛选（如相关系数、信息增益）只能识别特征与目标在数值上的关联，而无法区分以下情况：
- **直接因果**：因子 A 确实驱动了收益 Y。
- **共同观测 (Common Cause)**：因子 A 和收益 Y 都是由隐藏因素 Z 驱动的，A 并不产生预测力。

### 2. 快速上手

```python
from tradelearn.ml import CausalSelector

# 初始化因果选择器，使用 PC 算法
selector = CausalSelector(
    method="pc", 
    alpha=0.05, 
    max_reach=2
)

# 对特征集进行因果筛选
# data: DataFrame, 必须包含特征列和目标列
selected_features = selector.fit_transform(data, target="label")

print(f"原始特征数: {data.shape[1]}")
print(f"因果有效特征: {selected_features.columns.tolist()}")
```

---

## 在投研流水线中使用

因果筛选通常位于“特征工程”之后，“权重分配”之前。

```python
from tradelearn import research

# 1. 定义特征集
feature_set = research.FeatureSet(factors, target=label)
df = feature_set.fit_transform(bars)

# 2. 因果筛选（剔除伪相关因子）
selector = CausalSelector(method="fci")
df_causal = selector.fit_transform(df, target="label")

# 3. 基于因果特征进行训练或打分
scores = model.fit(df_causal).predict(test_causal)
```

---

## 技术背后的原理

`trade-learn` 深度集成了 `causal-learn` 生态：
- **PC 算法**：通过条件独立性测试构建因果图的有向无环图 (DAG)。
- **FCI 算法**：在存在潜变量（无法观测的混杂因素）的情况下，依然能尝试还原因果关系。

!!! tip
    **什么时候使用？**
    当你的因子池非常庞大（例如 >1000 个 Alpha 因子）且担心过拟合时，`CausalSelector` 是极佳的“去噪过滤器”。

!!! important
    **因果性与未来函数**
    本章节探讨如何利用 `CausalSelector` 避免在策略研究中误入未来函数的陷阱。

---

## 接下来阅读
- [研究指南](research.md)：了解如何将因果筛选嵌入完整 Pipeline。
- [API 参考 - ML](../api/reference/ml.md)：查看 `CausalSelector` 的详细参数。
