# 研究指南

`tradelearn.research` 用来组织机器学习策略和指数增强研究里的固定流程。

## 单标的时间序列与秒级标签

新研究可使用 `tradelearn.research.temporal`。输入仍是按 UTC 排序、键唯一的 `MultiIndex(timestamp, symbol)`；单标的只有一个 symbol，不增加虚构的横截面。`forward_return` 支持两种窗口：`observations` 是每个标的后移 N 条，`seconds` 是精确查询 `t+N 秒`，缺秒时标签为空且不插值。结果同时给出 `label_end`。这只是收盘价预测标签，交易入场时点、点差、手续费与成交能力应在回测中独立定义。

```python
from tradelearn.research.temporal import forward_return, assessment, purged_splits

labels = forward_return(bars, horizon=5, unit="seconds")
report = assessment(factors, labels, factors=["alpha"], block="1H", min_pairs=30)
sections = purged_splits(labels, train_end="2026-01-01", validation_end="2026-02-01")
```

时间序列 IC 是单个标的沿时间的 Pearson 相关，Rank IC 是沿时间的 Spearman 相关；它们不是原有的同一时刻跨标的 IC。分块 ICIR 为有效时间块 IC 均值除以样本标准差，不年化，块数不足或方差为零时为空。`purged_splits` 会剔除标签结束时间跨越下一段边界的样本；训练得到的标准化参数仍须只在训练段拟合。秒级数据可用 `1s`、`5s`、`10s`、`30s` 的 Bars 频率声明。多因子共同样本、显著性检验与可执行交易分析尚不由此模块自动完成。

## 典型流水线

```python
from tradelearn import research

feature_set = research.FeatureSet(
    {
        "alpha": lambda p: p.close.pct_change(20)
        / p.close.pct_change().rolling(20).std(),
        "size": lambda p: p.close,
    },
    target={"future_return": lambda p: p.close.pct_change().shift(-1)},
)
features = feature_set.fit_transform(bars, include_target=True).dropna()
train, test = research.time_split(features, split="2023-09-01", level="timestamp")

pipeline = research.Pipeline([
    research.preprocess.Winsorizer(columns=["alpha"]),
    research.preprocess.Neutralizer(columns=["alpha"], exposures=["size"]),
    research.preprocess.StandardScaler(columns=["alpha"]),
])

train = pipeline.fit_transform(train)
test = pipeline.transform(test)
scores = scorer.predict(test)

allocator = research.portfolio.Allocator.topk_equal(k=50, gross=0.95, max_weight=0.03)
weights = allocator.build(scores)
```

## 自定义 Pipeline 工具

`research.Pipeline` 接受 sklearn-like transformer。只要对象提供 `fit()` / `transform()` / `fit_transform()`，就可以放进流水线。约定很简单：

- `fit(train)` 只学习训练集状态，例如分位数、均值、行业暴露系数。
- `transform(data)` 只应用已学习状态，不能偷看测试集未来数据。
- 返回值保持原来的 index，方便后续按 `timestamp / symbol` 对齐权重。

```python
import pandas as pd
from tradelearn import research


class RankNormalizer:
    def __init__(self, column: str):
        self.column = column

    def fit(self, data: pd.DataFrame):
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out[self.column] = out.groupby(level="timestamp")[self.column].rank(pct=True)
        return out

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)


pipeline = research.Pipeline([
    RankNormalizer("alpha"),
    research.preprocess.StandardScaler(columns=["alpha"]),
])

train = pipeline.fit_transform(train)
test = pipeline.transform(test)
```

如果自定义工具还要写入实验记录，可以在 `ResearchRun` 里记录步骤参数：

```python
run = research.ResearchRun("jp_us_factor_study")
run.add_step("rank_normalize", category="preprocess", params={"column": "alpha"})
```

## 投研语义 vs 实盘语义

| 语义 | 适合场景 | 计算位置 | 回测执行 |
|---|---|---|---|
| 投研语义 | 离线因子检验、指数增强、模型评估 | 策略外提前算好 features / scores / weights | 策略读取 `research_result.weights` 下单 |
| 实盘语义 | paper/live、接近真实交易的逐 bar 推理 | 策略内用 `history_panel()` 取当前可见窗口 | 策略当场生成目标权重并下单 |

投研语义更快、更适合复盘；实盘语义更接近真实交易。

## 避免训练期进入评估

```python
test_bars = research.split_bars(bars, split="2023-09-01")
stats = tl.Backtest(test_bars, Strategy).run(research_result=research_result)
```
