# 数据模型与核心术语

`trade-learn` 的高性能很大程度上源于其对数据的组织方式。本页介绍框架的核心数据模型，并提供关键术语的统一定义。

---

## 数据模型 (Data Model)

`trade-learn` 处理的数据流经历从“静态存储”到“高性能内存”的转化：

### 1. 结构化 MultiIndex
无论是从 `Provider` 拉取还是从本地加载，数据在进入内核前统一规范化为 **Pandas MultiIndex (timestamp, symbol)**。
- **维度一 (timestamp)**：全局时钟，确保多标的对齐。
- **维度二 (symbol)**：资产标识。
- **Columns**：标准的 OHLCV + 自定义因子。

### 2. 内存中的 Panel 布局
在 Rust 侧，数据不再是零散的 DataFrame，而是一个连续的内存块（Buffer）。
- **零拷贝 (Zero-copy)**：利用 Apache Arrow，数据直接从内存映射到 Rust 内核，避免了 Python 对象序列化的巨大开销。
- **Clocked Alignment**：内核通过全局时钟推进，自动处理不同标的时间戳不齐（Missing Bars）的情况，确保回测的严谨性。

---

## 核心术语 (Glossary)

为了在投研流水线中保持沟通一致，我们将以下术语标准化：

### 运行机制类
- **Bar (K 线/棒)**：数据推进的最小时间单位。
- **Warmup (暖机期)**：策略启动初期，为了让长周期指标（如 MA200）计算就绪而预留的观测期。在此期间不触发 `next()`。
- **Next (步进)**：事件循环中调用策略逻辑的动作，代表处理“当前时刻可见数据”的机会。
- **Primary Feed (主数据源)**：`datas[0]`，决定了回测引擎的全局心钟。

### 交易与账户类
- **Order Intent (订单意图)**：用户在 `next()` 中提交的请求，尚未被内核接受。
- **Matching (撮合)**：内核根据当前 Bar 的价格（Open/Close）执行订单的过程。
- **Fill (成交单)**：订单成功执行后的确认记录。
- **Mark-to-Market (逐日盯市)**：在每根 Bar 结束时，按当前价格重新计算持仓市值和账户净值的动作。
- **PnLComm (扣费盈亏)**：扣除了交易佣金和滑点后的净盈亏。

### 结果分析类
- **Equity Curve (权益曲线)**：账户总资产随时间波动的连线。
- **Stats (结果对象)**：回测结束后的全量数据集合，包含权益、交易记录、收益指标等。
- **Drawdown (回撤)**：净值从最高点跌落的幅度，用于衡量风险。

---

## 数据的生命周期

```text
Provider (拉取) -> Parquet (缓存) -> Arrow (内存) -> Rust Core (计算) -> Stats (输出)
```

!!! note
    **统一的数据视口**
    无论你的数据源是 CSV、Parquet 还是数据库，Tradelearn 都会在进入内核前将其统一转化为内存中的 Arrow 格式，以确保策略执行时的高效。

!!! tip
    所有的术语在 `tradelearn.core` 契约中都有对应的代码对象定义。深入了解这些对象字段，请参阅 [核心契约](../internals/contracts.md)。
