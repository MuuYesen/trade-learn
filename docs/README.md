# trade-learn 技术手册

<p align="center">
  <img src="tradelearn-logo.png" alt="trade-learn logo" width="620" />
</p>

欢迎来到 **trade-learn** 官方文档。
本手册旨在帮助您从零开始，构建、测试并优化工业级的量化交易系统。

---

## 快速导航

<div class="grid cards" markdown>

-   **快速开始**
    ---
    30 行代码走通第一个回测。
    [立即前往 &rarr;](quickstart.md)

-   **投研流水线**
    ---
    学习如何将因子研究、因果分析与机器学习集成。
    [探索 Pipeline &rarr;](guides/research.md)

-   **核心架构**
    ---
    深入了解 Python / Rust 混合动力内核。
    [查看架构 &rarr;](concepts/architecture.md)

-   **一致性基准**
    ---
    查看与 Backtrader 及 Empyrical 的对齐数据。
    [性能与对齐 &rarr;](benchmarks.md)

</div>

---

## 文档地图

### 1. 策略开发 (Strategy Development)
*   **[模式对比](guides/strategy.md)**：Lite 与 Engine 模式的代码实现对比。
*   **[Lite 入门](guides/lite.md)**：最简单的策略书写方式。
*   **[Engine 指南](guides/engine.md)**：Backtrader 风格的高级策略控制。
*   **[指标生态](guides/indicators.md)**：支持 TDX、TradingView、TA-Lib 等口径。

### 2. 量化投研 (Research & ML)
*   **[研究指南](guides/research.md)**：构建可复现的投研流水线。
*   **[因果推断](guides/causal.md)**：利用因果发现破除伪相关因子。
*   **[实验追踪](guides/mlflow-lab-mcp.md)**：集成 MLflow 管理海量实验。

### 3. 底层内幕 (Internals)
*   **[Runtime 与 Runner](concepts/runtime.md)**：了解 Rust 是如何调度计算任务的。
*   **[撮合机制](internals/matching.md)**：深入探讨撮合精度与事件循环。
*   **[API 参考](api/reference.md)**：详尽的函数签名与类定义。

---

## 社区与支持
*   **GitHub**: [MuuYesen/trade-learn](https://github.com/MuuYesen/trade-learn)
*   **联系作者**: muyes88@gmail.com

!!! note
    本文档持续随项目迭代更新。如果您发现任何描述不清或示例错误，欢迎在 GitHub 上提交 Issue 或 PR。
