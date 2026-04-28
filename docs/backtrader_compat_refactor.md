# Tradelearn 架构重构总结 (微内核 + 门面模式)

## 1. 重构目标
将 Tradelearn 引擎从单体架构迁移到 **微内核 (Microkernel) + 门面 (Facade)** 架构，实现核心逻辑与 API 层的彻底解耦，同时确保与 Backtrader 基准策略的 100% 数值一致性。

## 2. 核心变更

### 2.1 目录结构调整
- **核心内核 (`tradelearn/backtest/core/`)**: 包含纯粹的执行引擎、策略基类和基础数据模型。
- **兼容门面 (`tradelearn/compat/backtrader/`)**: 提供 Backtrader 风格的 API 封装。
- **清理删除**: 删除了 `tradelearn/backtest/` 目录下过时的 `base.py`, `engine.py`, `models.py` 等单体文件。

### 2.2 技术优化
- **解耦循环依赖**: 通过在 `__init__.py` 中引入延迟加载机制，解决了重构后的模块循环引用问题。
- **全局上下文同步**: 引入了统一的 `_GlobalContext (_G)` 对象，确保在复杂初始化链中策略、数据和 Broker 的上下文状态保持一致。
- **数值精度对齐**: 
    - 修正了 `ATR`、`Highest`、`Lowest` 等指标的 `min_period` 计算逻辑。
    - 修复了由于 `__len__` 导致的策略对象真值判断 Bug（Truthiness issue）。
    - 确保了策略持仓与内核 Broker 的完全同步。

## 3. 验证结果
通过 `tests/runners/benchmark_bt.py` 验证，以下策略实现了 100% 数值对齐 (✅ EXACT)：
- QuickstartSmaCross
- SmaCross
- MigratedSmaCross
- Turtle (海龟策略)
- EnhancedRSI
- BetterMA
- MacdTharp
- OrderExecutionStrategy

## 4. 后续建议
当前架构已具备极高的扩展性。建议后续：
1. 在 `core` 层引入更多高性能的 Rust 实现。
2. 为其他框架（如 VectorBT）开发新的 Facade 门面。
