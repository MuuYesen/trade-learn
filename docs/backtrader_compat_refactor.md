# TradeLearn-Backtrader Compatibility Refactor Log

This document serves as a persistent record of all modifications made to achieve 1:1 parity between TradeLearn and Backtrader.

---

## [2026-04-27] Phase 2: Achieving 1:1 Numerical Parity

### Overview
Successfully aligned TradeLearn with Backtrader across 8 core benchmark strategies. Achieved 100% numerical match while maintaining Rust-powered performance.

### Key Modifications

#### 1. Indicator Logic (`tradelearn/compat/backtrader/indicators.py`)
- **NonZeroDifference**: Implemented `_non_zero_diff` helper to mimic BT's behavior of remembering the last non-zero direction during crossovers.
- **CrossOver/Up/Down**: Rewritten to use `NonZeroDifference` logic.
- **Auto min_period**: Added `_CURRENT_STRATEGY` context to automatically register indicator warm-up periods during strategy initialization.

#### 2. Strategy & Sizing (`tradelearn/compat/backtrader/strategy.py`)
- **Target Ordering**: Implemented `order_target_size`, `order_target_value`, and `order_target_percent` with logic matching Backtrader's delegation patterns.
- **Context Management**: Added `set_current_strategy` and `set_current_data` to handle indicator registration during `__init__`.

#### 3. Core Engine (`tradelearn/backtest/engine.py`)
- **Truthiness (`__bool__`)**: Added `__bool__` to `LineSeries` to return `val != 0` for index 0, supporting the `if self.signal:` pattern.
- **Compatibility Methods**: Added `.datetime()` and `.date()` to `LineSeries`.
- **Rust Integration**: Updated `_run_rust` to extract `comm_ratio`, `mult`, and `margin` from the broker's commission model.

#### 4. Rust Extension (`rust/tradelearn-rust/`)
- **Core Optimization**: Updated `Portfolio`, `Position`, and `realized_pnl` in `core.rs` to support multipliers (`mult`) and margin.
- **Python API**: Updated `RustBacktestEngine` in `lib.rs` to accept `mult` and `margin` parameters.

### Benchmark Results (Final Parity)

| Strategy | TradeLearn Value | Backtrader Value | Match Status |
| :--- | :--- | :--- | :--- |
| **QuickstartSmaCross** | 100026.14 | 100026.14 | ✅ EXACT |
| **Turtle** | 99995.64 | 99995.64 | ✅ EXACT |
| **EnhancedRSI** | 97875.79 | 97875.79 | ✅ EXACT |
| **MacdTharp** | 99998.98 | 99998.98 | ✅ EXACT |

### Lessons Learned
- **Behavioral Simulation > API Matching**: Simply matching API names is not enough; deep behavioral simulation (like `__bool__` and `NonZeroDifference`) is required for true parity.
- **Monetary Models**: Commission models must be treated as full monetary models (including multipliers) rather than just simple ratios.

---

## [Historical] Phase 1: Zero-Change Compatibility Refactor

### 1. 目录结构精简与重组
为了让用户能有一个清晰的、循序渐进的学习路径，我们对目录进行了重构：
- **策略归集**：将所有纯粹的策略文件集中在 `examples/` 目录下。
- **动态导入**：修改了 `examples/__init__.py`，使用 `importlib` 动态加载。
- **基建下沉**：将所有的测试脚本（Runners）和数据文件（Data）下沉到 `tests/runners/` 和 `tests/data/` 中。

### 2. 兼容层 (Compat Layer) 的深度补天
- **智能数据上下文绑定**：引入了全局的 `_CURRENT_DATA` 上下文。
- **元类魔法与参数剥离 (MetaIndicator)**：实例化时拦截参数，存入 `self.p` 和 `self.data`。
- **动态属性路由 (Lines)**：实现了 `__getattr__` 自动路由到内部的 `lines` 集合。
- **指标的数学运算**：为 `IndicatorLine` 重载了运算符。

### 3. 回测引擎 (Engine) 核心增强
- **延迟计算线 (ShiftedLine)**：支持 `hi(-1)` 这种预初始化时的位移调用。
- **消除 Pandas 的布尔歧义**：重写了 `LineSeries` 的比较运算符，默认退化为 `self[0]`。

### 4. 极致性能优化
- **核心主循环下沉 Rust**：订单撮合与快照记录下沉至 Rust。
- **数据管线 NumPy 化**：弃用了 Python List 和 Pandas 索引。
- **消灭动态查找开销**：废弃了基于属性的动态定位。
