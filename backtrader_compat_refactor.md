# TradeLearn-Backtrader Compatibility Refactor Log

This document serves as a persistent record of all modifications made to achieve 1:1 parity between TradeLearn and Backtrader.

## 2026-04-27: Achieving 1:1 Numerical Parity

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
