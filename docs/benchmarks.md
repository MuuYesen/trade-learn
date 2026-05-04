# 性能基准与对齐审计

`trade-learn` 的核心原则是 **“数值对齐高于一切”**。本页提供在 55 万根 Bar 的超大规模数据和 13 个标准策略样本下的性能表现与审计存证。

---

## 1. 吞吐性能存证 (Throughput)

我们通过单标的高频压测和全市场规模调仓压测，验证引擎在不同负载下的吞吐极限。

### 55 万根 Bar 单标的对齐压测
**场景**：单标的（3 年分钟线）、双均线策略。
**验证入口**：[`benchmark_throughput.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_throughput.py)

| 引擎模式 | 处理耗时 | 吞吐量 (Bars/s) | **加速比** | 最终权益 | 成交单数 | 闭环交易 | 状态 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 | - |

### 1000 标的大规模组合压测
**场景**：1000 标的、20 年数据、月频 Top-50 目标权重重平衡，合计 **5,040,000** 根 Bar。
**验证入口**：[`benchmark_target_weight_parity.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_target_weight_parity.py)

| 引擎模式 | 处理耗时 | 吞吐量 (Bars/s) | **加速比** | 最终权益 | 完成订单 | 调仓意图 | 重平衡次数 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **2.40s** | **2,094,237** | **119.1x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| **Tradelearn Engine** | **4.11s** | **1,225,594** | **69.7x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| Backtrader (Oracle) | 286.53s | 17,589 | 1.0x | 4,199,638.26 | 23,249 | 23,249 | 239 |

---

## 2. 投研流水线耗时拆解

展示在真实的投研全流程（数据准备 &rarr; 因子计算 &rarr; 权重生成 &rarr; 回测 &rarr; 报告）中的耗时占比。

| 环节 | 耗时 (Sec) | 占比 | 业务含义 |
|---|---|---|---|
| **数据预对齐** | 0.029s | 4.0% | 多资产 OHLCV 内存对齐与数据准备 |
| **特征/因子计算** | 0.033s | 4.6% | 指标计算、特征工程与标签生成 |
| **信号与仓位生成** | 0.103s | 14.2% | 信号过滤与组合目标权重分配 |
| **核心撮合执行** | **0.025s** | **3.4%** | **Tradelearn 引擎执行撮合记账** |
| **可视化报告渲染** | 0.442s | **61.0%** | Bokeh 交互式 HTML 报告生成 |
| **结果持久化** | 0.093s | 12.8% | MLflow 记录与导出 Artifacts |

!!! tip
    **性能优化提示**：可视化报告渲染占用了超过 **60%** 的总耗时。在进行大规模参数寻优（Optimization）或回测数千个因子时，建议**关闭报告生成功能**，这将立即使你的全链路吞吐量翻倍。

---

## 3. 策略对齐审计存证 (Audit)

!!! important
    **EXACT MATCH 意味着什么？**
    这意味着 Tradelearn 的 Rust 撮合核在处理每一根 Bar 的买入、卖出、手续费扣除、盈亏计算时，与 Backtrader 产生的每一位浮点数都完全一致。

本部分通过 13 个标准策略样本，证明 Tradelearn Engine 与 Backtrader 的决策逻辑在微观层面完全等价。

### 3.1 核心 Engine 策略 (单标的)
> 表内对比值格式为：`Tradelearn / Backtrader`

| 策略名称 | 最终权益 (Final Value) | 闭环交易数 | 累计 PnL | 状态 |
|---|---|---|---|---|
| **QuickstartSmaCross** | 100026.14 / 100026.14 | 16 / 16 | 26.14 / 26.14 | **EXACT** |
| **SmaCross (Standard)** | 99630.56 / 99630.56 | 3 / 3 | -247.72 / -247.72 | **EXACT** |
| **Turtle (海龟策略)** | 99995.64 / 99995.64 | 8 / 8 | -4.36 / -4.36 | **EXACT** |
| **Enhanced RSI** | 97875.79 / 97875.79 | 6 / 6 | -2124.21 / -2124.21 | **EXACT** |
| **MacdTharp** | 99998.98 / 99998.98 | 2 / 2 | -1.02 / -1.02 | **EXACT** |
| **Order Execution** | 99994.05 / 99994.05 | 13 / 13 | -5.95 / -5.95 | **EXACT** |

### 3.2 组合调仓策略 (多资产)
> 验证多数据源对齐、`order_target_percent` 及反波动权重语义。

| 策略名称 | 最终权益 (Final Value) | 订单笔数 | 加速比 | 状态 |
|---|---|---|---|---|
| **TargetPercent Portfolio** | 104447.50 / 104447.50 | 14 / 14 | 2.8x | **EXACT** |
| **AssetClass Portfolios** | 104003.95 / 104003.95 | 21 / 21 | 3.2x | **EXACT** |
| **Trend Filtered Portfolio** | 103430.20 / 103430.20 | 21 / 21 | 3.2x | **EXACT** |
| **Inverse Volatility** | 104410.00 / 104410.00 | 9 / 9 | 3.1x | **EXACT** |

---

## 4. 投研工作流示例

我们确保 Lite 与 Engine 两套入口生成的投研结果能进入同一套 `Stats` / 报告系统。

| 范围 | 完整示例路径 |
|---|---|
| **Lite 投研流** | `examples/research/index_enhance_lite_pipeline.py` |
| **Engine 投研流** | `examples/research/index_enhance_engine_pipeline.py` |

---

## 5. 复现命令

你可以通过运行以下脚本获取上述数据的实时验证：

```bash
# 复现单标对齐与吞吐
uv run python benchmarks/runners/benchmark_throughput.py --bars 550000

# 复现多资产组合对齐 (1000 标的)
uv run python benchmarks/runners/benchmark_target_weight_parity.py --symbols 1000 --bars 5040

# 复现全量示例策略审计
uv run python benchmarks/runners/benchmark_bt.py smart --repeat 1 --include-portfolio
```
