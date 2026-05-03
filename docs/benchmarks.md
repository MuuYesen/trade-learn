# 对齐与性能基线

trade-learn 的正确性先看 Engine。`tradelearn.engine` 以 Backtrader 为 oracle，`benchmarks/runners/benchmark_bt.py` 会对代表性策略做最终权益、交易明细和 PnL 对齐；当前主线要求 Engine 与 Backtrader 保持 `EXACT`。

Lite 不是 Backtrader facade，它是更薄的策略语法层。Lite 与 Engine 共用 `tradelearn.backtest` runtime 和 Rust 撮合内核，因此 Lite 的验收重点是：API 能正确接入同一 runtime，返回统计字段与 Engine 一致，最终权益 / 成交数 / 平仓交易数与同一策略语义保持一致。

## 大样本性能

本机基线只看两个问题：结果是否对齐、吞吐是否明显快于 Backtrader。

### 55 万 bar 单标的 SMA

| 引擎 | 耗时 | bars/s | 加速比 | Final Value | Fills | Closed Trades |
|---|---:|---:|---:|---:|---:|---:|
| Tradelearn Lite | 1.3253s | 414,990 | 27.9x | 118,399.33 | 10,299 | 5,149 |
| Tradelearn Engine | 3.3767s | 162,883 | 11.0x | 118,399.33 | 10,299 | 5,149 |
| Backtrader | 37.0270s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 |

### 1000 标的、约 20 年、月频 top-50 目标权重

总计 5,040,000 根 data bars。

| 引擎 | 耗时 | bars/s | 加速比 | Final Value | Completed Orders | Target Intents | Targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tradelearn Lite | 2.407s | 2,094,237 | 119.1x | 4,199,638.26 | 23,249 | 23,249 | 239 |
| Tradelearn Engine | 4.112s | 1,225,594 | 69.7x | 4,199,638.26 | 23,249 | 23,249 | 239 |
| Backtrader | 286.538s | 17,589 | 1.0x | 4,199,638.26 | 23,249 | 23,249 | 239 |

结论：Engine 与 Backtrader 在最终权益、完成订单数、目标意图数和 rebalance 次数上全部 `EXACT`；Engine 约为 Backtrader 的 **69.7x**，Lite 约为 **119.1x**。这些数字不作为跨机器的绝对性能承诺；正式质量门禁仍然是 Engine 与 Backtrader 的订单生命周期和交易明细保持 `EXACT`。

## 复现命令

```bash
uv run python benchmarks/runners/benchmark_bt.py smart --repeat 1 --warmup 0

uv run python benchmarks/runners/benchmark_bt.py smart --repeat 1 --warmup 0 --include-portfolio

uv run python benchmarks/runners/benchmark_throughput.py --bars 550000 --repeat 1 --warmup 0

uv run python benchmarks/runners/benchmark_target_weight_parity.py \
  --symbols 1000 --bars 5040 --holdings 50 --rebalance-every 21
```
