# 与外部库的语义一致性

trade-learn 的核心价值之一是**可对照、可复现**：你写的策略在 trade-learn 上跑出的结果，必须能和原生 backtrader、empyrical、pyfolio、alphalens 等业内基线对照得上。本页列出 trade-learn 在不同层面承诺的一致性目标和容忍度。

## 为什么这件事重要

trade-learn 把 empyrical / pyfolio / alphalens / pandas-ta-classic / MyTT / pyneCore 的工作融合进自己的 metrics / report / factor / indicators 模块。一旦数值飘了：

- 用户无法用 trade-learn 的报告说服老板 / 客户 / 论文评审
- 同一份策略在 trade-learn 和 backtrader 上的回测结果会不可解释地分叉
- 框架失去研究工具的可信基础

因此 trade-learn 把"对照基线"当作工程纪律而非美好愿望——CI 跑 50+ golden 对照，差异不允许沉默放宽。

## 分层容忍度

| 层 | 对照源 | 容忍度 | 说明 |
|---|---|---|---|
| **metrics 指标函数** | empyrical / pyfolio / alphalens 原版 | `rtol=1e-10` | 纯函数，必须严格一致 |
| **`tl.pta` / `bt.pta`**（pandas-ta-classic 口径） | pandas-ta-classic 原版 | `rtol=1e-10` | 薄封装，无应有差异 |
| **`tl.tdx` / `bt.tdx`**（A 股 TDX 口径） | MyTT 原版 | `rtol=1e-10` | 融合 MyTT，须完全一致 |
| **`tl.tdx` / `bt.tdx`** 抽查 | 通达信软件截图 | `rtol=1e-6` | 手工抽查 3–5 个指标 |
| **`tl.tv` / `bt.tv`**（TradingView 口径） | pyneCore / TV 截图 | `rtol=1e-6` | pyneCore 版本差异容忍 |
| **回测 trades**（决策层） | backtrader oracle | **0 差异**（时间/方向/size） | "一份策略可同时运行并比较" |
| **回测 equity 曲线** | backtrader oracle | `rtol=1e-6` | 浮点累积允许差 |
| **回测 summary** | backtrader oracle | `rtol=1e-4` | 差异必须可解释、登记在案 |

> "0 差异"特指**决策层**——同一时刻、同一方向、同一 size 的 trade 序列必须完全一致。equity 与 summary 因为浮点累积不可避免有微小偏差，但每根 trade 必须能在两侧一一对上。

## 双口径指标命名

同一个指标在不同生态有不同算法，trade-learn 用三个子命名空间显式区分：

| 命名空间 | 口径来源 | 适用场景 |
|---|---|---|
| `tl.pta` / `bt.pta` | pandas-ta-classic | 通用研究、与 ta-lib 对齐 |
| `tl.tdx` / `bt.tdx` | MyTT（通达信软件口径） | A 股策略，需要和通达信图对得上 |
| `tl.tv` / `bt.tv` | pyneCore（TradingView 口径） | 海外 / 加密，需要和 TV 图对得上 |

例如：

```python
import tradelearn.indicators as ta

ta.MACD(close)         # pandas-ta-classic 算法
tl.tdx.MACD(close)     # MyTT / 通达信算法（DIF / DEA / MACD）
tl.tv.MACD(close)      # pyneCore / TradingView 算法
```

> 用户**显式选择**口径，框架不做自动分派。`tl.pta` 与 `tl.tdx` 在初始化方式、平滑系数等细节上有差异，混用会导致信号时点不同。

## 决策层 0 差异的工程做法

`tests/golden/` 维护一份金标数据集：

```
tests/golden/
├── datasets/             # 行情 parquet（TV subset）
├── strategies/           # 10 个标准策略（SMA / RSI / MACD / KDJ / Supertrend / pairs / momentum / ...）
├── expected/v1.0/        # backtrader oracle 跑出的 expected
│   └── (10 策略 × 5 数据集 = 50 组 stats / trades / equity_curve)
├── indicators/           # 指标金标（pandas-ta / MyTT / pyneCore 输出）
└── returns/              # metrics 测试用 returns 序列
```

CI required check：

- `tests/consistency/` 全通过（metrics、indicators、factor、report 数值层）
- `tests/golden/` 全通过（trades 0 差异、equity `rtol=1e-6`、summary `rtol=1e-4`）

## 可接受差异的登记簿（Known Differences）

任何**可解释的、被允许保留**的差异都登记在 [v1 → v2 迁移 → Known Differences](migration.md#3-known-differences) 中。**未登记的差异 = bug**，必须修代码而不是修金标。

每条登记必须填齐：

- **位置**：差异出现在哪个函数 / 模块
- **影响**：谁会遇到、影响什么
- **原因**：为什么会有差异（算法 / 精度 / 实现方式）
- **影响评估**：对用户决策的实际影响（通常是"无"或"数值 < X%"）
- **测试**：对应 consistency test 路径与放宽后的 rtol

## 性能回归守护

正确性之外，速度也不能倒退：

- `benchmarks/` 在 CI 跑 pytest-benchmark
- 与基线比较，**回归超 20% 失败**
- 结果归档到 GitHub Pages，公开可见

详见 [性能基线](../benchmarks.md)。

## 相关阅读

- [v1 → v2 迁移](migration.md)：1.x 与 2.0 的 API 对照与 Known Differences
- [契约与边界](contracts.md)：定义"什么必须对齐"的字段口径
- [撮合与成交](matching.md)：决策层 0 差异背后的撮合规则
