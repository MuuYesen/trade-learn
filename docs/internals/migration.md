# 版本迁移 (Migration)

trade-learn 2.0 是对 1.x 的**完整重构**。2.0 不保留 API 兼容，本页的作用是说明**核心变更内容**，帮助从 1.x 迁移的用户快速建立新版本的心智模型。

!!! tip
    如果你是新用户，直接阅读 [快速开始](../quickstart.md) 即可。本页面向：曾经接触过 1.x 代码、维护 1.x fork、或在阅读历史 commit 时遇到旧 API 的读者。

---

## 架构层变动

| 方面 | 1.x (旧) | 2.0 (新) |
|---|---|---|
| **撮合引擎** | Python (基于 backtesting.py) | **Rust (高性能原生内核)** |
| **用户 API** | backtesting.py 混合风格 | **严格 Backtrader 兼容风格** |
| **Vendor 策略** | 内嵌第三方源码 | 核心自研逻辑，第三方库仅作为依赖 |
| **数据源** | 分散的数据抓取脚本 | **统一 Data Provider 契约 (TV/TDX)** |
| **开源协议** | 未明确 | **Apache-2.0** |

## 用户 API 对照

| 操作 | 1.x | 2.0 |
|---|---|---|
| **入口类** | `Backtest(data, Strategy).run()` | `Cerebro().run()` (Engine) / `Backtest().run()` (Lite) |
| **策略初始化** | `def init(self):` | `def __init__(self):` |
| **索引习惯** | `self.data.close[-1]` | `self.data.close[0]` (**[0] 代表当前 Bar**) |
| **参数声明** | 类属性 `fast = 10` | `params = (('fast', 10),)` |
| **指标注册** | `self.I(func, ...)` | 直接引用 `tl.tdx.MA(...)` / `bt.talib.RSI(...)` |

## 指标口径统一

2.0 显式区分了三大算法口径，彻底解决了 1.x 中指标计算“对不上”的问题：

-   **`tl.pta`**: 兼容 `pandas-ta`，通用研究口径。
-   **`tl.tdx`**: 兼容 **通达信 (MyTT)**，A 股研究口径。
-   **`tl.tv`**: 兼容 **TradingView (pyneCore)**，全球/加密研究口径。

## 报告与追踪增强

| 功能 | 1.x | 2.0 |
|---|---|---|
| **报告生成** | 手动调用 pyfolio / quantstats | `Reporter(stats).report("report.html")` |
| **实验追踪** | 无 | **内置 MLflow 深度集成** |
| **可视化** | Matplotlib 静态图 | **Bokeh / Plotly 交互式图表** |
| **数据导出** | 无 | **支持导出为 Excel (XLSX)** |
| **交互探索** | 无 | **集成 Pygwalker 探索分析** |

## 项目结构演进

2.0 引入了更清晰的路径定义，移除了 1.x 的分散模块：

-   `backtest-rs/`: 核心 Rust 代码。
-   `tradelearn/compat/`: 对 Backtrader 等框架的兼容层。
-   `tradelearn/mcp/`: 现代 AI/Agent 集成协议。
-   `tradelearn/lab/`: 交互式投研环境。

!!! tip
    **只需查看本章**
    Tradelearn 2.0 经历了底层的彻底重构。对于 1.x 用户，你只需要关注本页列出的核心语义变化，大部分代码仍可通过 `tradelearn.lite` 兼容层直接运行。

## 核心依赖变动

| 依赖 | 1.x | 2.0 |
|---|---|---|
| **行情拉取** | `yfinance` (已移除) | `opentdx` / `tvdatafeed` |
| **实验记录** | 无 | **`mlflow`** (核心组件) |
| **AI 协议** | 无 | **`mcp`** (Agent 集成) |
| **研究环境** | 独立脚本 | **`[lab]` extras** (JupyterLab 集成) |

## 数据缓存路径

| 类型 | 1.x | 2.0 |
|---|---|---|
| **缓存逻辑** | 隐式（每次运行重新拉取） | **显式缓存（Parquet 格式）** |
| **本地路径** | 无固定 | `./data/{symbol}_{range}.parquet` |
| **全局路径** | 无 | `~/.cache/tradelearn/` (可选) |

## 环境配置

2.0 引入了标准化环境变量，不再依赖硬编码：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `https://mlflow.leafquant.com` | MLflow 服务端地址 |
| `TRADELEARN_DATA_CACHE_DIR` | `./data` | 本地数据缓存目录 |
| `TRADELEARN_LOG_LEVEL` | `INFO` | 系统日志级别 |
| `TRADELEARN_SEED` | 无 | 全局随机种子（确保实验可复现） |

---

## 相关阅读
- [核心契约](contracts.md)：2.0 的对象契约定义。
- [一致性审计](consistency.md)：分层数值对齐标准。
