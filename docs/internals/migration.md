# 版本迁移 (Migration)

本页中的“2.0”是历史架构阶段名称，当前工作树metadata仍为0.2.9，尚未公开发行本轮改造。源码候选与已发布0.2.9必须按产物SHA区分；当前状态见[整合记录](../../../../../docs/参考资料/paseo-session-records/20261112-two-goals/TL-INTEGRATION/STATUS.json)。本页说明旧接口的实际迁移，不承诺任意旧策略原样运行。

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
| **指标注册** | `self.I(func, ...)` | 直接引用 `tradelearn.indicators.talib/tv/tdx`；自定义共享 `Indicator` |

Engine中`bt.talib`采用BT类及输出别名（例如MACD的macdsignal/macdhist）；来源函数的signal/hist名称不可直接套用。自定义指标在逐事件next中推进，不能在init对因果前缀调用to_series并缓存空批量结果。共享Indicator不再提供to_series/wrap_indicator，普通Line的前缀接口仍在。

## 指标口径统一

2.0 显式区分了三大算法口径，逐来源验证公式、种子、暖机、缺失与流式状态；不把同名当同口径：

-   **`indicators.talib`**: TA-Lib0.8.3持久状态及批量口径，默认来源；PTA和根ta快捷层退出。
-   **`indicators.tdx`**: 兼容 **通达信 (MyTT)**，A 股研究口径。
-   **`indicators.tv`**: 兼容 **TradingView (pyneCore)**，全球/加密研究口径。

## 报告与追踪增强

| 功能 | 1.x | 2.0 |
|---|---|---|
| **报告生成** | 手动调用 pyfolio / quantstats | `Reporter(stats).report("report.html")` |
| **实验追踪** | 无 | **内置 MLflow 深度集成** |
| **可视化** | Matplotlib 静态图 | **Bokeh / Plotly 交互式图表** |
| **数据导出** | 无 | **支持导出为 Excel (XLSX)** |
| **交互探索** | 无 | DataExplorer/领域分析结果，经共同Reporter导出 |

## 项目结构演进

2.0 引入了更清晰的路径定义，移除了 1.x 的分散模块：

-   `rust/tradelearn-rust/`: 核心 Rust 代码。
-   `tradelearn/engine/`与`tradelearn/lite/`: 两种外观共用`tradelearn/backtest/`核心。
-   AI Lite通过工作区`_agents/skills/`加载版本匹配研究技能；TL自建MCP已退役。
-   `tradelearn/lab/`: 交互式投研环境。

!!! tip
    **只需查看本章**
    Tradelearn 2.0 经历了底层的彻底重构。对于 1.x 用户，你只需要关注本页列出的核心语义变化，Lite保init/类属性参数，Engine保BT外观；旧I/PTA、Engine便捷方法和非BT参数按[Engine](../guides/engine.md)/[Lite](../guides/lite.md)逐项迁移。

## 核心依赖变动

| 依赖 | 1.x | 2.0 |
|---|---|---|
| **行情拉取** | `yfinance` (已移除) | `opentdx` / `tvdatafeed` |
| **实验记录** | 无 | `[research]` 内的独立运行后集成 |
| **AI 研究** | 无 | AI Lite/Skills；TL MCP与服务包装已删除 |
| **研究环境** | 独立脚本 | **`[research]` extras** (JupyterLab/AI Lite/Skills/MLflow) |

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
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | 本地记录或用户显式tracking URI，不自动启动服务 |
| `TRADELEARN_DATA_CACHE_DIR` | `./data` | 本地数据缓存目录 |
| `TRADELEARN_LOG_LEVEL` | `INFO` | 系统日志级别 |

---

## 相关阅读
- [核心契约](contracts.md)：2.0 的对象契约定义。
- [一致性审计](consistency.md)：分层数值对齐标准。
