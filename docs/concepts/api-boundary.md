# API 边界

Tradelearn 的用户 API 通过 facade 模块暴露；实现层和基础设施层不作为用户入口。

## 公开入口

用户代码应优先从这些模块导入：

| 模块 | 角色 |
|---|---|
| `tradelearn.engine` | Backtrader 风格事件驱动策略、回测、成本模型 |
| `tradelearn.lite` | 轻量策略与快速回测 |
| `tradelearn.data` | 数据源、缓存、数据探索 |
| `tradelearn.factor` | 因子分析、Alpha101/Alpha191 |
| `tradelearn.metrics` | 收益、风险、交易、因子指标 |
| `tradelearn.research` | 投研流水线、特征、实验结果 |
| `tradelearn.report` | 报告导出 |

示例：

```python
import tradelearn.engine as bt

commission = bt.CNAStockCommission()
```

## 内部实现

这些模块不作为用户 API：

| 模块 | 说明 |
|---|---|
| `tradelearn.backtest` | 回测 runtime、broker、撮合 glue |
| `tradelearn.core` | 内部契约、事件、错误、配置、日志、成本模型 |
| `tradelearn.report.charts` | 报告内部图表构造器 |
| `tradelearn.report.templates` | 报告模板 |
| `tradelearn.engine.*` | engine facade 的实现细节 |
| `tradelearn.utils` | 内部工具 |
| `tradelearn.query` | 数据查询实现细节 |
| `tradelearn.strategy` | 旧策略/实验性实现 |
| `tradelearn.cli` / `tradelearn.mcp` | 命令行和集成入口实现 |

## 实盘适配器边界

实盘 broker / data 适配器不作为稳定用户 facade。QMT、IB、CTP 等适配器建议放在私有扩展或 `deploy` 项目里维护，只通过 `tradelearn.engine` / `tradelearn.lite` 的策略意图和内部 broker-neutral 契约对接。

用户策略仍应只调用公开策略 API，例如 `buy`、`sell`、`order_target_percent`、`target_weights`。具体账户、行情、成交回报、批量状态查询由部署侧适配器负责。

不要在用户策略里写：

```python
from tradelearn.core.costs import CNAStockCommission
from tradelearn.backtest.models import Order
```

应改为：

```python
import tradelearn.engine as bt

commission = bt.CNAStockCommission()
side = bt.Order.Sell
```

## 维护规则

- 新的用户能力放到公开 facade 或由公开 facade 转发。
- `backtest` 和 `core` 可以被框架内部子模块导入，但不通过包顶层导出。
- 示例和文档中的用户代码不得从 `tradelearn.backtest` 或 `tradelearn.core` 导入。
- 内部测试需要访问契约对象时，应从具体子模块导入，例如 `tradelearn.core.broker_contracts`。
