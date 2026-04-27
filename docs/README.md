# 文档说明

trade-learn 2.0 重构期间的项目管理文档。

所有文档以**代码实际状态**为准,不按历史对话口头状态判断。

## 文档索引

### 1. 项目愿景

文件:[PROJECT.md](./PROJECT.md)(所有后续决策的北极星)

作用:

- 项目定位与八短语
- 路线图与阶段规划
- 核心设计原则与关键决策
- 最终架构与依赖分组
- 明确不做的事

### 2. 项目进度

文件:[PROGRESS.md](./PROGRESS.md)

作用:

- 看当前做到哪(已完成 / 进行中 / 未开始)
- 按阶段的完成度
- 各阶段任务的详细进度与剩余问题
- 版本发布节点追踪

### 3. 项目总览

文件:[PROJECT.md](./PROJECT.md)

作用:

- 项目目录结构与模块分工
- 各模块职责边界
- 依赖关系与数据流
- 1.x reference 代码位置与使用方式

### 4. 运行与操作

文件:[RUNBOOK.md](./RUNBOOK.md)

作用:

- 开发环境启动(uv / pytest / mkdocs / maturin)
- `tradelearn lab` / `tradelearn new` 使用
- MLflow 连接与验证
- 金标测试执行
- 常见问题排查

### 5. 设计规格

路径:`docs/specs/`(阶段 0 产出)

9 份核心 Spec:

- ARCHITECTURE / CONTRACTS / METRICS / INDICATORS
- BACKTEST / STRATEGY / REPORT
- CONSISTENCY / MIGRATION

作用:说明项目设计与契约,任何实现必须追溯到 Spec。

### 6. Clean-Room 设计笔记

路径:`docs/internal/`(阶段 2 Week 5 产出)

- matching-design.md(撮合算法、订单类型、成交价)
- event-loop.md(事件队列、bar 推进)
- portfolio.md(equity / margin / position 计算)

作用:Rust 撮合核实现的唯一参考,合规隔离来源。

## 新内容落地规则

| 内容类型 | 落地位置 |
|---|---|
| 新功能语义、接口契约、验收标准 | 追加到对应 `specs/*.md` 章节末尾 |
| 架构决策、技术选型 | `PROGRESS.md` ADR 表追加一行 |
| 实现设计细节（算法、数据结构、边界情况） | `internal/` 对应文件追加新章节 |
| 1.1+ 规划任务 | `PROGRESS.md` 对应阶段 checklist |

**不新建文件**：所有新内容优先追加到现有文件，确有必要拆分再讨论。

## 7. 用户文档站

路径:`docs/tutorials/` / `docs/how-to/` / `docs/concepts/` / `docs/api/`(阶段 9 产出)

mkdocs-material 驱动,docstring 为唯一源。
