# 更新日志 (Changelog)

本项目的所有重大变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本 (Semantic Versioning)](https://semver.org/spec/v2.0.0.html)。

## [0.2.3] - 2026-05-05

### 修复
- **进程生命周期管理**: 修复了 `tradelearn lab` 在按下 `Ctrl+C` 后无法彻底关闭 JupyterLab 和 MCP 服务子进程的问题，增加了强制清理逻辑防止端口冲突。
- **Jupyter 审美优化**: 修复了 Jupyter 环境下进度条显示粉色背景和重复渲染的问题，自动切换至 `tqdm.notebook` 蓝色原生进度条。

## [0.2.2] - 2026-05-05

### 新增
- **智能控制台反馈系统**:
    - 实现了 `SmartPbar` (tqdm 子类) 和 `ConsoleState` 来管理终端美学。
    - 添加了“智能间距”逻辑，自动确保执行阶段（数据获取 -> 因果推断 -> 回测运行 -> 报表生成）之间恰好有一个空行。
    - 集成了 `smart_print`，将手动输出与控制台状态跟踪器同步，防止多余或缺失的换行。
- **增强型回测指标**:
    - 引入了 **SQN (System Quality Number)**，用于评估系统的稳健性。
    - 添加了 **凯利准则 (Kelly Criterion)**，提供最优仓位规模建议。
    - 集成了 **市场暴露率 (Exposure Time %)**，追踪资金占用效率。
    - 扩展了回撤追踪，新增 **最大/平均回撤持续时间 (Drawdown Duration)**。
    - 标准化控制台输出为高密度单列排版，确保专业级的投研报告展示。

### 变更
- **终端 UI 精细化**:
    - 将所有旧有的 `print` 和 `logging` 进度指示器替换为同步的 `smart_tqdm` 进度条。
    - 屏蔽了 `tvDatafeed` 的冗余第三方警告，营造更纯净的研究环境。
    - 重构了 `Backtest.run` (Rust 内核)，支持毫秒级执行下的进度条可见性。
- **报告流水线**:
    - 优化了 `Reporter` 序列化逻辑，确保 JSON 输出能够优雅处理 `pandas.Timestamp` 和 `pandas.Timedelta` 对象。
    - 标准化了 `CausalSelector` 报告通过 `to_section()` 的嵌入方式。

### 修复
- 解决了由于流交织导致的进度条 0% 和 100% 状态出现在不同行的问题。
- 修复了 HTML 报告生成中，由于策略包含复杂时间指标导致的 JSON 序列化错误。
- 修正了交易持续时间计算，确保在所有回测引擎中正确捕捉 `dtopen` 和 `dtclose`。

## [0.2.0] - 2026-05-04

### 新增
- **Trade-Learn 2.0 正式发布**。
- 采用 Rust 编写的高性能回测内核。
- 集成用于因子筛选的因果推断套件。
- 使用 Bokeh 构建的现代 HTML 交互式 Tear Sheet。
- 支持多资产截面回测。
