# 2026-10-03 文档契约审计与修复

审计基线：GitHub master `a5d3648ace4764315138556b721d68ba1904ed5e`（PR #6 已合并）。
在独立 worktree 修改；原始本地工作区包含未提交的部署、指标及接口修改，未整体覆盖或上传。
依据：LeafQuant 通用开发指南及本仓库 contracts、matching、runtime、stats 文档。

## 为什么先前漏检

1. 主要验证既有 happy-path 测试，没有将文档中的时点、资金、缺失证据逐项转成反例。
2. 多资产测试通常共用整齐时间轴，未覆盖目标资产停止更新而主时钟继续前进。
3. Python 与 Rust 扩展、GitHub 与未提交本地源码没有始终绑定同一版本。
4. oracle 依赖未跟踪的本机文件，旧导入可悄悄落到已安装包；测试模块还污染全局 provider 导入。
5. Rust integration tests 引用已删除模块、PyO3 强制 extension feature 导致测试不能运行；没有有效 Rust CI 门禁。

## 已确认并修复

| 领域 | 反例与错误 | 修复与回归 |
|---|---|---|
| 有效期 | 目标资产仅首日有 bar，主时钟前进后限价单仍 Accepted | 主时钟撮合前扫过期队列；三 runner、双模式、open/close |
| 资金 | 现金100、价格10能卖空100万股；卖空所得还能继续开仓 | 预演成交，冻结卖空所得+等额成本保证金；覆盖连续/同批开空、跨资产、平空释放、反转、费用与乘数 |
| 止损限价 | smart 使用触发前价格成交；普通止损限价跨 bar 丢失激活 | 仅匹配触发后路径、持久激活；保留 exact 同 bar OHLC 约定 |
| 时间单位 | 微秒 UTC 索引重采样到1970年 | 显式转换秒单位；s/ms/us/ns与三种时区输入 |
| 数据频率 | TDX `1m` 实际请求月线 | 统一 `1m` 分钟、`1M`/`M`/`monthly` 月线与输出元数据 |
| 数据唯一性 | 重复 timestamp/symbol 通过边界校验 | 拒绝重复键，仍允许不同资产同时、停牌成交量0/NaN |
| 收益证据 | 缺价默认前填为零收益；propagate统计跳过NaN | 不自动补价；累积与标量统计传播缺失，拒绝无限值、非法年化周期 |
| ML 示例 | 全样本选因子、训练并在同批数据回测；标签缺价自动填补 | 时间留出、边界标签purge、训练窗口选特征；未来改价不改变训练模型 |
| 因子 | 未定义相关系数伪0；Alpha066除零后变有效排名 | Alpha002/003保留NaN；066非有限中间值不进入排名；数学/legacy oracle分别验证 |
| 测试隔离 | oracle导入改写sys.path、污染provider对象 | 成功与异常都恢复导入状态；固定快照及SHA256校验，无installed fallback |
| 测试可运行性 | 陈旧Rust API和examples路径、私有部署测试混入库suite | 恢复Rust测试、当前example入口、外部策略显式集成目录，保留原断言 |
| 基准工具 | worker硬退出永久等待；合成bar索引对齐后全NaN | 有界子进程结果收集；有效合成行情、独立预热与稳态计时 |
| 浮点与排名 | 加权窗口累加次序产生1 ULP差异，进一步改变Alpha191排名；无穷值被排名伪装 | 共用有界窗口归约，保持NumPy归约次序；非有限窗口输出NaN；17项独立数学回归 |
| MCP兼容 | 延迟类型注解使最低支持的MCP版本处理工具参数失败 | 使用真实类型注解，保留工具接口；相关回归通过 |

## 行为变化

- 资金不足保留现有 `Margin` 终态，修正文档曾写 `Rejected` 的漂移；cash仍为账面现金。
- 保证金按开仓成本1:1占用，不增加动态追保或自动强平；纯减仓在不负现金时允许执行。
- 不再给缺失证据制造有效收益/因子值；相关历史输出可能改变，这是正确性修复。
- `tests/integration/examples` 需显式配置外部策略目录，未配置会说明跳过，错误配置失败；不宣称已验收外部实盘策略。
- 本次验证不发送订单、不重跑生产换仓、不部署服务器。

## 验证记录

验证环境为本机 macOS / Python 3.12.4，Rust扩展从本次源码重新构建。分组执行后只重跑受影响模块；不是一次未中断的全仓运行，也不代表已完成 Linux/Windows/Python 全版本矩阵。

| 检查 | 结果及边界 |
|---|---|
| backtest（排除两个基准脚本测试文件） | 621 passed、29 skipped；28个单资产不适用组合和1个缺失旧设计文档 |
| report/strategy/core/data/docs/release/examples/engine/lite | 346个独立用例有通过记录；初次报告标题失败后，report整目录113项复测通过 |
| factor/indicators/ml/research | 170个独立用例有通过记录；修复归约差异后，rolling+Alpha101/191共30项复测通过 |
| metrics、顶层unit、golden、consistency | 212个独立用例有通过记录，41项缺私有示例或未来阶段脚手架跳过；MCP/Lab/optimize相关16项复测通过 |
| benchmark工具 | 14 passed、1 deselected；完整8策略耗时测试单独处理 |
| 实际策略对照 | 8/8策略分别与Backtrader结果一致；分批验证，不宣称最终单进程完整脚本通过 |
| stage3基准+文档 | 11 passed（其中7项文档与上述分组重叠） |
| core + core/data/metrics doctest | 61 passed（与上述分组有重叠） |
| Rust | `cargo test --workspace --locked --offline`：21 passed |
| oracle | `python scripts/check_oracle.py`：oracle、TDX、TV readiness均ok |
| 锁文件 | `uv lock --check --offline`通过 |

- 完整CI Ruff路径检查通过。新增冻结reference快照保留原始字节（含历史尾随空格）以保持SHA256；其余改动的 `git diff --cached --check -- . ':!reference/tradelearn_1x'` 通过。
- `python -m interrogate tradelearn/core tradelearn/data tradelearn/factor tradelearn/indicators tradelearn/metrics tradelearn/report --fail-under 90`：91.2%通过。补充6模块接口说明，去docstring并规范化已有import格式后，执行AST与基线等价。
- `pytest tests/unit/metrics tests/consistency/test_metrics.py --cov=tradelearn.metrics --cov-report=term-missing --cov-fail-under=100`：100 passed，524/524语句覆盖，100.00%。无需外部Alphalens；扩展到可选外部parity的试验运行曾在导入子进程等待中中断，不作为通过记录。

显式执行 `pytest tests/integration/examples -q` 得到3个模块跳过，原因均为未配置外部目录。

私有策略集成测试现位于 `tests/integration/examples`。未配置 `TRADELEARN_STRATEGY_EXAMPLES_ROOT` 时明确跳过，不能据此声称生产策略通过。没有执行生产换仓、服务器部署或交易。

## 本地版本边界

原发布目录HEAD为 `3ebf7e3`，包含用户未提交修改。以共同祖先 `b1c8d4f` 做三方预演，发现8个内容冲突及3个修改/删除冲突；没有自动覆盖或重置原目录。最终审计成果保存在独立持久worktree，推送GitHub现有 `master`。原发布目录不应被描述为已与master完全同步。最终工作区：`LeafQuant/.worktrees/tradelearn-document-audit-20261003`；分组日志与同步预演留存在同级 `tradelearn-audit-evidence-20261003`。

提交号以包含本报告的Git提交为准。
