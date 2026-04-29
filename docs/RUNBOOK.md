# 运行手册

## 正式开发目录

当前正式开发和运行目录固定为:

`/Users/muyesen/MAIN/Project/Personal/trade-learn-release`

所有命令默认在此目录下执行。

## 环境准备

### Python 环境

- Python **3.10 / 3.11 / 3.12**(不支持 < 3.10)
- 包管理器:**uv**(替代 pip + venv,阶段 0 迁移)

```bash
# 安装 uv(macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境 + 装依赖
uv sync                          # 用 pyproject.toml + uv.lock
uv sync --extra lab              # 含 JupyterLab
```

### Rust 工具链(开发 Rust 核时需要)

```bash
# 安装 rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# 构建 Rust 核到 Python 包
cd backtest-rs/
maturin develop --release        # 产物自动安装到 tradelearn/backtest/
```

用户不需要装 Rust,`pip install trade-learn` 会装预编译 wheel。

## 分支与工作流

### 主分支

| 分支 | 用途 |
|---|---|
| `master` | 1.x 稳定版,不动 |
| `v2` | 重构主干,所有新工作在此 |
| `feature/*` | 单个任务分支,PR 合回 v2 |

### 典型工作流

```bash
git checkout v2
git pull
git checkout -b feature/metrics-sharpe
# ... 改代码 ...
uv run pytest tests/unit/metrics/
git commit -m "feat(metrics): add sharpe ratio with unified periods"
git push origin feature/metrics-sharpe
# GitHub 开 PR,CI 通过后合回 v2
```

### 1.x reference oracle

冻结副本放在 `reference/tradelearn_1x/`,**整个重构期不动**。

CI 里作为对照源:

```python
# tests/consistency/test_xxx.py
from reference.tradelearn_1x.backtest import Backtest as OldBT
from tradelearn.engine import Cerebro

# 同样输入,两者必须产出一致
```

## 常用命令

### 测试

```bash
uv run pytest                                # 全部
uv run pytest tests/unit/                    # 仅单元
uv run pytest tests/consistency/             # 一致性对照(重构期关键)
uv run pytest tests/golden/                  # 金标对照
uv run pytest --doctest-modules tradelearn/  # docstring 示例
```

### Backtrader 对齐与 Lite smoke

```bash
# Backtrader 8 个迁移策略数值对齐
uv run python benchmarks/runners/benchmark_bt.py

# CI 使用的稳定性能口径: warmup + repeat median,低于阈值或非 EXACT 会失败
uv run python benchmarks/runners/benchmark_bt.py smart --warmup 1 --repeat 3 --min-speedup 1.2

# Lite 语法层与 1.x 策略 smoke
uv run pytest tests/unit/lite tests/unit/examples/test_1x_strategy_examples.py -q
```

说明:

- `benchmark_bt.py` 是底层正确性的正式验收,必须保持 Backtrader `EXACT`。
- Lite 只测试语法适配层是否接入同一 runtime,不单独做撮合/portfolio 对齐。
- CI 口径使用 `--warmup 1 --repeat 3`,避免 0.1s 级别单次噪声误判。
- `--min-speedup` 是退化门槛,不是发布宣传性能目标;Backtrader 当前设置为保守的 `1.2x`。
- Backtrader 策略不提供 `Strategy.I`;指标缓存接在 `bt.indicators.*` 内部。
- Lite 策略使用 `Strategy.I(...)`,内部复用 `BatchIndicatorCache`。
- 不再维护 `backtesting.py` 对齐 runner;正式测试不应 import `backtesting.py`。

### 吞吐 Benchmark 与 profiling 口径

`benchmark_bt.py` 用于 Backtrader 数值对齐审计,不是极限吞吐测试。大数据吞吐使用
`benchmark_throughput.py`:

```bash
uv run python benchmarks/runners/benchmark_throughput.py --bars 550000 --repeat 1 --warmup 0
```

2026-04-29 本地 55 万根 1min K 线实测。该 runner 使用默认 next-bar 语义,并要求
Engine、Lite、Backtrader 的 final value / fills / closed trades 口径一致:

| 引擎 | 时间 | bars/s | vs 1,682 bars/s | vs 419,552 bars/s | Final Value | Fills | Closed Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tradelearn Engine | 3.0494s | 180,364 | 107.2x | 43.0% | 118399.33 | 10,299 | 5,149 |
| Tradelearn Lite | 2.0122s | 273,334 | 162.5x | 65.1% | 118399.33 | 10,299 | 5,149 |
| Backtrader | 33.6004s | 16,369 | 9.7x | 3.9% | 118399.33 | 10,299 | 5,149 |

同一策略阶段拆分 profiling:

| 阶段 | Engine | Lite | Backtrader |
|---|---:|---:|---:|
| 数据生成 | 0.0224s | 0.0224s | 0.0224s |
| DataFrame copy | 0.0096s | 0.0014s | 0.0015s |
| runner/feed 初始化 | 0.0333s | 0.0054s | 0.0001s |
| `run()` 主循环 | 2.9494s | 2.0368s | 33.2570s |
| 结果读取 | ~0s | 0.0077s | ~0s |
| 合计 | 2.9923s | 2.0513s | 33.2588s |
| bars/s | 183,803 | 268,119 | 16,537 |

结论:

- 55 万 bar 下,Tradelearn Engine 约 **18.0 万 bars/s**,Lite 约 **27.3 万 bars/s**。
- 99% 以上时间在 `run()` 主循环,数据构造、feed 初始化和结果读取不是瓶颈。
- cProfile 显示主要开销来自每 bar Python 回调、策略 `next()`、`position()`、indicator/line
  `__getitem__` 和成交回传;Rust `run_bar_loop` 包含 callback 调度,其中 Python `on_rust_bar`
  回调仍是主成本。
- 和 419,552 bars/s 的差距主要来自 Tradelearn 保留事件驱动 Python 策略 API,每根 bar 仍进入
  Python `next()`;纯批处理或更少 Python callback 的框架吞吐口径不可直接等价比较。
- 若测试 `trade_on_close=True`,Engine 与 Lite 也必须保持语义一致;吞吐 runner 默认不打开该选项,
  以便和 Backtrader 默认 next-bar 口径比较。

### Lint / 格式化

```bash
uv run ruff check tradelearn/
uv run ruff format tradelearn/
uv run interrogate tradelearn/ --fail-under 90    # docstring 覆盖率
```

### Rust

```bash
cd backtest-rs/
cargo test                       # Rust 侧单元测试
cargo bench                      # 性能基准
maturin develop --release        # 构建并安装到 tradelearn/backtest/
```

### 文档

```bash
uv run mkdocs serve              # 本地预览 http://127.0.0.1:8000
uv run mkdocs build --strict     # 构建(warning 即失败)
```

## 金标测试

### 构建金标基线(阶段 0 Week 2)

```bash
python scripts/build_golden.py --version 1.x
# 用 reference/tradelearn_1x/ 跑 10 策略 × 10 数据集
# 写入 tests/golden/expected/v1.0/*.json
```

**产出冻结**:`tests/golden/expected/v1.0/` 不再改动(除非明确 MIGRATION 声明)。

### 运行一致性对照

```bash
uv run pytest tests/consistency/ -v
# 期望:
#   - metrics:rtol=1e-10
#   - ta.*:对 pandas-ta-classic rtol=1e-10
#   - ta.tdx.*:对 MyTT rtol=1e-10
#   - trades 时间/方向 0 差异
#   - equity rtol=1e-6
#   - stats rtol=1e-4
```

失败时定位:

```bash
uv run pytest tests/consistency/test_sma_cross.py::test_trades_identical -v --tb=long
```

## 启动 trade-learn Lab(用户视角)

用户装完后:

```bash
pip install trade-learn[lab]
tradelearn new my_research
cd my_research
tradelearn lab
```

### 环境变量

```bash
export MLFLOW_TRACKING_URI=https://mlflow.leafquant.com   # 默认远程
export TRADELEARN_DATA_CACHE_DIR=./data                    # 默认项目内
export TRADELEARN_LOG_LEVEL=INFO                           # DEBUG/INFO/WARN/ERROR
```

### 诊断

```bash
tradelearn doctor
```

输出:

```
  Python:     3.11.8 ✅
  Rust core:  loaded ✅
  MLflow:     https://mlflow.leafquant.com ✅ reachable
  MCP:        ready ✅
  Data cache: 0 symbols, 0 MB
  ⚠️  LLM API key not set (Jupyter AI disabled)
```

### MLflow 连接验证

```bash
python -c "import mlflow; mlflow.set_tracking_uri('https://mlflow.leafquant.com'); print(mlflow.search_experiments())"
```

## CI 本地预跑

```bash
# 提 PR 前本地跑一遍 CI 等价流程
make ci-local
# 或手动:
uv run ruff check && \
uv run ruff format --check && \
uv run interrogate --fail-under 90 && \
uv run pytest tests/unit/ tests/consistency/ && \
uv run mkdocs build --strict
```

## 构建与发版

### 构建 wheel

```bash
uv build                         # 产物在 dist/
# 含 Rust 核的跨平台 wheel 需 cibuildwheel(CI 里做)
```

### 发版流程

```bash
git tag v0.1.0
git push --tags
# CI 自动:
#   - 跑全量测试
#   - 构建所有平台 wheel
#   - 发 PyPI
#   - mike deploy 文档
#   - GitHub Release
```

**永不手动 pip upload**。

## 异常排查

### `tradelearn lab` 端口被占用

```bash
tradelearn lab --port 8889
# 或找出并杀进程
lsof -i :8888
```

### MLflow 连不上

```bash
tradelearn doctor                           # 看诊断
curl -I https://mlflow.leafquant.com         # 手工测连通
```

连不上时:Analyzer 会 warn 并跳过上报,**不中断回测**。

### 数据缓存问题

```bash
tradelearn data list                        # 看现有缓存
tradelearn data clear --symbol GOOG         # 清单个
tradelearn data clear                       # 清全部
```

### Rust 核 import 失败

```bash
# 确认 .so 存在
ls tradelearn/backtest/_core*.so

# 重建
cd backtest-rs/
maturin develop --release --force
```

### 金标测试飘

- 首先查 `docs/MIGRATION.md` 的 Known Differences
- 若无记录,这是 bug,**不要改金标,改代码**

## 工作纪律

### 契约先行

任何模块的新 API 先写/更新 `docs/specs/`,再写代码。**Spec 不定,代码不写**。

### 金标不过,PR 不合

CI 的 `tests/consistency/` 是 required check。PR 红灯不许 merge。

### Clean-Room 冷冻期(阶段 2 Week 5 结束后 2 周)

不打开 backtesting.py 源码。违反了等于作废这两周的设计笔记,重来。

### 指标口径变更

任何 metrics / indicator 的算法变动,必须:
1. 更新 `docs/specs/METRICS_SPEC.md` 或 `INDICATORS_SPEC.md`
2. 更新金标测试
3. 在 `MIGRATION.md` 记录"为什么变、变了什么、对用户影响"

### 1.x reference 不动

`reference/tradelearn_1x/` 整个重构期只读。想删改请走 PR + 详细理由。

## 关键文档引用

- [PROJECT.md](./PROJECT.md) — 愿景 / 路线图 / 决策 / 项目结构
- [PROGRESS.md](./PROGRESS.md) — 当前进度
- `docs/specs/` — 9 份设计规格(阶段 0 产出)
- `docs/internal/` — Clean-Room 设计笔记(阶段 2 Week 5 产出)
