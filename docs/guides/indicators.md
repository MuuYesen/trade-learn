# 指标指南

本页描述 2026-11-12 未发行源码候选。公共规则沿[框架指标规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#indicator-sources-streaming-20261111)和[事件执行规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#stream-first-event-backtest-20261111)；源码测试、目标平台构建、已发行和生产策略验收分别记录，不能互相替代。

## 来源与实际范围

默认公式来源为 TA-Lib，使用明确命名空间。根 `tradelearn.ta` 已退出，`tradelearn.indicators` 保来源导航与公共类型，不再包含重复的 `ta.sma/ema/...` 快捷公式。PTA 已退役，直接导入旧模块会给出迁移错误，没有隐藏 pandas-ta 后端。

| 来源 | 固定盘点范围 | 当前源码公开 | 限制 |
|---|---|---|---|
| `tradelearn.indicators.talib` | TA-Lib 0.8.3，237 项 | 233 项及大小写别名 | DPO/FRACTAL/ICHIMOKU 的显示移位及 ZIGZAG 修订语义未合格，不导出；内部 NaN/Inf 输入拒绝 |
| `tradelearn.indicators.tv` | pynesys-pynecore 6.13.3，67 项 ta 算子 | 66 项及大小写别名 | rci 的 NaN 反例不结束，未合格；旧 adx/bbands/ichimoku 兼容组合不再公开 |
| `tradelearn.indicators.tdx` | MyTT 固定提交 71 个函数，基础函数与组合分列 | 67 项：39 个基础算子、28 个组合 | 对照真实固定 MyTT 批量源码，修正旧 TL 简化公式/暖机/舍入/单位；CONST/RET/TOPRANGE/LOWRANGE 不合格 |
| KunQuant | 0.1.12 实包：Alpha101 注册82项，Alpha158默认配置158个特征；没有Alpha191 | macOS arm64 定向修正产物通过真实23项原生资格；内部显式产物接口 | 按用户最新决定直接采用所提供的101/158口径，不适配旧TL排名，不自建或更名Alpha191；平台、编译配置、表达式及产物SHA必须匹配 |

完整逐项输入、参数、输出、暖机、拒绝原因在源码 [capabilities.json](../../tradelearn/indicators/capabilities.json)。上游版本、提交、文件摘要及 Alpha101 缺项见[本轮固定来源记录](../../../../../docs/参考资料/paseo-session-records/20261112-two-goals/TL2/upstream-source-inventory.json)。公共支持按已测 profile 解释，不代表所有参数组合、全部上游函数、Linux 或现役策略已验收。上游更多能力继续逐项适配，TDX 的 CONST 使用未来最终值，RET 是非序列标量访问器，TOPRANGE/LOWRANGE 的全历史搜索尚无合格有界状态，均不公开。TDX 输入条件限定非缺失布尔值，REF/DIFF 拒绝负数未来引用；批量空输入统一返回空结果，规范化上游 DMA/CROSS/XSII 的空输入缺陷。

TA-Lib 使用 `TA-Lib==0.8.3`，TV 使用 `pynesys-pynecore==6.13.3`，安装配置由项目依赖入口维护。0.6.8 的数组末值 stream 接口被明确拒绝。旧 PyneCore 的 function_isolation 不可继续使用，当前接入固定 6.13.3 的 instance_state 状态向量私有接口；升级必须重新验收。

## 批量与逐 Bar

```python
from tradelearn.indicators import talib, tv, tdx

# Pandas 输入保索引；多输入必须严格对齐。
ma = talib.SMA(close, timeperiod=20)
macd = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
# MACD 列顺序：macd / signal / hist。

# 每个策略/feed/标的/周期/参数配置单独持有状态。
indicator = talib.EMA.configured(timeperiod=20)
state = indicator.new_state()
for bar in closed_bars:
    value = indicator.on_bar(bar, state=state)
indicator.reset(state=state)
```

同一个指标可被 Engine/Lite 的共享 Lines 调用：

```python
from tradelearn.indicators import talib

# Engine __init__ 或 Lite init 内：
self.ma = talib.EMA(self.data.close, timeperiod=20)
# next 内读取 self.ma[0]；读取不推进，多个输出共享一次状态推进。
```

Engine 不再导出 TL 专属 `bt.tv/bt.tdx/bt.pta`；显式导入来源模块后，Engine/Lite 共用相同适配器和事实。动态参数 Line 不作为固定参数接受，跨 feed 输入须先由调用方按决策时点明确对齐；不自动等待、填充或混入过旧资产。

批量路径直接调用官方 TA-Lib 数组函数或一次推进 PyneCore 状态向量；流式路径持续保留原生句柄/状态向量或 TDX 有界算子，不重新计算历史。TA-Lib 仅在有限暖机期构建种子数组，开句柄后释放种子；不将递归 EMA 状态截断到行情窗口。官方特殊种子超出声明上限会拒绝，不增长无界历史。

TA-Lib 前导联合缺值可保持暖机不可用，开始有效输入后的非有限输入在批量与流式均拒绝；不进入上游 C 未定义的内部 NaN 行为。整数输出的官方暖机零标记保留，同时由 `minimum_period` 区分尚未暖机，不把该零当有效信号。TV/TDX 沿各自缺失规则。TA-Lib 活动句柄会由上游拒绝全局配置修改，TL 还检查种子期配置未变化；不同策略不能用全局设置悄悄改变既有状态。

## TV 的明确输入

源码签名和能力清单列出数据参数与公式参数。普通 source 算子输入 close；`correlation/cross` 等使用 close、source2；OHLCV 算子只取声明的必要列。参数名沿当前原生来源：例如 MACD 为 `fastlen/slowlen/siglen`，DMI 为 `diLength/adxSmoothing`，SuperTrend 为 `factor/atrPeriod`。旧适配调用须显式迁移。

```python
ema = tv.ema(close, length=20)
bands = tv.bb(close, length=20, mult=2.0)  # basis / upper / lower
trend = tv.dmi(high, low, close, diLength=14, adxSmoothing=14)
# plus_di / minus_di / adx
vwap = tv.vwap(source, volume, anchor)
vwap_bands = tv.vwap(source, volume, anchor, stdev_mult=2.0)
# anchor 是同索引非缺失 bool 序列；True 开始新段。
pivots = tv.pivot_point_levels(open, high, low, close, anchor, type="Traditional")
```

CCI/MFI 延续 TL 的 HLC3 输入约定。`pivothigh/pivotlow` 只在右侧确认 Bar 到达时输出，不回填过去的策略可见值。锚定 VWAP 支持显式 source/volume/anchor 与可选标准差带，不推断会话或用零补缺失量价。

## 自定义协议与生命周期

保留函数式及类式自定义指标，不要求所有用户写类，也不承诺任意 lambda 自动编译。

```python
from tradelearn.indicators import FunctionIndicator

# 有限输入窗口 + 用户持久状态；每个读取只复用已有结果。
double = FunctionIndicator(
    "double", lambda close: close * 2,
    bar_columns=("close",), history_capacity=1, warmup=1,
    stream_func=lambda buffers, state, params: buffers["close"][-1] * 2,
)
```

更复杂的实现可传 `stream_factory(params)`，返回有 `update(*scalars)` 的持久实例。无显式流式实现的 FunctionIndicator 保留普通 batch compute，但 `on_bar` 报错；它不属于合格实时指标。调用者显式声明 `bar_columns`、参数、`warmup`（整数或参数函数）及有限工作历史，负责自定义后端自身状态容量和依赖。TL 不能静态证明任意 Python 内部没有隐藏批量调用。

共享 Lines 通过 `bind_stream(args, kwargs)` 获取隔离配置与有序 `inputs`，每个接纳事件调用一次 `binding.update(*scalars)`，其余读取 `binding.value`。`binding.indicator.minimum_period/output_names` 对应实际绑定参数。参数变化、跨对象误用状态、后端异常后继续推进均拒绝；异常状态须 reset 后按来源恢复资料重新暖机。已返回的输出不因后续更新而修改。

`indicators.streaming.StreamGraph` 是独立事件适配器，不创建交易循环：定义节点后按拓扑更新，稳定 sequence 是投递身份，timestamp 不能用于去重。完全相同的最后事件重复读取复用结果；修订/乱序拒绝，后端失败后整图拒绝继续。相同时间的不同合法 sequence 可分别推进。输入接收方应保持原事件 sequence，不能给重放事件重新编号。ready 同时检查依赖暖机与有限结果。自定义节点只有提供 snapshot/restore 才支持图快照；身份/参数/拓扑不兼容会拒绝。TA-Lib 句柄不可便携序列化，重启走 reset/重放暖机，不声称落盘句柄可直接恢复。

## KunQuant 提供的 Alpha101 / Alpha158

按本轮用户最新决定，直接使用固定 `KunQuant==0.1.12` 提供的 `predefined.Alpha101` 和 `predefined.Alpha158`，不为旧 TL 近似公式另写替代。Alpha101 实际注册82项，不是全部101项；Alpha158按配置生成特征，当前默认采用9个KBar、4个当期价格比值及145个滚动特征，共158项。两者独立命名，Alpha158不是Alpha191的别名。该实包没有Alpha191，不把已有TL pandas Alpha191称为KunQuant自带或已取得此原生资格。

`_kunquant_graph.build_alpha101(names)` 与 `build_alpha158(config=None)` 构造真实上游 Function，并核来源文件摘要、实际所需输入、输出与表达式摘要。它们不在事件内编译，不接收任意lambda。Alpha158窗口必须为因果整数，滚动窗口大于零，拒绝负窗口未来引用。每种编译layout须重新构造Function，因为上游优化会修改图对象。源码图定义不自动取得数值资格。

完整图需要来源明确的 `open/close/high/low/volume/amount`；VWAP沿所提供公式从amount/volume构造，不能把缺失成交额暗中填为close×volume。选择不需要成交额的合格子图时，实际图输入会裁剪；不为凑特征数量伪造市场输入。Alpha158配置纳入表达式摘要，不能用一个配置的证据外推所有配置。

内部 `_kunquant.Artifact` 记录表达式/库摘要、固定源码/运行时、OS/架构、编译器、ISA、精度、SIMD宽度、输入输出、真实资产顺序及保守暖机。`KunQuantBatch.compute(...)` 一次执行TS矩阵，`KunQuantStream.update(...)` 持续推进对应STREAM状态。两者共同遵循所编译模块的逐输出不可靠前缀声明，前缀不可用，不合成零信号；这是所提供后端的保守可用性边界，不冒充最早数学可用Bar，也不等同旧TL pandas暖机。

截面要求完整真实集合、固定顺序、同一决策时点；不补假资产凑SIMD，不混入过旧或未来输入。`sequence` 表示稳定投递身份，同时间不同sequence可分别推进，完全相同的身份和输入只复用结果，修订/乱序拒绝。输出显式copy并与内部去重缓存隔离。构造流状态必须显式传入 `identity=(strategy, feed, timeframe)` 三个非空字符串。快照v4保原生bytes和最后事件身份/缓存，恢复核身份及整个产物指纹；输出仍需查看各自可用性，数学无定义的源结果不补零。

编译只在研究/部署准备阶段执行。`_kunquant_compile.compiler_config` 仅补KunQuant0.1.12拆分源码缺失的状态声明头，不改库或公式；`prepare_causal_extrema` 只对固定源图的argmin/argmax调用上游 `WindowedReduce.decompose`，规避长窗口位置的局部时间索引错误；其余算子保留原生路径。不全局启用 `no_skip_list`，因为Alpha158分位数依赖skip-list。每个TS/STREAM模块使用独立库，避免上游拆分源码全局符号冲突。当前macOS运行库要求14.0，不能把本机产物称为旧macOS、Linux或Windows已验。默认轻量安装不启动编译器，运行按实际产物和依赖加载。

真实原生测试覆盖源批量/流式、常量/并列、真实资产顺序、输出所有权、重复事件和兼容/不兼容恢复。整个82/158输出集合仍以本轮RESULT的实际证据为准，不能用图构造成功、几个基础算子或编译成功代替。旧TL Alpha001的min并列排名与所提供KunQuant的average排名不同；用户已选择所提供口径，应显式迁移策略/模型，现役产物不自动切换。

## TDX 固定来源迁移

固定来源为 `mpquant/MyTT@7cd36ae13cae56657284badc8d5a6b7b8ed62a37`，保留原源文件摘要供测试复核。组合公式复用该固定源代码，仅将白名单基础算子接到持久实例；不接受用户传入任意函数。滚动算子只保声明窗口，EMA/SMA/DMA 保递归状态，依赖暖机从公式传播。多输入严格对齐，FILTER 等上游可变函数使用输入副本，不修改调用者行情。

旧 TL 内部 MyTT 近似公式仍可用于迁移比较，不是公共后端。新来源 MA 等需完整窗口；MACD/RSI/WR/BIAS/BOLL/PSY 沿上游舍入；OBV 保上游 `/10000` 单位；KTN 使用典型价中线与两倍 ATR；DMI 的 ADXR、EMV、MASS、ASI、XSII 等复用真实公式。不能沿旧结果直接延续策略阈值或宣称经济行为不变。重启按相同来源/参数重新暖机，现役策略须独立验收。

## PTA 与快捷层迁移

本次是未发行源码破坏性迁移，现役策略不自动切换。调用方应逐项复核种子、暖机、NaN、单位及列顺序。

| 原调用 | 显式迁移方向 |
|---|---|
| `ta.sma(..., period=p)` | `talib.SMA(..., timeperiod=p)` |
| `ta.ema/rsi/atr(..., length=p)` | 对应 `talib.EMA/RSI/ATR(..., timeperiod=p)`，须重新核种子与缺失 |
| `ta.macd(..., fast=f, slow=s, signal=g)` | `talib.MACD(..., fastperiod=f, slowperiod=s, signalperiod=g)`；旧快捷列 macd/hist/signal 改为 macd/signal/hist |
| `ta.bbands(..., length=p, std=d)` | `talib.BBANDS(..., timeperiod=p, nbdevup=d, nbdevdn=d)`；输出 upperband/middleband/lowerband |
| 旧三列 `ta.adx` | 分别调用 ADX/PLUS_DI/MINUS_DI，显式组装所需结果 |
| 会话 `ta.vwap` | 明确 source/volume/anchor 后迁 TV VWAP；不自动替换成其他同名公式 |

## 可执行验证与未验层

从源码根目录、安装固定依赖后执行：

```bash
python -m pytest tests/unit/indicators -q
```

套件包含完整固定 TA-Lib 范围的参考批量/前缀/常量/前导 NaN/非默认参数检查，TV 批流/隔离/缺失/重放，固定 MyTT 的 67 项普通/NaN/常量/逐前缀与空输入对照，及旧 TL 公式迁移反例，显式锚定与因果枢轴，自定义事件图、无 PTA 导入和真实 Engine/Lite 接入。代表性 9 个指标各推进 50,000 次，比较 10,000 与 50,000 次工作状态占用并禁止调用 batch；不据此宣传整库恒定时间、全库内存或端到端交易性能。

2026-11-12 定向修正后的六个 TS/STREAM 原生库已编译，真实23项数值/常量/并列/恢复/身份/容量测试通过；收据见[原生运行记录](../../../../../docs/参考资料/paseo-session-records/20261112-two-goals/TL2/NATIVE-CAUSAL-V2-RECEIPT.json)。这覆盖本次固定 macOS arm64、float64、8个真实合成资产及所记录编译配置，不能外推其他平台或所有参数。旧 skip-list 长窗口位置差异及全局 no_skip_list 分位数编译失败保留为历史反例，不复用其旧二进制。

最终整合、wheel安装和平台限制见[整合记录](../../../../../docs/参考资料/paseo-session-records/20261112-two-goals/TL-INTEGRATION/STATUS.json)；四个明确排除的 MyTT 函数、Linux/Windows原生图、所有参数组合、公开发布或生产自然数据仍未获资格。

固定 PyneCore 6.13.3 要求 Python 3.11+；Python 3.10 的基础安装不提供 TV 指标，不能以 TA/TDX 或批量回退冒充。CI 在 3.10 显式列出不可用的 TV 案例，仍执行 TA/TDX 容量与基础核心；3.11+ 缺少已声明 PyneCore 则直接失败。此范围不等同所有 Python/平台已通过。
