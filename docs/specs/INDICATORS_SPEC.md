# INDICATORS SPEC

`tradelearn.indicators`(对外 `tradelearn.ta`)模块规格——本土 + 国际双轨指标体系。

## 1. 三层命名空间

| 命名空间 | 来源 | 对齐基准 | 场景 |
|---|---|---|---|
| `ta.*` | pandas-ta-classic | pandas-ta-classic 原版 | 通用基座,跨市场通用 |
| `ta.tdx.*` | [MyTT](https://github.com/mpquant/MyTT)(MIT) | 通达信软件 | 内盘(A 股) |
| `ta.tv.*` | pyneCore 后端 | TradingView 图表 | 外盘(美股 / 加密) |

**同名指标允许存在,口径不同**——用户显式选择。

## 2. 模块布局

```
tradelearn/indicators/
├── __init__.py              # 暴露 ta 命名空间
├── base.py                  # Indicator 基类协议
├── core/                    # ta.* (pandas-ta-classic 薄封装)
│   ├── momentum.py
│   ├── overlap.py
│   ├── volatility.py
│   ├── volume.py
│   └── trend.py
├── tdx/                     # ta.tdx.*
│   ├── __init__.py          # 暴露 tdx30 全部 + 额外 A 股指标
│   ├── tdx30.py             # 通达信 30 个经典指标(MyTT 源)
│   ├── kdj.py / dmi.py / wr.py / cci_tdx.py / ...
│   └── mytt_ref/            # MyTT 源码的 Clean-Room 封装
└── tv/                      # ta.tv.*
    ├── __init__.py
    ├── supertrend.py / ichimoku.py / vwap_session.py / ...
    └── pyne_adapter.py      # pyneCore 的接入层
```

## 3. Python import 路径

```python
from tradelearn import ta

# 通用
ta.sma(close, period=20)
ta.rsi(close, length=14)
ta.macd(close)                    # 返回 DataFrame(macd, signal, hist)
ta.bbands(close, length=20)       # (lower, mid, upper)

# 内盘(通达信)
ta.tdx.macd(close)                # 通达信口径,DIF/DEA 命名
ta.tdx.kdj(high, low, close)
ta.tdx.ma(close, 5)

# 外盘(TradingView)
ta.tv.macd(close)                 # Pine Script 口径
ta.tv.supertrend(high, low, close, 10, 3)
ta.tv.ichimoku(high, low, close)
```

## 4. Indicator 基类(base.py)

```python
from typing import Protocol
import pandas as pd

class Indicator(Protocol):
    """所有指标统一协议。"""

    name: str
    params: dict

    def compute(self, *args, **kwargs) -> pd.Series | pd.DataFrame:
        """批量模式:输入整段行情,输出整段指标序列。"""

    def on_bar(self, bar) -> float | tuple:
        """流式模式(可选):单根 bar 增量更新。未实现则报错。"""
```

**要求**:
- 所有函数式指标(`ta.sma` 等)**同时也是 Indicator 实例**,通过 `ta.sma` 即可调用也可 `ta.sma.on_bar(...)`
- 或提供两套:函数用于批量,类用于流式(TBD,阶段 2 确认)

## 5. ta.* 实现约定

### 5.1 直接封装 pandas-ta-classic

```python
# tradelearn/indicators/core/momentum.py
import pandas_ta_classic as pta

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (generic).

    Different from ta.tdx.rsi (Tongdaxin formula) and
    ta.tv.rsi (TradingView formula). See docs/concepts/indicator-namespaces.md.
    """
    return pta.rsi(close, length=length)
```

**不重写算法**,只封装 + 中文 docstring + 口径差异说明。

### 5.2 docstring 必须警示差异

每个有对应 tdx / tv 版本的指标,docstring 必须写明:

```
NOTE: 数值与 ta.tdx.rsi / ta.tv.rsi 可能有差异。
按市场选择对应命名空间。
```

## 6. ta.tdx.* 实现约定

### 6.1 算法源:MyTT

[MyTT](https://github.com/mpquant/MyTT) 是通达信公式 Python 复刻,MIT 协议,可以**直接参考代码**融合进项目。

### 6.2 融合而非 pip 依赖

将 MyTT 的核心函数融入 `tradelearn/indicators/tdx/tdx30.py`:

```python
# tradelearn/indicators/tdx/tdx30.py

def MA(close, N):
    """通达信 MA(N日均线)。"""
    return close.rolling(N).mean()

def EMA(close, N):
    """通达信 EMA(2/(N+1) 加权)。"""
    return close.ewm(span=N, adjust=False).mean()

def MACD(close, SHORT=12, LONG=26, MID=9):
    """通达信 MACD:返回 DIF / DEA / MACD 三列。"""
    dif = EMA(close, SHORT) - EMA(close, LONG)
    dea = EMA(dif, MID)
    macd = (dif - dea) * 2
    return pd.DataFrame({'DIF': dif, 'DEA': dea, 'MACD': macd})
```

**变量名、函数名保留通达信习惯**(大写 / DIF / DEA),这是 A 股用户的肌肉记忆。

### 6.3 tdx30 完整清单(来源 MyTT)

30 个经典指标,按类别:

| 类别 | 指标 |
|---|---|
| 均线 | MA / EMA / SMA / WMA / EXPMA |
| 趋势 | MACD / DMA / TRIX / DMI |
| 摆动 | KDJ / RSI / WR / CCI / BIAS |
| 波动 | BOLL / ATR |
| 能量 | OBV / VR / PSY |
| 压力支撑 | SAR |
| 量价 | MFI / VOSC |
| ...(补齐至 30) | |

完整清单在阶段 2 Week 2 按 MyTT 实际函数列出。

### 6.4 命名空间暴露

```python
# tradelearn/indicators/tdx/__init__.py
from .tdx30 import *          # MA/EMA/MACD/KDJ/...
from . import tdx30           # 也允许 ta.tdx.tdx30.MA

# 额外 A 股指标(不在 tdx30 里的)
from .kdj_j import kdj_j
from .northern_flow import northern_flow
```

用户可通过:
- `ta.tdx.MACD(...)` — 直接调用
- `ta.tdx30.MACD(...)` — 通过子命名空间访问 tdx30

**两者等价**,`ta.tdx30` 是 `ta.tdx` 的子集别名(历史兼容 + 语义明确)。

## 7. ta.tv.* 实现约定

### 7.1 后端:pyneCore

pyneCore 复刻 TradingView Pine Script 生态,作为 `ta.tv.*` 的算法引擎。

### 7.2 封装模式

```python
# tradelearn/indicators/tv/supertrend.py
from pynecore import ta as pyne_ta

def supertrend(high, low, close, length=10, multiplier=3):
    """TradingView 风格 Supertrend。"""
    return pyne_ta.supertrend(high, low, close, length, multiplier)
```

**注意**:pyneCore 目前生态较新,阶段 0 的 PoC 要验证:
- 能否覆盖 20 个常用 TV 指标(RSI / MACD / BB / Supertrend / Ichimoku / VWAP / ADX / ...)
- 流式增量计算是否正常
- 维护活跃度

若 PoC 不达标,降级方案:**用 pandas-ta-classic 薄封装**,数值对齐 TV 尽力而为,不引入 pyneCore 依赖。

## 8. 金标对照

### 8.1 容忍度

| 命名空间 | 对照源 | rtol |
|---|---|---|
| `ta.*` | pandas-ta-classic 原版 | `1e-10` |
| `ta.tdx.*` | MyTT 原版 | `1e-10` |
| `ta.tdx.*`(抽样) | 通达信软件导出(手工抽查 3–5 指标) | `1e-6` |
| `ta.tv.*` | pyneCore / TradingView 截图 | `1e-6` |

### 8.2 金标数据集

```
tests/golden/indicators/
├── core/
│   ├── goog_rsi_14.parquet         # pandas-ta-classic 跑出
│   └── ...
├── tdx/
│   ├── 000001_macd.parquet         # MyTT 跑出
│   ├── 000001_macd_tdx_screenshot.csv  # 通达信软件手工导出
│   └── ...
└── tv/
    ├── AAPL_supertrend.parquet     # pyneCore 跑出
    └── ...
```

### 8.3 测试示例

```python
# tests/consistency/indicators/test_tdx_macd.py
def test_tdx_macd_matches_mytt():
    bars = load_golden("tdx/000001_daily")
    ours = ta.tdx.MACD(bars.close)
    theirs = mytt.MACD(bars.close)   # MyTT 原版
    assert np.allclose(ours, theirs, rtol=1e-10)

def test_tdx_macd_matches_tongdaxin_screenshot():
    bars = load_golden("tdx/000001_daily")
    ours = ta.tdx.MACD(bars.close)
    expected = load_golden("tdx/000001_macd_tdx_screenshot")
    assert np.allclose(ours['DIF'].iloc[-1], expected['DIF'].iloc[-1], rtol=1e-6)
```

## 9. 指标可组合

```python
class MyStrategy(Strategy):
    def __init__(self):
        self.sma20 = ta.sma(self.data.close, length=20)
        self.rsi_of_sma = ta.rsi(self.sma20, length=14)    # 指标套指标
```

内部实现:`ta.rsi` 接受 `Series | Indicator`,统一处理。

## 10. 可选依赖

```toml
# 通用 + 内盘 → 核心依赖
pandas-ta-classic = "*"
pynecore = "*"
# MyTT 不 pip 依赖,直接融合源码(MIT,合规)
```

## 11. 文档要求

每个指标的 docstring 必须含:

1. 一句话描述
2. Parameters / Returns
3. 与其它命名空间同名指标的差异说明(若有)
4. 一个可跑的 Example
5. `See Also` 跨引用

## 12. 不做的事

- ❌ 自造指标库(信号库、策略模板库)
- ❌ 自动选择"最合适"的命名空间(ambiguous,用户显式)
- ❌ 自动向量化 → Pine Script 转换
- ❌ 在 `ta.*` 下挂用户自定义指标(用户直接用函数即可)
