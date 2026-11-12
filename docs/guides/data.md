# 数据指南

> **tvkit实际候选资格（2026-11-12，完整迁移未通过）**：固定官方0.18.1已实际安装/导入，内部有限调用适配`data._tvkit`已验证同步/Notebook显式async、UTC/日期端点、超时/取消释放及每实例凭据隔离，不读取浏览器或TVKIT环境凭据。实际API仅支持splits/dividends，不能保现有`adjust="none"`；标的信息/搜索/锚点回放亦缺对等公开API，上游缺volume默认0也不能证明原始量完整。故当前默认Provider仍保原Transport，旧Charting消费者和websocket-client未退休；这不是迁移完成或新实盘数据循环。准确目标/许可边界见[唯一规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#tradingview-tvkit-candidate-20261111)。

trade-learn 策略层统一面对 OHLCV DataFrame 或 `MultiIndex(timestamp, symbol)` panel。数据层的目标是：provider 取到的数据可以直接进入 Lite / Engine，不需要用户手工拆成多份 DataFrame。

## 标准 Bars 形态

| 形态 | 索引 | 列 |
|---|---|---|
| 单标的 | `DatetimeIndex` 或可转换为时间索引 | `open`, `high`, `low`, `close`, `volume` |
| 多标的 | `MultiIndex(timestamp, symbol)` | `open`, `high`, `low`, `close`, `volume` |

列名会 normalize 成小写；额外列也会保留为小写 line。

```python
self.data.factor[0]
data.get_array("factor")
```

不推荐在策略里使用 `self.data.FACTOR`。

## 单标的读取

```python
from tradelearn.data import TradingViewProvider

provider = TradingViewProvider()
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")
```

Engine:

```python
import tradelearn.engine as bt

cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=bars.xs("NASDAQ:AAPL", level="symbol")), name="NASDAQ:AAPL")
```

Lite:

```python
import tradelearn.lite as tl

stats = tl.Backtest(bars, Strategy).run()
```

## 多标的读取

```python
symbols = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"]
bars = provider.history_ohlc(symbols, start="2023-01-01", end="2024-01-01")
```

多标的返回值是一个 panel：

```text
index: timestamp, symbol
columns: open, high, low, close, volume
```

Engine使用原生Feed对象，按身份显式装配：

```python
cerebro = bt.Cerebro()
for symbol in bars.index.get_level_values("symbol").unique():
    cerebro.adddata(bt.feeds.PandasData(dataname=bars.xs(symbol, level="symbol")), name=symbol)
```

Lite 也可以直接接收 panel：

```python
stats = tl.Backtest(bars, PortfolioStrategy).run()
```

策略里可以按 ticker 访问：

```python
class PortfolioStrategy(tl.Strategy):
    def next(self):
        aapl = self.datas["NASDAQ:AAPL"]
        msft = self.datas["NASDAQ:MSFT"]
        if aapl.close[0] > msft.close[0]:
            self.target_weights({"NASDAQ:AAPL": 0.8, "cash": 0.2})
```

## Symbol 对应规则

### TradingView

TradingView 数据建议使用 `EXCHANGE:SYMBOL`：

```python
provider.history_ohlc("NASDAQ:AAPL", start="2024-01-01")
provider.history_ohlc(["NASDAQ:AAPL", "NASDAQ:MSFT"], start="2024-01-01")
```

输出的 `symbol` 层保持 TradingView 形式，例如 `NASDAQ:AAPL`。

如果调用时传入裸 symbol，需要同时提供 `exchange`：

```python
provider.history_ohlc("AAPL", exchange="NASDAQ", start="2024-01-01")
```

### TDX

Provider 的频率区分大小写：`1m` 表示一分钟，`1M` 表示月线。
TDX 与 TradingView 均接受月线别名 `M` / `monthly`，返回的 `attrs["freq"]` 统一为 `1M`。
旧版 TDX 将 `1m` 错误映射为月线；依赖该行为的调用应改用 `1M`。

TDX 输出会归一为 canonical symbol：

```python
provider.history_ohlc("SZ:000001", start="2024-01-01")
provider.history_ohlc("SH:600000", start="2024-01-01")
```

裸 6 位代码会按规则推断交易所；如果无法确定，框架会要求用户显式写 `SH:` 或 `SZ:`，避免把同一个股票映射到错误市场。

```python
# 推荐
"SZ:000001"
"SH:600000"

# 不推荐: 如果存在歧义会抛错
"000001"
```

## 共同有界窗口

Lite通过`self.window(size=20, data=self.datas, ago=0)`取多源面板，默认不传data取单源时间索引OHLCV；Engine通过独立`tradelearn.backtest.history.window(self, size=20, data=self.datas)`接研究组件。共享实现按每源已交付cursor和策略共同决策时点截断后，再应用ago和size，按旧到新返回。

`ago=-1`排除当前可见Bar，未来偏移、非正/非整数size、字段缺失、重复时间索引明确拒绝；历史不足返回现有行，不填值/补Bar/联网。源时区保持，跨naive/aware时钟拒绝。单源`attrs['symbol']`为真实feed名，FeatureSet转换研究标准面板时保留身份并转UTC。修改快照不影响feed。直接传feed集合时共同截止取首feed当前已交付时间，策略调用应传self以使用真实决策时钟。未启动首源返回空窗口。

旧`build_history_panel(datas, lookback=None, *, fields=..., size=None, ago=0)`仅作兼容转换，也使用共同提取；不在Engine新增history_panel/window策略方法。研究不提供DataFrame访问器或公开FeatureDataset。

## DuckDB文件查询

```python
from tradelearn.data import open_dataset
with open_dataset("/authorized/bars/**/*.parquet") as dataset:
    bars = dataset.read(symbols=["NASDAQ:AAPL"], start="2026-01-01", end="2026-06-01")
    close = dataset.read(columns=["close"])
    explorer = dataset.profile(symbols=["NASDAQ:AAPL"])
    explorer.report("/tmp/data.xlsx")
```

Dataset直接查询同schema的本地CSV或Parquet文件，不复制为第二数据库。文件必须有timestamp、symbol和OHLCV；timestamp须带显式时区，返回UTC有序唯一面板，边界start/end均包含。筛选在SQL源端执行，OHLCV验证后投影字段；无匹配行保标准空索引。重复键、缺列、空身份或时区不明拒绝，不自动修复。可用glob或显式路径列表，不支持混合CSV/Parquet；跨文件schema差异拒绝。attrs记录源文件路径、大小/mtime和查询条件，mtime不是内容密码学摘要。`dataset.iter_batches(batch_size=65536, ...)`按相同有序查询输出有界批次，跨批重复键也拒绝；完整消费或关闭迭代器后再复用同一连接。Dataset每次读取源文件，不暗用旧缓存；close/context manager释放连接。

`BarsCache.read(..., symbols=..., start=..., end=..., columns=...)`复用同一DuckDB文件查询，仅读取缓存已覆盖范围，不联网补数；无筛选保持原Parquet读取。BarsCache仍负责已获取Bars的可恢复Parquet与TTL元数据，DuckDBBarsBackend负责显式数据库表。三个职责不合并成隐式副本或自动补数。Provider继续同步返回Bars；tvkit版本、平台及数据许可未合格前保持候选，不默认读取浏览器Cookie。

## 数据源扩展

自定义 provider 只需要返回标准 Bars：

```python
class MyProvider:
    def history_ohlc(self, symbol, *, start=None, end=None, freq="1d"):
        ...
        return bars
```

只要返回的是单标的 OHLCV DataFrame 或多标的 `MultiIndex(timestamp, symbol)` panel，就可以直接接入 Engine / Lite。


DataExplorer 的报告表格快照按数据/索引/列类型/来源属性与内部计算版本自动失效；修改返回表不会污染缓存，重复 HTML/Excel 不重做统计。Tushare 默认 HTTP 客户端仅在显式请求时导入 requests；数据 facade 和自定义 transport 不因此要求 requests。缺依赖给出 research 安装提示，凭据及自定义 transport 错误继续脱敏。

固定宽度Rust重采样从真实秒标签重建索引时显式使用纳秒dtype，避免Pandas 2/3默认表示差异；输入s/ms/us/ns按声明单位转换，真实标签和时区不变，不因此声称原数据有纳秒测量精度。
