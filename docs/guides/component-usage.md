# 组件调用速查

本页对应未发布源码候选。Engine以固定Backtrader源码为对照；实际已支持和明确拒绝的能力见[Engine指南](engine.md)。Lite共用Runtime、Lines、指标状态与订单事实，保留便利外观。

实时使用`from tradelearn_runtime import NautilusStore`，沿`getdata/getbroker`组装Cerebro，通常`engine.run()`；额外监控参数使用`store.run(engine, monitor=...)`。只有一个官方NA宿主，配置与能力边界见[执行接入指南](live.md)。

## Lite / Engine 总览

| 能力 | Lite | Engine |
|---|---|---|
| 导入 | `import tradelearn.lite as tl` | `import tradelearn.engine as bt` |
| 策略基类 | `tl.Strategy` | `bt.Strategy` |
| 初始化 | `init()`、类属性参数 | `__init__()`、`params` |
| 数据 | `self.data.close[0]`、`self.datas["AAPL"]` | `self.data.close[0]`、`getdatabyname()` |
| 来源指标 | `tradelearn.indicators.talib/tv/tdx` | 同一来源对象；`bt.talib`保BT入口 |
| 自定义指标 | `class MyInd(tl.Indicator)` | `class MyInd(bt.Indicator)`；同一基类 |
| 组合调仓 | `target_weights(weights)` | 逐标的`order_target_percent()`或独立`backtest.targets.target_weights` |
| 结果 | `stats = backtest.run()` | `[strategy] = cerebro.run(); stats = strategy.stats` |
| 报告 | `Reporter(stats).report(path)` | 同一独立Reporter |

## 数据输入

Lite直接接单标的OHLCV、symbol映射或`MultiIndex(timestamp, symbol)`面板。Engine显式注册Feed：

```python
cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=bars), name="AAPL")
# 多标的面板由调用方按symbol拆分，每个Feed单独登记。
for symbol, frame in panel.groupby(level="symbol"):
    cerebro.adddata(bt.feeds.PandasData(
        dataname=frame.droplevel("symbol")), name=str(symbol))
```

## 暖机与窗口

在初始化中建立指标依赖。共享Runtime按各依赖minimum_period调用prenext、nextstart、next。Engine多Feed恰处最小周期边界时可连续调用nextstart，符合固定BT；Lite首次可运行回调只调一次nextstart。Lite可用`addminperiod(20)`显式要求20条数据；旧`start_on_bar(n)`对应n+1条。

Lite的`self.window(20)`返回当前已交付前缀的独立快照，单feed为时间索引，多feed为timestamp/symbol面板。Engine窗口使用独立`tradelearn.backtest.history.window(strategy, size=20)`，不扩展BT策略基类。正ago、未来数据和无效窗口参数拒绝；详细语义见[Lite指南](lite.md)。

## 下单与持仓

| 目标 | Lite | Engine |
|---|---|---|
| 买入/卖出 | `self.buy(size=100)` / `self.sell(size=100)` | 同名BT参数 |
| 平仓 | `self.position().close()` / `self.close()` | `self.close()` |
| 当前持仓 | `self.position(ticker=None)` | `self.position` / `self.getposition()` |
| 组合目标权重 | `self.target_weights({"AAPL": 0.5})` | 独立`tradelearn.backtest.targets.target_weights(self, weights)` |

position只表达已成交状态；Lite.orders是当前活动订单tuple。调仓会考虑活动根订单剩余量，关联止盈止损单不重复当目标仓位。

## 共享指标

```python
from tradelearn.indicators import talib, tv, tdx

# 在Engine.__init__或Lite.init中定义；实际资格见指标指南。
self.ma20 = talib.SMA(self.data.close, timeperiod=20)
self.rsi14 = talib.RSI(self.data.close, timeperiod=14)
self.tv_rsi = tv.rsi(self.data.close, length=14)
self.mixed = (self.data.high + self.data.low) / 2
```

每个调用创建独立持久状态；多输出共用一次事件更新。任意Pandas函数不自动编译成流式指标；旧I/PTA路径已退出。自定义类声明共享Lines或显式实现持久后端，不能读取未来全量数据。

## Stats 与报告

两入口共享`summary/equity/returns/fills/trades/positions/orders/config`；Lite附加`records/strategy`，底层结果可用`stats.stats`读取。

完整已平仓交易摘要使用含费用`pnlcomm`，并在metric_metadata显式标记；无亏损分母的profit_factor不可计算，保NaN。Reporter仅沿生产者显式基准消费，保原gross列。

摘要`annual_return`、`volatility`使用小数比例；`max_drawdown`是带负号的小数比例。不可计算的Sharpe/Sortino/Calmar等保NaN，不补零。底层`Stats.metric_metadata`记录单位、缺失原因和年化参数。当前摘要使用252的逐Bar默认profile，不能据此宣称输入已满足交易日日历；其他频率应在独立`Reporter(stats, periods=...)`明确提供年化参数。价格与收益样本须由调用方先核资格。

```python
from tradelearn.report import Reporter
report = Reporter(stats, periods=252)
facts = report.metric_facts()  # 原始值、单位、原因
report.report("report.html")
```

## Research 组件

```python
import tradelearn.research as research
import tradelearn.research.preprocess as pp
import tradelearn.research.portfolio as pf

feature_set = research.FeatureSet({"alpha": lambda p: p.close.pct_change(20)})
features = feature_set.fit_transform(bars).dropna()

pipe = research.Pipeline(
    [
        pp.Winsorizer(columns=["alpha"]),
        pp.StandardScaler(columns=["alpha"]),
    ]
)
features = pipe.fit_transform(features)

weights = pf.Allocator(
    select=pf.TopK(k=50),
    weight=pf.EqualWeight(gross=0.95),
    constrain=pf.Constraints(max_weight=0.05, normalize=True),
).build(scores)
```

Research 组件遵循 sklearn-like 心智：需要训练集状态的组件实现 `fit()` / `transform()` / `fit_transform()`；只做无状态排序或权重计算的组件可以保留为轻量类或函数。
