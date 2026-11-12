# Lite

当前源码候选，未代表已发行或生产切换。Lite保留`Backtest.run`、`init`、类属性参数、`equity/record/window/target_weights/close_all`，使用与Engine相同的Runtime、Lines、Indicator和下单事实。

```python
from tradelearn import lite as tl
from tradelearn.indicators import talib

class Cross(tl.Strategy):
    period = 20
    def init(self):
        self.average = talib.SMA(self.data.close, timeperiod=self.period)
    def next(self):
        if not self.position() and self.data.close[0] > self.average[0]:
            self.buy(size=1, tag="entry")
        self.record(signal=self.average[0])

result = tl.Backtest(bars, Cross, cash=100_000).run(period=20)
```

`tl.Indicator is tradelearn.engine.Indicator`。简单表达式在init定义后随事件推进；复杂状态可以继承同一个Indicator。旧`I`任意向量/预计算包装及Lite独立IndicatorProxy已退出，现成来源指标直接调用；普通Pandas研究仍在Research层显式执行。init没有未来Bar；运行时只读已交付前缀。`addminperiod(n)`为共同暖机入口，旧start_on_bar(n)迁为addminperiod(n+1)。

`self.window(size=n)`返回当前feed时间索引OHLCV快照，attrs保symbol；`data=self.datas`返回(timestamp,symbol)面板。`ago=-1`排除当前Bar，拒绝正ago、非法size和未来数据。历史不足返回已有行；快照修改不影响feed，不补Bar/填值/联网。实现唯一委托[共同窗口](research.md)。

持仓`position(ticker=None).size/is_long/is_short/pl/pl_pct`只表示成交事实；未成交意图在共同订单状态中。`orders`为当前策略活动订单的tuple，含提交中、部分成交、撤单处理中与未知提交，终态才移除；历史用运行结果`result.orders`。没有自动重复订单锁；`target_weights`考虑在途量，重复同Bar相同目标不重复提交。`notify_order/notify_trade`可选，用户覆盖不妨碍独立观察链。

`buy/sell`的size为正有限数量，limit+stop表示StopLimit；sl/tp使用共同括号订单组，缺省的保护侧不会另造一单。`close(ticker=...,size=...)`和`close_all()`只提交本策略已成交持仓的平仓意图，不保证成交。`position().close(portion)`的portion必须在(0,1]。实时能力还须通过NA及具体venue校验，未知提交不得自动重发。

`target_equal`迁为Allocator等权后`target_weights`；`target_percent`使用`order_target_percent`；空实现prepare_data退出；history_panel迁window。storage仅运行内自定义状态，不默认承诺跨运行恢复。研究组件通过显式参数或bundle接入，不依赖基类research_result自动注入。

`Backtest` 当前仅支持默认 `margin=1/hedging=False/exclusive_orders=False/holding=None/trade_start_date=None/lot_size=1/fail_fast=True`；非默认值在构造时明确拒绝，不能接受后忽略。`trade_on_close`、`cash`、`commission`、`match_mode` 和 `stats_mode` 则实际传入共同运行时。

`optimize(**parameter_grid)` 在本进程顺序执行有限网格，每个 trial 使用新 Backtest、同一执行配置和独立空 storage，按 `return_pct` 返回最佳 LiteStats；不序列化原生运行时，不启动进程池。非空 storage 与空网格明确拒绝。普通 `run()` 的显式 storage 用法保留。

源码验收入口：`tests/unit/lite/test_engine_lite_replay.py`、`test_shared_runtime_contract.py`、`test_integration_backtest_contract.py`与`tests/test_lite_native_projection.py`。最后一项是投影fixture；官方NA真实离线回放另列，不把fixture当券商连接。
