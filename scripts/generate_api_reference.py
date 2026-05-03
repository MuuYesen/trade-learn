"""Generate mkdocstrings API Reference pages for the documentation site."""

from __future__ import annotations

import argparse
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApiReferenceModule:
    """One public API module rendered in the reference page."""

    title: str
    import_path: str
    summary: str
    common_entries: tuple[str, ...]


API_REFERENCE_MODULES: tuple[ApiReferenceModule, ...] = (
    ApiReferenceModule(
        "Lite",
        "tradelearn.lite",
        "Tradelearn 1.x 风格轻量 API。",
        ("Backtest", "Strategy", "talib", "pta", "tdx", "tv"),
    ),
    ApiReferenceModule(
        "Engine",
        "tradelearn.engine",
        "Backtrader 风格高级事件驱动 API。",
        ("Cerebro", "Strategy", "Order", "Analyzer", "Sizer"),
    ),
    ApiReferenceModule(
        "数据",
        "tradelearn.data",
        "K 线数据、provider、缓存与重采样工具。",
        ("DataProvider", "CacheProvider", "resample_frame"),
    ),
    ApiReferenceModule(
        "指标",
        "tradelearn.indicators",
        "技术指标 facade,包含 pandas-ta-classic / TDX / TradingView 生态入口。",
        ("sma", "ema", "rsi", "macd", "pta", "talib", "tdx", "tv"),
    ),
    ApiReferenceModule(
        "收益风险指标",
        "tradelearn.metrics",
        "收益、风险、回撤与因子评价指标。",
        ("returns", "sharpe", "max_drawdown", "alpha_beta"),
    ),
    ApiReferenceModule(
        "因子",
        "tradelearn.factor",
        "Alpha 公式与因子分析工具。",
        ("FactorAnalyzer", "alphas"),
    ),
    ApiReferenceModule(
        "报告",
        "tradelearn.report",
        "HTML、Excel 与研究报告导出。",
        ("Report", "TearSheet", "export_excel"),
    ),
    ApiReferenceModule(
        "研究",
        "tradelearn.research",
        "指数增强研发流程记录、预处理与组合构建工具。",
        ("ResearchRun", "ResearchResult", "preprocess", "portfolio"),
    ),
    ApiReferenceModule(
        "机器学习",
        "tradelearn.ml",
        "机器学习、模型注册与特征筛选。",
        ("AutoML", "CausalSelector", "ModelRegistry", "ModelLoader"),
    ),
)


@dataclass(frozen=True)
class ApiDocTarget:
    """One callable rendered in a generated guide page."""

    heading: str
    obj: Callable[..., Any]
    summary: str
    params: dict[str, str]
    returns: str | None = None
    examples: tuple[str, ...] = ()


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", None)
    if name is not None:
        return name
    text = str(annotation)
    return text.replace("typing.", "")


def _default_value(parameter: inspect.Parameter) -> str:
    if parameter.default is inspect.Parameter.empty:
        return "required"
    return repr(parameter.default)


def _table_code(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return f"`{escaped}`"


def _signature_text(obj: Callable[..., Any]) -> str:
    return str(inspect.signature(obj))


def _parameter_rows(
    obj: Callable[..., Any],
    descriptions: dict[str, str],
) -> list[str]:
    parameters = [
        item for item in inspect.signature(obj).parameters.items() if item[0] not in {"self", "cls"}
    ]
    if not parameters:
        return ["参数说明：无。"]
    rows = ["| 参数 | 类型 | 默认值 | 说明 |", "|---|---|---|---|"]
    for name, parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            display = f"*{name}"
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            display = f"**{name}"
        else:
            display = name
        rows.append(
            "| "
            f"`{display}` | "
            f"{_table_code(_annotation_name(parameter.annotation))} | "
            f"{_table_code(_default_value(parameter))} | "
            f"{descriptions.get(name, '')} |"
        )
    return rows


def _render_target(target: ApiDocTarget) -> list[str]:
    parts = [
        f"## `{target.heading}`",
        "",
        target.summary,
        "",
        "```python",
        f"{target.heading}{_signature_text(target.obj)}",
        "```",
        "",
        "参数说明：",
        "",
        *_parameter_rows(target.obj, target.params),
    ]
    if target.returns:
        parts.extend(["", f"返回值：{target.returns}"])
    for example in target.examples:
        parts.extend(["", "```python", example.rstrip(), "```"])
    parts.append("")
    return parts


def _render_guide(
    title: str,
    source_module: str,
    intro: str,
    targets: tuple[ApiDocTarget, ...],
) -> str:
    parts = [
        f"# {title}",
        "",
        intro,
        "",
        "本页偏查询:用于核对函数签名、参数名和返回值。第一次写策略请先看 "
        "[Lite 指南](lite.md)、[Engine 指南](engine.md) 或 [策略编写指南](strategy.md)。",
        "",
        f"本页由 `scripts/generate_api_reference.py` 从 `{source_module}` 运行时代码签名生成。",
        "",
    ]
    for target in targets:
        parts.extend(_render_target(target))
    return "\n".join(parts).rstrip() + "\n"


def _public_symbol_names(module: ApiReferenceModule) -> tuple[str, ...]:
    try:
        imported = importlib.import_module(module.import_path)
    except Exception:
        return module.common_entries
    exported = getattr(imported, "__all__", None)
    if exported:
        return tuple(str(name) for name in exported)
    return module.common_entries


def _public_symbols_text(module: ApiReferenceModule, *, limit: int = 10) -> str:
    names = _public_symbol_names(module)
    if not names:
        return ""
    shown = names[:limit]
    text = ", ".join(f"`{name}`" for name in shown)
    remaining = len(names) - len(shown)
    if remaining > 0:
        text += f", ... (+{remaining})"
    return text


def _member_signature(member: Any) -> str:
    if isinstance(member, property):
        return ""
    try:
        return str(inspect.signature(member))
    except (TypeError, ValueError):
        return ""


def _member_summary(member: Any) -> str:
    doc = inspect.getdoc(member.fget if isinstance(member, property) else member)
    if not doc:
        return ""
    return doc.splitlines()[0]


def _render_member_index(title: str, cls: type[Any], names: tuple[str, ...]) -> list[str]:
    rows = [
        f"### {title}",
        "",
        "| 成员 | 类型 | 签名 | 说明 |",
        "|---|---|---|---|",
    ]
    for name in names:
        member = inspect.getattr_static(cls, name, None)
        if member is None:
            continue
        kind = "属性" if isinstance(member, property) else "方法"
        signature = _member_signature(getattr(cls, name, member))
        rows.append(
            "| "
            f"`{cls.__name__}.{name}` | "
            f"{kind} | "
            f"`{signature}` | "
            f"{_member_summary(member)} |"
        )
    rows.append("")
    return rows


def render_engine_api_guide() -> str:
    """Render the Engine guide from public facade signatures."""
    from tradelearn.engine.cerebro import Cerebro
    from tradelearn.engine.strategy import Strategy

    targets = (
        ApiDocTarget(
            "Cerebro.__init__",
            Cerebro.__init__,
            "创建 Backtrader 风格的高级事件驱动回测运行器。",
            {
                "match_mode": (
                    "`exact` 对齐 Backtrader bar 级撮合; "
                    "`smart` 使用趋势路径推演和止损优先悲观撮合。"
                ),
                "callback_batch": "保留参数。正式事件驱动路径仍逐 bar callback。",
                "trade_on_close": "是否在当前 bar close 成交; 默认下一根 bar open 成交。",
                "exactbars": "Backtrader 兼容保留参数。",
                "stdstats": "是否挂载默认 observer。",
                "slippage": "slippage 配置对象。",
                "commission": "commission 配置对象。",
                "mode": "`backtest` / `paper` / `live`; 当前主路径是 backtest。",
                "kwargs": "保留给 Backtrader facade 的额外配置。",
            },
            returns="`Cerebro` 实例。",
            examples=(
                "import tradelearn.engine as bt\n\ncerebro = bt.Cerebro(match_mode='exact')",
            ),
        ),
        ApiDocTarget(
            "Cerebro.adddata",
            Cerebro.adddata,
            "添加一个 DataFrame 或 DataFeed 到运行器。",
            {
                "data": "包含 `open/high/low/close/volume` 的 DataFrame,或已经构造好的 DataFeed。",
                "name": "数据名称; 后续可通过 `datasbyname` 或 `getdatabyname()` 查询。",
            },
            returns="添加后的 data feed。",
        ),
        ApiDocTarget(
            "Cerebro.addstrategy",
            Cerebro.addstrategy,
            "注册策略类及策略参数。",
            {
                "strategy": "`bt.Strategy` 子类。",
                "args": "传给策略构造的 positional 参数。",
                "kwargs": "传给策略参数系统的 keyword 参数。",
            },
            returns="`None`。",
        ),
        ApiDocTarget(
            "Cerebro.run",
            Cerebro.run,
            "执行回测并返回策略实例列表。",
            {},
            returns="`list[Strategy]`。",
            examples=("results = cerebro.run()\nstrategy = results[0]",),
        ),
        ApiDocTarget(
            "Strategy.buy",
            Strategy.buy,
            "提交买入订单。",
            {
                "data": "目标数据 feed; 默认主数据。",
                "size": "订单数量; `None` 时使用 sizer。",
                "price": "Limit/Stop 价格。",
                "exectype": "`Order.Market` / `Order.Limit` / `Order.Stop` 等。",
                "kwargs": "订单元数据,如 `parent`、`oco`、`transmit`、`info`。",
            },
            returns="`Order`。",
        ),
        ApiDocTarget(
            "Strategy.sell",
            Strategy.sell,
            "提交卖出订单。",
            {
                "data": "目标数据 feed; 默认主数据。",
                "size": "订单数量; `None` 时使用 sizer。",
                "price": "Limit/Stop 价格。",
                "exectype": "`Order.Market` / `Order.Limit` / `Order.Stop` 等。",
                "kwargs": "订单元数据,如 `parent`、`oco`、`transmit`、`info`。",
            },
            returns="`Order`。",
        ),
        ApiDocTarget(
            "Strategy.buy_bracket",
            Strategy.buy_bracket,
            "提交买入 bracket 订单,生成 main / stop / limit 三联订单。",
            {
                "data": "目标数据 feed; 默认主数据。",
                "size": "订单数量。",
                "price": "主订单价格。",
                "plimit": "主订单 limit price 别名。",
                "exectype": "主订单执行类型。",
                "valid": "订单有效期,当前作为元数据保留。",
                "trailamount": "Trailing stop 金额参数。",
                "trailpercent": "Trailing stop 百分比参数。",
                "oargs": "主订单额外参数。",
                "stopprice": "止损订单触发价格。",
                "stopexec": "止损订单执行类型。",
                "stopargs": "止损订单额外参数。",
                "limitprice": "止盈订单价格。",
                "limitexec": "止盈订单执行类型。",
                "limitargs": "止盈订单额外参数。",
                "kwargs": "透传给底层订单的额外参数。",
            },
            returns="`list[Order]`: `[main, stop, limit]`。",
        ),
        ApiDocTarget(
            "Strategy.order_target_percent",
            Strategy.order_target_percent,
            "按账户权益比例调整目标持仓。",
            {
                "data": "目标数据 feed; 默认主数据。",
                "target": "目标权益比例,如 `0.5` 表示 50%。",
                "kwargs": "透传给 `buy` / `sell` / `close`。",
            },
            returns="`Order | None`。",
        ),
    )
    guide = _render_guide(
        "Engine API 签名",
        "tradelearn.engine",
        (
            "`tradelearn.engine` 是 Backtrader 风格高级入口。"
            "本文档的签名来自运行时代码,参数说明由生成器元数据补充。"
        ),
        targets,
    )
    parts = [
        guide.rstrip(),
        "",
        "## Engine 完整接口",
        "",
        "下表从运行时代码自动抽取,用于补足指南未逐段展开的常用接口。",
        "",
        *_render_member_index(
            "Cerebro",
            Cerebro,
            (
                "setcash",
                "getbroker",
                "setbroker",
                "setcommission",
                "set_coc",
                "adddata",
                "chaindata",
                "rolloverdata",
                "resampledata",
                "replaydata",
                "addstrategy",
                "optstrategy",
                "addanalyzer",
                "addobserver",
                "addwriter",
                "getwriterheaders",
                "getwriterinfo",
                "getwritervalues",
                "addstore",
                "addtimer",
                "addcalendar",
                "addsizer",
                "setsizer",
                "add_signal",
                "signal_strategy",
                "signal_concurrent",
                "signal_accumulate",
                "runstop",
                "plot",
                "run",
            ),
        ),
        *_render_member_index(
            "Strategy",
            Strategy,
            (
                "datetime",
                "position",
                "start",
                "init",
                "prenext",
                "next",
                "stop",
                "notify_order",
                "notify_trade",
                "notify_cashvalue",
                "getposition",
                "getdatabyname",
                "getpositionbyname",
                "setsizer",
                "getsizer",
                "getsizing",
                "submit_order",
                "buy",
                "sell",
                "close",
                "cancel",
                "order_target_size",
                "order_target_value",
                "order_target_percent",
                "buy_bracket",
                "sell_bracket",
                "addminperiod",
            ),
        ),
    ]
    return "\n".join(parts).rstrip() + "\n"


def render_lite_api_guide() -> str:
    """Render the Lite guide from public facade signatures."""
    from tradelearn.lite.backtest import Backtest
    from tradelearn.lite.data import LiteDataProxy
    from tradelearn.lite.position import PositionProxy
    from tradelearn.lite.strategy import Strategy

    targets = (
        ApiDocTarget(
            "Backtest.__init__",
            Backtest.__init__,
            "创建 Lite 回测运行器。Lite 是 Tradelearn 1.x 风格入口,不是 backtesting.py 兼容层。",
            {
                "data": "单资产 DataFrame 或 `{ticker: DataFrame}` 多资产输入。",
                "strategy": "`tradelearn.lite.Strategy` 子类。",
                "cash": "初始资金。",
                "commission": "手续费比例。",
                "margin": "保证金配置保留参数。",
                "trade_on_close": "是否在当前 bar close 成交。",
                "hedging": "保留参数。",
                "exclusive_orders": "保留参数。",
                "holding": "初始持仓保留参数。",
                "trade_start_date": "开始交易日期保留参数。",
                "lot_size": "最小交易单位保留参数。",
                "fail_fast": "保留参数。",
                "stats_mode": "`full` 返回完整 orders/fills/trades/equity; `lazy` 只计算轻量 summary。",
                "storage": "策略可读写的共享存储。",
                "match_mode": "撮合模式; 默认 `exact` 以复用 Engine/Backtrader 对齐路径。",
            },
            returns="`Backtest` 实例。",
        ),
        ApiDocTarget(
            "Backtest.run",
            Backtest.run,
            "执行回测并返回核心统计。",
            {
                "kwargs": "策略参数覆盖值; 参数必须先作为策略类属性声明。",
            },
            returns=(
                "`LiteStats`,支持 `stats['final_value']` 等 summary key,"
                "并提供 `summary`、`equity`、`returns`、`fills`、`trades`、"
                "`positions`、`orders`、`records`、`strategy`、`config`。"
            ),
            examples=("stats = Backtest(data, MyStrategy).run(fast=10, slow=30)",),
        ),
        ApiDocTarget(
            "Backtest.optimize",
            Backtest.optimize,
            "执行简单 grid search,按 `Return [%]` 选择最优结果。",
            {
                "kwargs": "参数网格,如 `fast=range(5, 20, 5)`。",
            },
            returns="最佳参数组合对应的 `pd.Series`。",
        ),
        ApiDocTarget(
            "Strategy.I",
            Strategy.I,
            "声明指标。callable 会批量预计算; Series/array-like 会包装为渐进揭示指标。",
            {
                "funcval": "指标函数、Series、DataFrame 或 array-like。",
                "args": "传给指标函数的位置参数。",
                "name": "指标名称。",
                "plot": "绘图元数据。",
                "overlay": "绘图元数据。",
                "color": "绘图元数据。",
                "scatter": "绘图元数据。",
                "kwargs": "传给指标函数,或保存到指标 attrs。",
            },
            returns="`IndicatorProxy` 或 `IndicatorBundle`。",
            examples=(
                "def zscore(close, window=20):\n"
                "    series = close.to_series() if hasattr(close, 'to_series') else close\n"
                "    return (series - series.rolling(window).mean()) / series.rolling(window).std()\n\n"
                "self.z = self.I(zscore, self.data.close, window=20)",
            ),
        ),
        ApiDocTarget(
            "Strategy.position",
            Strategy.position,
            "返回当前 ticker 的 Lite 持仓代理。",
            {
                "ticker": "目标 ticker; `None` 表示主数据。",
            },
            returns="`PositionProxy`。",
        ),
        ApiDocTarget(
            "Strategy.buy",
            Strategy.buy,
            "提交 Lite 买入订单。",
            {
                "ticker": "目标 ticker; `None` 表示主数据。",
                "size": "订单数量; 百分比仓位请使用 `order_target_percent()` 或 `target_percent()`。",
                "limit": "限价价格。",
                "stop": "Stop 触发价格。",
                "sl": "止损价格; 有 `sl` 或 `tp` 时走 bracket。",
                "tp": "止盈价格; 有 `sl` 或 `tp` 时走 bracket。",
                "tag": "写入订单 `info['tag']` 的业务标签。",
            },
            returns="`Order` 或 bracket `list[Order]`。",
        ),
        ApiDocTarget(
            "Strategy.sell",
            Strategy.sell,
            "提交 Lite 卖出订单。",
            {
                "ticker": "目标 ticker; `None` 表示主数据。",
                "size": "订单数量; 百分比仓位请使用 `order_target_percent()` 或 `target_percent()`。",
                "limit": "限价价格。",
                "stop": "Stop 触发价格。",
                "sl": "止损价格; 有 `sl` 或 `tp` 时走 bracket。",
                "tp": "止盈价格; 有 `sl` 或 `tp` 时走 bracket。",
                "tag": "写入订单 `info['tag']` 的业务标签。",
            },
            returns="`Order` 或 bracket `list[Order]`。",
        ),
        ApiDocTarget(
            "Strategy.order_target_percent",
            Strategy.order_target_percent,
            "按账户权益比例调整目标持仓。",
            {
                "ticker": "目标 ticker; `None` 表示主数据。",
                "target": "目标权益比例,如 `0.5` 表示 50%。",
                "kwargs": "透传给底层 `buy` / `sell` / `close`。",
            },
            returns="`Order | None`。",
        ),
        ApiDocTarget(
            "Strategy.target_percent",
            Strategy.target_percent,
            "按 ticker 直接表达目标组合权重。",
            {
                "ticker": "目标 ticker。",
                "target": "目标权益比例,如 `0.5` 表示 50%。",
            },
            returns="`Order | None`。",
        ),
        ApiDocTarget(
            "Strategy.target_weights",
            Strategy.target_weights,
            "按一组 ticker 权重调整目标组合。`cash` 可作为保留键表达现金权重。",
            {
                "weights": "ticker 到目标权重的映射,或 pandas Series。",
                "close_missing": "是否把未出现在 weights 中的已知 ticker 调整为 0。",
            },
            returns="`list[Order]`。",
        ),
        ApiDocTarget(
            "Strategy.target_equal",
            Strategy.target_equal,
            "把总目标权重等分给一组 ticker。",
            {
                "tickers": "目标 ticker 列表。",
                "weight": "这组 ticker 合计占用的权益比例。",
                "close_missing": "是否把未出现在 tickers 中的已知 ticker 调整为 0。",
            },
            returns="`list[Order]`。",
        ),
        ApiDocTarget(
            "Strategy.close_all",
            Strategy.close_all,
            "清空 Lite 已知数据源对应的目标持仓。",
            {},
            returns="`list[Order]`。",
        ),
        ApiDocTarget(
            "Strategy.record",
            Strategy.record,
            "记录策略内部序列,结果写入 stats 的 `_records`。",
            {
                "name": "记录名称。",
                "plot": "绘图元数据。",
                "overlay": "绘图元数据。",
                "color": "绘图元数据。",
                "scatter": "绘图元数据。",
                "kwargs": "要记录的键值对。",
            },
            returns="`None`。",
        ),
    )
    guide = _render_guide(
        "Lite API 签名",
        "tradelearn.lite",
        (
            "`tradelearn.lite` 是 Tradelearn 1.x 风格轻量入口。"
            "本文档的签名来自运行时代码,参数说明由生成器元数据补充。"
        ),
        targets,
    )
    parts = [
        guide.rstrip(),
        "",
        "## Lite 完整接口",
        "",
        "下表从运行时代码自动抽取,用于检查 Lite 语法糖暴露了哪些入口。",
        "",
        *_render_member_index(
            "Backtest",
            Backtest,
            ("run", "optimize", "plot", "report"),
        ),
        *_render_member_index(
            "Strategy",
            Strategy,
            (
                "position",
                "I",
                "buy",
                "sell",
                "cancel",
                "order_target_size",
                "order_target_value",
                "order_target_percent",
                "target_percent",
                "target_weights",
                "target_equal",
                "close_all",
                "buy_bracket",
                "sell_bracket",
                "record",
                "equity",
                "storage",
                "orders",
                "trades",
                "closed_trades",
                "start_on_day",
                "start_on_bar",
                "prepare_data",
            ),
        ),
        *_render_member_index(
            "LiteDataProxy",
            LiteDataProxy,
            ("df", "index", "now", "tickers", "the_ticker", "pip"),
        ),
        *_render_member_index(
            "PositionProxy",
            PositionProxy,
            ("size", "close", "pl", "pl_pct", "is_long", "is_short"),
        ),
    ]
    return "\n".join(parts).rstrip() + "\n"


def render_api_reference(modules: tuple[ApiReferenceModule, ...] = API_REFERENCE_MODULES) -> str:
    """Render the readable API reference index page."""
    parts = [
        "# API 参考",
        "",
        "本页由 `scripts/generate_api_reference.py` 自动生成。",
        "",
        "把这里当作 API 地图:需要示例和调用流程时看指南;需要完整类、函数、"
        "参数签名时看模块参考。",
        "",
        "## 先看这里",
        "",
        "| 目标 | 阅读 |",
        "|---|---|",
        "| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API 签名](../guides/lite-api.md) |",
        "| 编写 Backtrader 风格事件策略、Analyzer、Observer、Sizer | "
        "[Engine API 签名](../guides/engine-api.md) |",
        "| 从零编写策略并理解两种入口差异 | [策略编写指南](../guides/strategy.md) |",
        "| 查询精确类/函数签名和完整 docstring | 下方模块参考链接 |",
        "",
        "## 公开模块",
        "",
        "| 模块 | 用途 | 常用入口 | 完整参考 |",
        "|---|---|---|---|",
    ]
    for module in modules:
        slug = _module_slug(module)
        entries = _public_symbols_text(module)
        parts.append(
            "| "
            f"`{module.import_path}` | "
            f"{module.summary} | "
            f"{entries} | "
            f"[{module.title}](reference/{slug}.md) |"
        )
    parts.extend(
        [
            "",
            "## 按模块列出公开符号",
            "",
        ]
    )
    for module in modules:
        symbols = ", ".join(f"`{name}`" for name in _public_symbol_names(module))
        parts.extend(
            [
                f"### `{module.import_path}`",
                "",
                symbols or "_没有显式 `__all__`; 请查看完整参考。_",
                "",
            ]
        )
    parts.extend(
        [
            "",
            "## 自动生成页面",
            "",
            "- [Lite API 签名](../guides/lite-api.md)",
            "- [Engine API 签名](../guides/engine-api.md)",
            "- [策略编写指南](../guides/strategy.md)",
        ]
    )
    for module in modules:
        slug = _module_slug(module)
        parts.append(f"- [`{module.import_path}`](reference/{slug}.md)")
    return "\n".join(parts).rstrip() + "\n"


def render_strategy_writing_guide() -> str:
    """Render a practical guide for writing Engine and Lite strategies."""
    parts = [
        "# 策略编写指南",
        "",
        "本页由 `scripts/generate_api_reference.py` 自动生成,只说明两种策略入口的共同心智模型。",
        "完整函数签名请看 Lite / Engine API 签名页,完整工作流请看对应指南。",
        "",
        "Tradelearn 当前有两个用户入口:",
        "",
        (
            "- `tradelearn.lite`: Tradelearn 1.x 风格轻量 API,"
            "适合快速写单文件策略和研究原型。"
        ),
        (
            "- `tradelearn.engine`: Backtrader 风格高级 API,适合复杂事件策略、"
            "Analyzer、Observer、Sizer、多数据和参数优化。"
        ),
        "",
        (
            "两者共享同一套 `tradelearn.backtest` runtime 和 Rust 撮合内核。"
            "区别应该只在语法翻译层,不应该各自维护不同的业务逻辑。"
        ),
        "",
        "## 基本规则",
        "",
        "- `line[0]` 是当前 bar,`line[-1]` 是前一根 bar。",
        (
        "- 指标不下沉 Rust;使用真实 TA-Lib、pandas-ta-classic、TDX、TradingView "
            "等 Python 指标生态批量计算。"
        ),
        (
            "- Engine 是 Backtrader 数值对齐主入口;修改撮合、订单、broker、"
            "生命周期后必须跑 Backtrader 对齐测试。"
        ),
        (
            "- Lite 只验证语法层是否正确接入同一 runtime;"
            "底层正确性仍以 Engine/Backtrader 对齐为主。"
        ),
        "",
        "## Lite 策略",
        "",
        (
            "Lite 策略继承 `tradelearn.lite.Strategy`,"
            "通常在 `init()` 声明指标,在 `next()` 写交易逻辑。"
        ),
        "",
        "```python",
        "import pandas as pd",
        "import tradelearn.lite as tl",
        "",
        "",
        "class SmaCross(tl.Strategy):",
        "    fast = 10",
        "    slow = 30",
        "",
        "    def init(self):",
        "        self.fast_sma = tl.tdx.MA(self.data.close, N=self.fast)",
        "        self.slow_sma = tl.tdx.MA(self.data.close, N=self.slow)",
        "",
        "    def next(self):",
        "        if not self.position() and self.fast_sma[0] > self.slow_sma[0]:",
        "            self.buy(size=1)",
        "        elif self.position() and self.fast_sma[0] < self.slow_sma[0]:",
        "            self.position().close()",
        "",
        "",
        "data = pd.read_csv('bars.csv', parse_dates=True, index_col=0)",
        "stats = tl.Backtest(data, SmaCross, cash=100_000, match_mode='exact').run()",
        "print(stats['final_value'])",
        "```",
        "",
        "Lite 常用写法:",
        "",
        "| 需求 | 写法 |",
        "|---|---|",
        "| 当前价格 | `self.data.close[0]` |",
        "| 前一根价格 | `self.data.close[-1]` |",
        "| 内置指标 | `tl.tdx.MA(self.data.close, N=20)` |",
        "| 自定义函数指标 | `self.I(my_func, self.data.close, window=20)` |",
        "| 当前持仓 | `self.position()` |",
        "| 指定 ticker 持仓 | `self.position('BTCUSDT')` |",
        "| 买入/卖出 | `self.buy(size=...)` / `self.sell(size=...)` |",
        "| 权益比例调仓 | `self.order_target_percent(ticker='BTCUSDT', target=0.5)` |",
        "| 止损止盈 | `self.buy(sl=..., tp=...)` |",
        "| 记录序列 | `self.record(signal=value)` |",
        "| 当前权益 | `self.equity` |",
        "| 运行存储 | `self.storage` |",
        "",
        "Lite `run()` 返回 `LiteStats`:",
        "",
        "```python",
        "stats = tl.Backtest(data, SmaCross).run()",
        "stats['final_value']",
        "stats.summary",
        "stats.equity",
        "stats.trades",
        "stats.records",
        "stats.strategy",
        "stats.config",
        "```",
        "",
        "## Engine 策略",
        "",
        (
            "Engine 策略继承 `tradelearn.engine.Strategy`,"
            "通常在 `__init__()` 声明指标,在 `next()` 写交易逻辑。"
        ),
        "",
        "```python",
        "import pandas as pd",
        "import tradelearn.engine as bt",
        "",
        "",
        "class SmaCross(bt.Strategy):",
        "    params = (",
        "        ('fast', 10),",
        "        ('slow', 30),",
        "    )",
        "",
        "    def __init__(self):",
        "        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)",
        "        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)",
        "",
        "    def next(self):",
        "        if not self.position and self.fast[0] > self.slow[0]:",
        "            self.buy(size=1)",
        "        elif self.position and self.fast[0] < self.slow[0]:",
        "            self.close()",
        "",
        "",
        "data = pd.read_csv('bars.csv', parse_dates=True, index_col=0)",
        "cerebro = bt.Cerebro(match_mode='exact')",
        "cerebro.adddata(data, name='BTCUSDT')",
        "cerebro.addstrategy(SmaCross, fast=10, slow=30)",
        "cerebro.broker.setcash(100_000)",
        "strategy = cerebro.run()[0]",
        "print(strategy.broker.getvalue())",
        "```",
        "",
        "Engine 常用写法:",
        "",
        "| 需求 | 写法 |",
        "|---|---|",
        "| 当前价格 | `self.data.close[0]` |",
        "| 前一根价格 | `self.data.close[-1]` |",
        "| 当前持仓 | `self.position` 或 `self.getposition(data)` |",
        "| 内置指标 | `bt.tdx.MA(self.data.close, N=20)` |",
        "| 复杂自定义指标 | `class MyInd(bt.Indicator)` |",
        "| 买入/卖出 | `self.buy(size=...)` / `self.sell(size=...)` |",
        "| 平仓 | `self.close()` |",
        "| 目标仓位 | `self.order_target_size(...)` / `self.order_target_percent(...)` |",
        "| bracket | `self.buy_bracket(...)` / `self.sell_bracket(...)` |",
        "| 多数据查询 | `self.getdatabyname('BTCUSDT')` |",
        "| 订单通知 | `notify_order(self, order)` |",
        "| 交易通知 | `notify_trade(self, trade)` |",
        "",
        "## 指标写法",
        "",
        "Engine 内置指标使用 vendor 命名空间,复杂自定义指标使用 `bt.Indicator`:",
        "",
        "```python",
        "self.ma20 = bt.tdx.MA(self.data.close, N=20)",
        "self.rsi14 = bt.talib.RSI(self.data.close, timeperiod=14)",
        "```",
        "",
        "Lite 内置指标也直接使用 vendor 命名空间;只有自定义函数才需要 `self.I(...)`:",
        "",
        "```python",
        "self.ma20 = tl.tdx.MA(self.data.close, N=20)",
        "self.macd = tl.talib.MACD(self.data.close)",
        "self.custom = self.I(my_func, self.data.close, window=20)",
        "```",
        "",
        "TA-Lib / pandas-ta-classic / TDX / TradingView 指标都保留在 Python 生态中,不要写成 Rust 指标:",
        "",
        "```python",
        "self.talib_sma = tl.talib.SMA(self.data.close, timeperiod=20)",
        "self.pta_sma = tl.pta.SMA(self.data.close, length=20)",
        "self.tdx_ma = tl.tdx.MA(self.data.close, N=20)",
        "self.tv_rsi = tl.tv.RSI(self.data.close, length=14)",
        "```",
        "",
        "## 测试验收",
        "",
        "底层撮合和订单正确性以 Engine/Backtrader 对齐为主:",
        "",
        "```bash",
        "uv run python benchmarks/runners/benchmark_bt.py",
        "```",
        "",
        (
            "Lite 策略测试重点是语法层能否跑通并接入同一 runtime,"
            "例如 `self.I(...)`、`position()`、`data.close[0]`、"
            "`record()`、`sl/tp`。"
        ),
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def _module_slug(module: ApiReferenceModule) -> str:
    return module.import_path.rsplit(".", 1)[-1].replace("_", "-")


def render_module_reference(module: ApiReferenceModule) -> str:
    """Render one mkdocstrings-backed module reference page."""
    parts = [
        f"# {module.title} 参考",
        "",
        module.summary,
        "",
        "[返回 API 参考](../reference.md)",
        "",
        f"::: {module.import_path}",
        "    options:",
        "      show_source: false",
        "      show_root_heading: false",
        "      show_root_toc_entry: false",
        "      show_root_full_path: false",
        "      show_object_full_path: false",
        "      show_bases: false",
        "      show_docstring_parameters: true",
        "      show_docstring_returns: true",
        "      show_docstring_raises: true",
        "      show_signature_annotations: true",
        "      separate_signature: true",
        "      members_order: source",
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_api_reference_pages(
    docs_dir: Path | str = Path("docs"),
    modules: tuple[ApiReferenceModule, ...] = API_REFERENCE_MODULES,
) -> tuple[Path, ...]:
    """Write the readable index and per-module mkdocstrings reference pages."""
    docs_path = Path(docs_dir)
    outputs = [write_api_reference(docs_path)]
    reference_dir = docs_path / "api" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for module in modules:
        output = reference_dir / f"{_module_slug(module)}.md"
        output.write_text(render_module_reference(module), encoding="utf-8")
        outputs.append(output)
    return tuple(outputs)


def write_api_reference(docs_dir: Path | str = Path("docs")) -> Path:
    """Write the generated API Reference page under ``docs_dir/api/reference.md``."""
    output = Path(docs_dir) / "api" / "reference.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_api_reference(), encoding="utf-8")
    return output


def write_api_guides(docs_dir: Path | str = Path("docs")) -> tuple[Path, Path, Path]:
    """Write generated API guide pages under ``docs_dir/guides``."""
    guides_dir = Path(docs_dir) / "guides"
    guides_dir.mkdir(parents=True, exist_ok=True)
    engine_output = guides_dir / "engine-api.md"
    lite_output = guides_dir / "lite-api.md"
    strategy_output = guides_dir / "strategy.md"
    engine_output.write_text(render_engine_api_guide(), encoding="utf-8")
    lite_output.write_text(render_lite_api_guide(), encoding="utf-8")
    strategy_output.write_text(render_strategy_writing_guide(), encoding="utf-8")
    return engine_output, lite_output, strategy_output


def write_api_docs(docs_dir: Path | str = Path("docs")) -> tuple[Path, ...]:
    """Write every generated API documentation page."""
    reference_pages = write_api_reference_pages(docs_dir)
    guides = write_api_guides(docs_dir)
    return (*reference_pages, *guides)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for local documentation generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation root directory.",
    )
    args = parser.parse_args(argv)
    for output in write_api_docs(args.docs_dir):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
