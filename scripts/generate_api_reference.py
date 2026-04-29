"""Generate mkdocstrings API Reference pages for the documentation site."""

from __future__ import annotations

import argparse
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
        "Engine",
        "tradelearn.engine",
        "Backtrader 风格高级事件驱动 API。",
        ("Cerebro", "Strategy", "Order", "ind", "Analyzer", "Sizer"),
    ),
    ApiReferenceModule(
        "Lite",
        "tradelearn.lite",
        "Tradelearn 1.x 风格轻量 API。",
        ("Backtest", "Strategy", "Signal", "SignalStrategy"),
    ),
    ApiReferenceModule(
        "Data",
        "tradelearn.data",
        "K 线数据、provider、缓存与重采样工具。",
        ("DataProvider", "CacheProvider", "resample_frame"),
    ),
    ApiReferenceModule(
        "Indicators",
        "tradelearn.indicators",
        "技术指标 facade,包含 pandas-ta-classic / TDX / TradingView 生态入口。",
        ("sma", "ema", "rsi", "macd", "ta", "tdx", "tv"),
    ),
    ApiReferenceModule(
        "Metrics",
        "tradelearn.metrics",
        "收益、风险、回撤与因子评价指标。",
        ("returns", "sharpe", "max_drawdown", "alpha_beta"),
    ),
    ApiReferenceModule(
        "Factor",
        "tradelearn.factor",
        "Alpha 公式与因子分析工具。",
        ("FactorAnalyzer", "alphas"),
    ),
    ApiReferenceModule(
        "Report",
        "tradelearn.report",
        "HTML、Excel 与研究报告导出。",
        ("Report", "TearSheet", "export_excel"),
    ),
    ApiReferenceModule(
        "ML",
        "tradelearn.ml",
        "机器学习策略、特征存储、模型注册与特征筛选。",
        ("MLStrategy", "FeatureStore", "ModelRegistry", "FeatureSelector"),
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
    rows = ["| 参数 | 类型 | 默认值 | 说明 |", "|---|---|---|---|"]
    for name, parameter in inspect.signature(obj).parameters.items():
        if name in {"self", "cls"}:
            continue
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
        *_parameter_rows(target.obj, target.params),
    ]
    if target.returns:
        parts.extend(["", f"返回: {target.returns}"])
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
        f"Generated from `{source_module}` code signatures by `scripts/generate_api_reference.py`.",
        "",
    ]
    for target in targets:
        parts.extend(_render_target(target))
    return "\n".join(parts).rstrip() + "\n"


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
    return _render_guide(
        "Engine API",
        "tradelearn.engine",
        (
            "`tradelearn.engine` 是 Backtrader 风格高级入口。"
            "本文档的签名来自运行时代码,参数说明由生成器元数据补充。"
        ),
        targets,
    )


def render_lite_api_guide() -> str:
    """Render the Lite guide from public facade signatures."""
    from tradelearn.lite.backtest import Backtest
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
                "`pd.Series`,包含 `Equity Final [$]`、`Return [%]`、"
                "`# Trades`、`Win Rate [%]`、`_strategy`、`_records`。"
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
            examples=("self.sma = self.I(ta.sma, self.data.close, period=20)",),
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
                "size": "数量。`0 < size < 1` 表示按权益比例换算整数数量; 整数表示单位数量。",
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
                "size": "数量。`0 < size < 1` 表示按权益比例换算整数数量; 整数表示单位数量。",
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
    return _render_guide(
        "Lite API",
        "tradelearn.lite",
        (
            "`tradelearn.lite` 是 Tradelearn 1.x 风格轻量入口。"
            "本文档的签名来自运行时代码,参数说明由生成器元数据补充。"
        ),
        targets,
    )


def render_api_reference(modules: tuple[ApiReferenceModule, ...] = API_REFERENCE_MODULES) -> str:
    """Render the readable API reference index page."""
    parts = [
        "# API Reference",
        "",
        "本页由 `scripts/generate_api_reference.py` 自动生成。",
        "",
        "把这里当作 API 地图:需要示例和调用流程时看 Guide;需要完整类、函数、"
        "参数签名时看模块 Reference。",
        "",
        "## 先看这里",
        "",
        "| 目标 | 阅读 |",
        "|---|---|",
        "| 编写 Backtrader 风格事件策略、Analyzer、Observer、Sizer | "
        "[Engine API Guide](engine.md) |",
        "| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API Guide](lite.md) |",
        "| 查询精确类/函数签名和完整 docstring | 下方模块 Reference 链接 |",
        "",
        "## 公开模块",
        "",
        "| 模块 | 用途 | 常用入口 | 完整 Reference |",
        "|---|---|---|---|",
    ]
    for module in modules:
        slug = _module_slug(module)
        entries = ", ".join(f"`{entry}`" for entry in module.common_entries)
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
            "## Generated Pages",
            "",
            "- [Engine API Guide](engine.md)",
            "- [Lite API Guide](lite.md)",
        ]
    )
    for module in modules:
        slug = _module_slug(module)
        parts.append(f"- [`{module.import_path}`](reference/{slug}.md)")
    return "\n".join(parts).rstrip() + "\n"


def _module_slug(module: ApiReferenceModule) -> str:
    return module.import_path.rsplit(".", 1)[-1].replace("_", "-")


def render_module_reference(module: ApiReferenceModule) -> str:
    """Render one mkdocstrings-backed module reference page."""
    parts = [
        f"# {module.title} Reference",
        "",
        module.summary,
        "",
        "[Back to API Reference](../reference.md)",
        "",
        f"::: {module.import_path}",
        "    options:",
        "      show_source: true",
        "      show_root_heading: true",
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


def write_api_guides(docs_dir: Path | str = Path("docs")) -> tuple[Path, Path]:
    """Write generated Engine and Lite API guide pages under ``docs_dir/api``."""
    api_dir = Path(docs_dir) / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    engine_output = api_dir / "engine.md"
    lite_output = api_dir / "lite.md"
    engine_output.write_text(render_engine_api_guide(), encoding="utf-8")
    lite_output.write_text(render_lite_api_guide(), encoding="utf-8")
    return engine_output, lite_output


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
