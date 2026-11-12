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
        "Lightweight Tradelearn 1.x style API.",
        ("Backtest", "Strategy", "TargetOrderConstraints"),
    ),
    ApiReferenceModule(
        "Engine",
        "tradelearn.engine",
        "Backtrader-style advanced event-driven API.",
        ("Cerebro", "Strategy", "Order", "Analyzer", "Sizer"),
    ),
    ApiReferenceModule(
        "Data",
        "tradelearn.data",
        "OHLCV data providers, caching, and resampling utilities.",
        ("DataProvider", "CacheProvider", "resample_frame"),
    ),
    ApiReferenceModule(
        "Indicators",
        "tradelearn.indicators",
        "Qualified TA-Lib, TDX and TradingView indicators with stateful execution.",
        ("sma", "ema", "rsi", "macd", "adx", "atr", "bbands", "vwap"),
    ),
    ApiReferenceModule(
        "Metrics",
        "tradelearn.metrics",
        "Return, risk, drawdown, trade, and factor evaluation metrics.",
        ("simple_returns", "sharpe", "max_drawdown", "alpha", "beta"),
    ),
    ApiReferenceModule(
        "Factor",
        "tradelearn.factor",
        "Alpha formulas and factor analysis tools.",
        ("FactorAnalyzer", "alphas"),
    ),
    ApiReferenceModule(
        "Report",
        "tradelearn.report",
        "HTML, Excel, and research report export utilities.",
        ("Reporter", "ReportContext", "ReportSection"),
    ),
    ApiReferenceModule(
        "Research",
        "tradelearn.research",
        "Research workflow tracking, preprocessing, and portfolio construction tools.",
        ("ResearchRun", "ResearchResult", "preprocess", "portfolio"),
    ),
    ApiReferenceModule(
        "Search",
        "tradelearn.integrations.search",
        "Explicit parameter grid execution outside the Engine and core runtime.",
        ("grid_search", "GridSearchResult"),
    ),
    ApiReferenceModule(
        "MLflow",
        "tradelearn.integrations.mlflow",
        "Explicit post-run recording of completed results; native Run ownership.",
        ("log", "LogResult"),
    ),
    ApiReferenceModule(
        "Runtime",
        "tradelearn_runtime",
        "Independent official Nautilus execution host and Store integration.",
        ("NautilusStore",),
    ),
    ApiReferenceModule(
        "ML",
        "tradelearn.ml",
        "Machine learning, model registry, and feature selection utilities.",
        ("CausalSelector", "ModelRegistry", "ModelLoader"),
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
        "本页偏查询:用于核对函数签名、参数名和返回值。第一次写策略请先看 "
        "[Lite Guide](lite.md)、[Engine Guide](engine.md) 或 "
        "[Strategy Writing Guide](strategy.md)。",
        "",
        f"Generated from `{source_module}` code signatures by `scripts/generate_api_reference.py`.",
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
        "| Member | Kind | Signature | Summary |",
        "|---|---|---|---|",
    ]
    for name in names:
        member = inspect.getattr_static(cls, name, None)
        if member is None:
            continue
        kind = (
            "property"
            if isinstance(member, property)
            else "type"
            if inspect.isclass(member)
            else "method"
            if callable(member)
            else "attribute"
        )
        signature = _member_signature(getattr(cls, name, member))
        rows.append(
            f"| `{cls.__name__}.{name}` | {kind} | `{signature}` | {_member_summary(member)} |"
        )
    rows.append("")
    return rows


def _signature_targets(cls: type[Any], names: tuple[str, ...]) -> tuple[ApiDocTarget, ...]:
    """Describe existing callable members without fabricating removed aliases."""
    return tuple(
        ApiDocTarget(
            f"{cls.__name__}.{name}",
            getattr(cls, name),
            _member_summary(getattr(cls, name)),
            {},
        )
        for name in names
        if callable(getattr(cls, name, None))
    )


def render_engine_api_guide() -> str:
    """Render the strict Engine surface from current public signatures."""
    from tradelearn.engine import Cerebro, Strategy

    targets = (
        *_signature_targets(
            Cerebro,
            (
                "__init__",
                "adddata",
                "addstrategy",
                "addanalyzer",
                "addobserver",
                "optstrategy",
                "run",
                "plot",
            ),
        ),
        *_signature_targets(
            Strategy,
            (
                "__init__",
                "buy",
                "sell",
                "close",
                "cancel",
                "buy_bracket",
                "sell_bracket",
                "order_target_size",
                "order_target_value",
                "order_target_percent",
            ),
        ),
    )
    rendered = _render_guide(
        "Engine API",
        "tradelearn.engine",
        "Engine公开接口以Backtrader名称/位置/参数/返回/行为为边界；"
        "非BT报告、MLflow、运行时Store通过独立模块接入。实际语义资格见Engine指南。",
        targets,
    )
    parts = [rendered, "## Engine 完整接口", ""]
    for cls in (Cerebro, Strategy):
        names = tuple(name for name in dir(cls) if not name.startswith("_"))
        parts.extend(_render_member_index(cls.__name__, cls, names))
    return "\n".join(parts).rstrip() + "\n"


def render_lite_api_guide() -> str:
    """Render convenience members while excluding the retired I wrapper."""
    from tradelearn.lite.backtest import Backtest
    from tradelearn.lite.position import PositionProxy
    from tradelearn.lite.strategy import Strategy

    targets = (
        *_signature_targets(Backtest, ("__init__", "run", "optimize", "report", "log_mlflow")),
        *_signature_targets(
            Strategy,
            (
                "init",
                "window",
                "buy",
                "sell",
                "cancel",
                "position",
                "target_weights",
                "target_percent",
                "record",
            ),
        ),
    )
    rendered = _render_guide(
        "Lite API",
        "tradelearn.lite",
        "Lite保留init、类属性参数、window、target_weights、record/equity便利。"
        "与Engine共享计算与订单事实，不要求任意策略类原样互换。",
        targets,
    )
    parts = [rendered, "## Lite 完整接口", ""]
    for cls in (Backtest, Strategy, PositionProxy):
        names = tuple(name for name in dir(cls) if not name.startswith("_"))
        parts.extend(_render_member_index(cls.__name__, cls, names))
    return "\n".join(parts).rstrip() + "\n"


def render_api_reference(modules: tuple[ApiReferenceModule, ...] = API_REFERENCE_MODULES) -> str:
    """Render the readable API reference index page."""
    parts = [
        "# API 参考",
        "",
        "本页由 `scripts/generate_api_reference.py` 自动生成。",
        "",
        "## 先看这里",
        "",
        "把这里当作 Tradelearn 的 API 地图：",
        "- **Guide**：侧重“怎么用”，提供场景化示例和逻辑说明。",
        "- **Reference**：侧重“是什么”，提供精确的类/函数签名、参数定义和 Docstrings。",
        "",
        "## 快速链接",
        "",
        "| 目标 | 阅读 |",
        "|---|---|",
        "| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API 签名](../guides/lite-api.md) |",
        "| 编写 Backtrader 风格高级策略 | [Engine API 签名](../guides/engine-api.md) |",
        "| 理解两种入口的设计差异 | [策略编写指南](../guides/strategy.md) |",
        "",
        "## 公开模块",
        "",
        "| 模块 | 用途 | 常用入口 | 完整 Reference |",
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
    parts.extend(["", "## 按模块列出公开符号", ""])
    for module in modules:
        slug = _module_slug(module)
        entries = _public_symbols_text(module, limit=40)
        parts.extend(
            [
                f"### [`{module.import_path}`](reference/{slug}.md)",
                "",
                entries or "_无公开符号_",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def render_strategy_writing_guide() -> str:
    """Generate examples using qualified indicators and post-run integrations."""
    return """# 策略编写指南

`line[0]` 是当前 bar，`line[-1]` 是前一根bar；窗口只含决策时已交付数据。

## Engine 策略

```python
import tradelearn.engine as bt
from tradelearn.indicators import talib

class SmaCross(bt.Strategy):
    params = (("period", 20),)
    def __init__(self):
        self.ma = talib.SMA(self.data.close, timeperiod=self.p.period)
    def next(self):
        if not self.position and self.data.close[0] > self.ma[0]:
            self.buy(size=1)

cerebro = bt.Cerebro()
cerebro.broker.setcash(100000)
cerebro.adddata(bt.feeds.PandasData(dataname=bars))
cerebro.addstrategy(SmaCross)
strategies = cerebro.run()
```

非BT功能显式独立接入，不在Cerebro增加report/window/交易Runtime方法。
实时Store位于 `from tradelearn_runtime import NautilusStore`，由官方NA单一宿主执行。

## Lite 策略

```python
import tradelearn.lite as tl
from tradelearn.indicators import talib

class SmaCross(tl.Strategy):
    period = 20
    def init(self):
        self.ma = talib.SMA(self.data.close, timeperiod=self.period)
    def next(self):
        if not self.position() and self.data.close[0] > self.ma[0]:
            self.buy(size=1)

backtest = tl.Backtest(bars, SmaCross)
result = backtest.run()
```

Lite的window、target_weights、record/equity是便利入口，notify_order/notify_trade可选。
自定义指标沿共同Indicator状态推进；不通过任意lambda或整段重算冒充真实流式。

## 已有结果与报告

```python
from tradelearn.report import Reporter
Reporter(strategies[0].stats).report("report.html")
from tradelearn.integrations.mlflow import log
logged = log(strategies, research_result=research_result)
```

报告只消费已计算结果，不重新训练、回测或联网；附件及模型发布显式选择。
指标来源为TA-Lib、TV/PyneCore和TDX，KunQuant仅合格因子图；按来源资格使用。

## 测试验收

同输入比较Engine/Lite的共同计算事实、订单与缺失/单位，不要求策略类原样互换。
基准命令：`uv run python benchmarks/runners/benchmark_bt.py`；性能不替代数值资格。
"""


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
        "      show_source: false",
        "      show_root_heading: false",
        "      show_root_toc_entry: false",
        "      show_root_full_path: false",
        "      show_object_full_path: false",
        "      show_bases: false",
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
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Write reference pages without overwriting owner-maintained guides.",
    )
    args = parser.parse_args(argv)
    outputs = (
        write_api_reference_pages(args.docs_dir)
        if args.reference_only
        else write_api_docs(args.docs_dir)
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
