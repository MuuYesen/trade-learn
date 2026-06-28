#!/usr/bin/env python
"""Build golden baselines from the frozen 1.x oracle.

Stage 0 wires the command, manifest validation, and dry-run behavior. Real
expected generation is intentionally blocked until concrete adapters exist.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.core.errors import GoldenDataError  # noqa: E402

REFERENCE = ROOT / "reference" / "tradelearn_1x"
MANIFEST = ROOT / "tests" / "golden" / "manifest.json"
DATASETS_ROOT = ROOT / "tests" / "golden" / "datasets"
STRATEGY_DIR = ROOT / "tests" / "golden" / "strategies"
PROVIDER_MODULES = {
    "tdx": "opentdx.tdxClient",
    "tv": "tvDatafeed",
}

REFERENCE_TDX_PACKAGE = "moot" + "dx"
REFERENCE_TDX_MODULE = REFERENCE_TDX_PACKAGE + ".quotes"
_LAST_REFERENCE_TDX_ERROR: str | None = None


def _module_available(name: str) -> bool:
    """Return whether a module can be imported without raising."""

    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__spec__", None) is None:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def load_manifest(path: Path = MANIFEST) -> dict[str, object]:
    """Load the golden manifest."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def planned_jobs(manifest: dict[str, object]) -> list[tuple[str, str]]:
    """Return ``(strategy, dataset)`` jobs from the manifest."""

    strategies = [item["name"] for item in manifest["strategies"]]
    datasets = [item["symbol"] for item in manifest["datasets"]]
    return [(strategy, dataset) for strategy in strategies for dataset in datasets]


def dataset_filename(dataset: dict[str, str]) -> str:
    """Return the documented parquet filename for a manifest dataset."""

    return f"{dataset['symbol']}_{dataset['start']}_{dataset['end']}_{dataset['freq']}.parquet"


def dataset_path(dataset: dict[str, str], datasets_root: Path = DATASETS_ROOT) -> Path:
    """Return the documented parquet path for a manifest dataset."""

    return datasets_root / dataset["engine"] / dataset_filename(dataset)


def dataset_label(dataset: dict[str, str]) -> str:
    """Return a provider-aware dataset label for diagnostics."""

    if dataset["engine"] == "tv" and dataset.get("exchange"):
        return f"tv:{dataset['exchange']}:{dataset['symbol']}"
    return f"{dataset['engine']}:{dataset['symbol']}"


def validate_reference() -> None:
    """Ensure the frozen 1.x oracle exists."""

    if not (REFERENCE / "query" / "query.py").exists():
        raise GoldenDataError("reference/tradelearn_1x is missing the 1.x Query oracle")


def ensure_reference_path() -> None:
    """Make the frozen oracle importable before the active package."""

    reference_parent = str(REFERENCE.parent)
    if reference_parent in sys.path:
        sys.path.remove(reference_parent)
    sys.path.insert(0, reference_parent)


def _install_reference_tdx_bridge() -> None:
    """Bridge the frozen 1.x TDX import to the current opentdx provider."""

    package = sys.modules.setdefault(REFERENCE_TDX_PACKAGE, types.ModuleType(REFERENCE_TDX_PACKAGE))
    quotes = types.ModuleType(REFERENCE_TDX_MODULE)

    class Quotes:
        @staticmethod
        def factory(**_: object) -> object:
            return _OpenTdxReferenceClient()

    quotes.Quotes = Quotes
    sys.modules[REFERENCE_TDX_MODULE] = quotes
    package.__dict__["quotes"] = quotes


class _OpenTdxReferenceClient:
    """Compatibility client for the frozen 1.x Query TDX branch."""

    def ohlc(
        self,
        *,
        symbol: str,
        begin: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> Any:
        from tradelearn.data import TdxProvider

        global _LAST_REFERENCE_TDX_ERROR
        try:
            bars = TdxProvider().history_ohlc(symbol, start=begin, end=end)
        except Exception as exc:
            _LAST_REFERENCE_TDX_ERROR = f"{type(exc).__name__}: {exc}"
            raise
        frame = bars.rename(columns={"timestamp": "datetime", "symbol": "code"}).copy()
        frame["date"] = frame["datetime"]
        frame["factor"] = 1.0
        if "amount" not in frame.columns:
            frame["amount"] = frame["close"] * frame["volume"] * 100
        return frame.set_index("datetime")


def _install_provider_stubs() -> None:
    """Install minimal provider stubs for oracle import diagnostics."""

    if not _module_available("yfinance"):
        sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
    _install_reference_tdx_bridge()
    if not _module_available("tvDatafeed"):
        tvdatafeed = types.ModuleType("tvDatafeed")
        tvdatafeed.TvDatafeed = object
        tvdatafeed.Interval = object
        sys.modules["tvDatafeed"] = tvdatafeed


def load_reference_query(allow_provider_stubs: bool = False) -> Any:
    """Load Query from the frozen 1.x oracle."""

    validate_reference()
    ensure_reference_path()
    if allow_provider_stubs:
        _install_provider_stubs()
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "tradelearn" or name.startswith("tradelearn.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    reference_pkg = types.ModuleType("tradelearn")
    reference_pkg.__path__ = [str(REFERENCE)]
    sys.modules["tradelearn"] = reference_pkg
    try:
        query_module = importlib.import_module("tradelearn.query")
        Query = query_module.Query
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        message = f"failed to import reference Query; missing module: {missing}"
        raise GoldenDataError(message) from exc
    except Exception as exc:
        message = f"failed to import reference Query: {type(exc).__name__}: {exc}"
        raise GoldenDataError(message) from exc
    finally:
        loaded_reference_modules = [
            name
            for name in sys.modules
            if name == "tradelearn" or name.startswith("tradelearn.")
        ]
        for name in loaded_reference_modules:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return Query


def provider_statuses() -> dict[str, bool]:
    """Return import availability for 1.x legacy optional provider modules."""

    return {
        engine: _module_available(module)
        for engine, module in PROVIDER_MODULES.items()
    }


def validate_dataset_providers(manifest: dict[str, object]) -> None:
    """Fail before fetching when a manifest provider is unavailable."""

    statuses = provider_statuses()
    engines = sorted({dataset["engine"] for dataset in manifest["datasets"]})
    missing = [engine for engine in engines if not statuses.get(engine, False)]
    if not missing:
        return

    details = ", ".join(
        f"{engine}:{PROVIDER_MODULES.get(engine, 'unknown')}" for engine in missing
    )
    raise GoldenDataError(
        "dataset provider unavailable: "
        f"{details}; install available providers before live access validation"
    )


def filter_manifest_by_engine(manifest: dict[str, object], engine: str) -> dict[str, object]:
    """Return a manifest limited to one data engine, or the original manifest."""

    if engine == "all":
        return manifest
    return {
        **manifest,
        "datasets": [
            dataset
            for dataset in manifest["datasets"]
            if dataset["engine"] == engine
        ],
    }


def fetch_dataset(query: Any, dataset: dict[str, str]) -> Any:
    """Fetch one dataset through the 1.x Query oracle."""

    global _LAST_REFERENCE_TDX_ERROR
    if dataset["engine"] == "tdx":
        _LAST_REFERENCE_TDX_ERROR = None
    try:
        data = query.history_ohlc(
            engine=dataset["engine"],
            symbol=dataset["symbol"],
            exchange=dataset.get("exchange"),
            start=dataset["start"],
            end=dataset["end"],
        )
    except Exception as exc:
        raise GoldenDataError(
            f"{exc}"
        ) from exc
    if data is None:
        if dataset["engine"] == "tdx" and _LAST_REFERENCE_TDX_ERROR:
            raise GoldenDataError(
                "reference Query returned no data for "
                f"{dataset_label(dataset)}; {_LAST_REFERENCE_TDX_ERROR}"
            )
        raise GoldenDataError(
            f"reference Query returned no data for {dataset_label(dataset)}"
        )
    return data


def write_dataset(data: Any, path: Path) -> None:
    """Persist a fetched dataset as parquet."""

    if data is None:
        raise GoldenDataError("dataset generation failed: reference Query returned None")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_parquet(path)
    except Exception as exc:
        raise GoldenDataError(f"dataset generation failed while writing {path}: {exc}") from exc


def build_datasets(
    manifest: dict[str, object],
    datasets_root: Path = DATASETS_ROOT,
) -> tuple[int, int]:
    """Fetch and write all manifest datasets."""

    validate_dataset_providers(manifest)
    query = load_reference_query(allow_provider_stubs=True)
    datasets = manifest["datasets"]
    failures: list[str] = []
    successes = 0
    for dataset in manifest["datasets"]:
        label = dataset_label(dataset)
        try:
            data = fetch_dataset(query, dataset)
            path = dataset_path(dataset, datasets_root)
            write_dataset(data, path)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"dataset={label} status=failed reason={exc}")
            continue
        successes += 1
        print(f"dataset={label} status=ok path={path}")

    total = len(datasets)
    print(f"datasets={successes}/{total}")
    if failures:
        joined = "; ".join(failures)
        raise GoldenDataError(f"{successes}/{total} datasets generated; failures: {joined}")
    return successes, total


def _strategy_class(strategy_name: str, strategy_dir: Path = STRATEGY_DIR) -> type[Any]:
    """Load one golden Strategy adapter by manifest name."""

    path = strategy_dir / f"{strategy_name}.py"
    if not path.exists():
        raise GoldenDataError(f"missing strategy adapter: {strategy_name}")
    namespace = runpy.run_path(str(path))
    from tradelearn.engine import Strategy

    strategy_classes = [
        value
        for key, value in namespace.items()
        if key.endswith("Strategy")
        and isinstance(value, type)
        and value is not Strategy
        and issubclass(value, Strategy)
    ]
    if len(strategy_classes) != 1:
        raise GoldenDataError(f"missing runnable Strategy adapter: {strategy_name}")
    return strategy_classes[0]


def _clean_json_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into stable JSON payloads."""

    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    return value


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    """Return JSON-safe records for a pandas DataFrame-like object."""

    if getattr(frame, "empty", False):
        return []
    records = frame.reset_index().to_dict(orient="records")
    return [_clean_json_value(record) for record in records]


def _series_records(series: Any) -> list[dict[str, Any]]:
    """Return JSON-safe records for a pandas Series-like object."""

    if getattr(series, "empty", False):
        return []
    frame = series.rename("value").reset_index()
    frame.columns = ["datetime", "value"]
    return [_clean_json_value(record) for record in frame.to_dict(orient="records")]


def _utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return bars with a UTC-aware datetime index for stable golden JSON."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    bars = frame.copy()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    return bars


def run_expected_job(
    strategy_name: str,
    dataset: dict[str, str],
    datasets_root: Path,
) -> dict[str, Any]:
    """Run one TV subset golden adapter and return expected payload."""

    from tradelearn.engine import Cerebro

    path = dataset_path(dataset, datasets_root)
    if not path.exists():
        raise GoldenDataError(f"missing dataset parquet: {path}")
    strategy_cls = _strategy_class(strategy_name)
    bars = _utc_index(pd.read_parquet(path))
    cerebro = Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.adddata(bars, name=dataset["symbol"])
    cerebro.addstrategy(strategy_cls)
    [strategy] = cerebro.run()
    stats = strategy.stats
    if stats is None:
        raise GoldenDataError(f"strategy did not produce stats: {strategy_name}")
    return {
        "version": "v1.0",
        "strategy": strategy_name,
        "dataset": dataset["symbol"],
        "engine": dataset["engine"],
        "source": str(path),
        "summary": _clean_json_value(stats.summary),
        "trades": _frame_records(stats.trades),
        "orders": _frame_records(stats.orders),
        "fills": _frame_records(stats.fills),
        "positions": _frame_records(stats.positions),
        "equity": _series_records(stats.equity),
    }


def run_backtrader_expected_job(
    strategy_name: str,
    dataset: dict[str, str],
    datasets_root: Path,
) -> dict[str, Any]:
    """Run one Backtrader oracle job and return expected payload."""

    from scripts.run_backtrader_oracle import SUPPORTED_BACKTRADER_STRATEGIES

    if strategy_name not in SUPPORTED_BACKTRADER_STRATEGIES:
        raise GoldenDataError(
            "Backtrader oracle currently supports "
            f"{', '.join(SUPPORTED_BACKTRADER_STRATEGIES)}"
        )
    from scripts.run_backtrader_oracle import run_backtrader_oracle

    path = dataset_path(dataset, datasets_root)
    return run_backtrader_oracle(strategy_name, path, dataset=dataset["symbol"])


def expected_path(strategy: str, dataset: str, out: Path) -> Path:
    """Return the expected JSON path used by readiness checks."""

    return out / f"{strategy}__{dataset}.json"


def build_expected(
    manifest: dict[str, object],
    out: Path,
    datasets_root: Path,
    oracle: str = "tradelearn",
) -> tuple[int, int]:
    """Build expected JSON files for every manifest strategy/dataset job."""

    datasets_by_symbol = {dataset["symbol"]: dataset for dataset in manifest["datasets"]}
    failures: list[str] = []
    successes = 0
    jobs = planned_jobs(manifest)
    if oracle == "backtrader":
        from scripts.run_backtrader_oracle import SUPPORTED_BACKTRADER_STRATEGIES

        jobs = [
            (strategy, dataset)
            for strategy, dataset in jobs
            if strategy in SUPPORTED_BACKTRADER_STRATEGIES
        ]
    payloads: list[tuple[str, Path, dict[str, Any]]] = []
    for strategy_name, dataset_symbol in jobs:
        dataset = datasets_by_symbol[dataset_symbol]
        label = f"{strategy_name}:{dataset_symbol}"
        try:
            if oracle == "backtrader":
                payload = run_backtrader_expected_job(strategy_name, dataset, datasets_root)
            else:
                payload = run_expected_job(strategy_name, dataset, datasets_root)
            path = expected_path(strategy_name, dataset_symbol, out)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"expected={label} status=failed reason={exc}")
            continue
        successes += 1
        payloads.append((label, path, payload))

    total = len(jobs)
    print(f"expected={successes}/{total}")
    if failures:
        joined = "; ".join(failures)
        raise GoldenDataError(f"{successes}/{total} expected generated; failures: {joined}")
    for label, path, payload in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        print(f"expected={label} status=ok path={path}")
    return successes, total


def build(
    version: str,
    out: Path,
    dry_run: bool,
    datasets_only: bool,
    engine: str,
    datasets_root: Path,
    oracle: str,
) -> int:
    """Validate inputs and either print planned jobs or fail clearly."""

    if version != "1.x":
        raise GoldenDataError("Stage 0 only supports --version 1.x")
    manifest = filter_manifest_by_engine(load_manifest(), engine)
    out.mkdir(parents=True, exist_ok=True)
    jobs = planned_jobs(manifest)

    if dry_run:
        print(f"reference={REFERENCE}")
        print(f"out={out}")
        print(f"jobs={len(jobs)}")
        for strategy, dataset in jobs:
            print(f"{strategy}:{dataset}")
        return 0

    if datasets_only:
        try:
            build_datasets(manifest, datasets_root)
        except GoldenDataError as exc:
            raise GoldenDataError(f"dataset generation failed: {exc}") from exc
        return 0

    try:
        build_expected(manifest, out, datasets_root, oracle=oracle)
    except GoldenDataError as exc:
        raise GoldenDataError(f"expected generation failed: {exc}") from exc
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Oracle version, currently 1.x")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs only")
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only build dataset parquet files",
    )
    parser.add_argument(
        "--engine",
        choices=["all", "tv", "tdx"],
        default="all",
        help="Limit dataset generation and dry-run jobs to one provider engine",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DATASETS_ROOT,
        help="Root directory for generated golden dataset parquet files",
    )
    parser.add_argument(
        "--oracle",
        choices=["tradelearn", "backtrader"],
        default="tradelearn",
        help="Expected generator backend; backtrader currently supports SMA/MACD/KDJ parity smoke",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the golden builder CLI."""

    args = parse_args(argv)
    try:
        return build(
            args.version,
            args.out,
            args.dry_run,
            args.datasets_only,
            args.engine,
            args.datasets_root,
            args.oracle,
        )
    except GoldenDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
