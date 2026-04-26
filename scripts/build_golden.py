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
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.core import GoldenDataError  # noqa: E402

REFERENCE = ROOT / "reference" / "tradelearn_1x"
MANIFEST = ROOT / "tests" / "golden" / "manifest.json"
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


def dataset_path(dataset: dict[str, str]) -> Path:
    """Return the documented parquet path for a manifest dataset."""

    filename = (
        f"{dataset['symbol']}_{dataset['start']}_{dataset['end']}_{dataset['freq']}.parquet"
    )
    return ROOT / "tests" / "golden" / "datasets" / dataset["engine"] / filename


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
        from tradelearn.data import OpenTdxProvider

        global _LAST_REFERENCE_TDX_ERROR
        try:
            bars = OpenTdxProvider().history_ohlc(symbol, start=begin, end=end)
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


def build_datasets(manifest: dict[str, object]) -> tuple[int, int]:
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
            path = dataset_path(dataset)
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


def build(version: str, out: Path, dry_run: bool, datasets_only: bool) -> int:
    """Validate inputs and either print planned jobs or fail clearly."""

    if version != "1.x":
        raise GoldenDataError("Stage 0 only supports --version 1.x")
    validate_reference()
    manifest = load_manifest()
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
            build_datasets(manifest)
        except GoldenDataError as exc:
            raise GoldenDataError(f"dataset generation failed: {exc}") from exc
        return 0

    raise GoldenDataError(
        "expected generation is blocked until a concrete 1.x backtest adapter exists"
    )


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the golden builder CLI."""

    args = parse_args(argv)
    try:
        return build(args.version, args.out, args.dry_run, args.datasets_only)
    except GoldenDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
