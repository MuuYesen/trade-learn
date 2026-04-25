"""Parquet-backed Bars cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradelearn.data.bars import bars_fingerprint


@dataclass(frozen=True)
class CacheEntry:
    """Paths written for one cached Bars object."""

    data_path: Path
    meta_path: Path
    fingerprint: str


class BarsCache:
    """Simple parquet cache for Bars DataFrames."""

    def __init__(self, root: Path | str) -> None:
        """Create a cache rooted at ``root``."""
        self.root = Path(root)

    def write(self, engine: str, symbol: str, freq: str, bars: pd.DataFrame) -> CacheEntry:
        """Write Bars data and sidecar metadata."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = bars_fingerprint(bars)

        bars.to_parquet(data_path)
        meta = {
            "fingerprint": fingerprint,
            "attrs": dict(bars.attrs),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return CacheEntry(data_path=data_path, meta_path=meta_path, fingerprint=fingerprint)

    def read(self, engine: str, symbol: str, freq: str) -> pd.DataFrame:
        """Read cached Bars data and restore attrs."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        bars = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bars.attrs.update(meta["attrs"])
        return bars

    def fingerprint(self, engine: str, symbol: str, freq: str) -> str:
        """Return the cached fingerprint for one engine/symbol/frequency key."""
        meta = json.loads(self._meta_path(engine, symbol, freq).read_text(encoding="utf-8"))
        return str(meta["fingerprint"])

    def _data_path(self, engine: str, symbol: str, freq: str) -> Path:
        """Return the parquet path for a cache key."""
        return self.root / engine / symbol / f"{freq}.parquet"

    def _meta_path(self, engine: str, symbol: str, freq: str) -> Path:
        """Return the metadata sidecar path for a cache key."""
        return self.root / engine / symbol / f"{freq}.json"
