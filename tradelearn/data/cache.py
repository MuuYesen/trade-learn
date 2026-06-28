"""Parquet-backed Bars cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tradelearn.core.errors import DataError
from tradelearn.data.bars import bars_fingerprint


class CacheMissError(DataError):
    """Raised when a requested Bars cache entry is missing."""


class CacheExpiredError(DataError):
    """Raised when a requested Bars cache entry is stale in online mode."""


@dataclass(frozen=True)
class CacheEntry:
    """Paths written for one cached Bars object."""

    data_path: Path
    meta_path: Path
    fingerprint: str
    written_at: pd.Timestamp


class BarsCache:
    """Simple parquet cache for Bars DataFrames."""

    def __init__(
        self,
        root: Path | str,
        *,
        ttl: timedelta | int | float | None = None,
        offline: bool = False,
    ) -> None:
        """Create a cache rooted at ``root``."""
        self.root = Path(root)
        self.ttl = _coerce_ttl(ttl)
        self.offline = offline

    def write(
        self,
        engine: str,
        symbol: str,
        freq: str,
        bars: pd.DataFrame,
        *,
        now: pd.Timestamp | datetime | str | None = None,
    ) -> CacheEntry:
        """Write Bars data and sidecar metadata."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = bars_fingerprint(bars)
        written_at = _coerce_timestamp(now)

        bars.to_parquet(data_path)
        meta = {
            "fingerprint": fingerprint,
            "written_at": written_at.isoformat(),
            "attrs": dict(bars.attrs),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return CacheEntry(
            data_path=data_path,
            meta_path=meta_path,
            fingerprint=fingerprint,
            written_at=written_at,
        )

    def read(
        self,
        engine: str,
        symbol: str,
        freq: str,
        *,
        ttl: timedelta | int | float | None = None,
        offline: bool | None = None,
        now: pd.Timestamp | datetime | str | None = None,
    ) -> pd.DataFrame:
        """Read cached Bars data and restore attrs."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        self._raise_if_missing(data_path, meta_path)
        stale = not self.is_fresh(engine, symbol, freq, ttl=ttl, now=now)
        allow_stale = self.offline if offline is None else offline
        if stale and not allow_stale:
            raise CacheExpiredError(f"Bars cache entry is stale: {engine}/{symbol}/{freq}")

        bars = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bars.attrs.update(meta["attrs"])
        if stale:
            bars.attrs["cache_stale"] = True
        return bars

    def fingerprint(self, engine: str, symbol: str, freq: str) -> str:
        """Return the cached fingerprint for one engine/symbol/frequency key."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        self._raise_if_missing(data_path, meta_path)
        meta = json.loads(self._meta_path(engine, symbol, freq).read_text(encoding="utf-8"))
        return str(meta["fingerprint"])

    def exists(self, engine: str, symbol: str, freq: str) -> bool:
        """Return whether both parquet data and metadata exist for a cache key."""
        return self._data_path(engine, symbol, freq).exists() and self._meta_path(
            engine,
            symbol,
            freq,
        ).exists()

    def is_fresh(
        self,
        engine: str,
        symbol: str,
        freq: str,
        *,
        ttl: timedelta | int | float | None = None,
        now: pd.Timestamp | datetime | str | None = None,
    ) -> bool:
        """Return whether a cache entry exists and has not exceeded its TTL."""
        data_path = self._data_path(engine, symbol, freq)
        meta_path = self._meta_path(engine, symbol, freq)
        if not data_path.exists() or not meta_path.exists():
            return False

        effective_ttl = self.ttl if ttl is None else _coerce_ttl(ttl)
        if effective_ttl is None:
            return True

        written_at = self._written_at(data_path, meta_path)
        age = _coerce_timestamp(now) - written_at
        return age <= effective_ttl

    def _data_path(self, engine: str, symbol: str, freq: str) -> Path:
        """Return the parquet path for a cache key."""
        return self.root / engine / symbol / f"{freq}.parquet"

    def _meta_path(self, engine: str, symbol: str, freq: str) -> Path:
        """Return the metadata sidecar path for a cache key."""
        return self.root / engine / symbol / f"{freq}.json"

    def _written_at(self, data_path: Path, meta_path: Path) -> pd.Timestamp:
        """Return metadata write time, falling back to parquet mtime for old caches."""
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "written_at" in meta:
            return _coerce_timestamp(meta["written_at"])
        return pd.Timestamp(datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc))

    def _raise_if_missing(self, data_path: Path, meta_path: Path) -> None:
        """Raise a data-layer error when a cache entry is incomplete."""
        if not data_path.exists() or not meta_path.exists():
            raise CacheMissError(f"Bars cache entry is missing: {data_path}")


def _coerce_timestamp(value: pd.Timestamp | datetime | str | None) -> pd.Timestamp:
    """Return a UTC-aware timestamp."""
    timestamp = pd.Timestamp.now(tz=timezone.utc) if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_ttl(value: timedelta | int | float | None) -> timedelta | None:
    """Return a timedelta TTL, accepting seconds for numeric values."""
    if value is None or isinstance(value, timedelta):
        return value
    if isinstance(value, int | float):
        return timedelta(seconds=float(value))
    raise TypeError(f"Unsupported cache ttl: {type(value).__name__}")
