"""Timezone helpers for UTC-aware trade-learn contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def ensure_utc(value: Any) -> pd.Timestamp:
    """Return ``value`` as a UTC-aware pandas timestamp.

    Naive values are interpreted as UTC. Aware values are converted to UTC.
    """

    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def utc_now() -> datetime:
    """Return the current UTC-aware datetime."""

    return datetime.now(timezone.utc)
