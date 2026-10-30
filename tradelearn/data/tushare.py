"""Bounded Tushare A-share data adapter (no SDK or credential files required)."""

from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd
import requests

from tradelearn.data.bars import normalize_bars


class TushareProvider:
    """Daily A-share prices, with weekly/monthly aggregation of adjusted daily data.

    Requests cover at most 25 years and return at most 5,000 daily observations.
    The latest observations are retained when the cap is reached; the result
    then has ``possibly_truncated=True``. Period bars include partial periods
    and use the first observed session date.
    Pre-adjustment uses the last factor on/before the fixed ``anchor`` (default:
    requested end). Post-adjustment follows Tushare's absolute factor convention:
    raw price * adj_factor. Volume and amount are never price-adjusted.
    """

    PAGE_SIZE = 1000
    MAX_BARS = 5000
    ENDPOINT = "https://api.tushare.pro"

    def __init__(self, token=None, transport=None):
        self._token = token if token is not None else os.environ.get("TUSHARE_TOKEN")
        self._transport = transport or self._request
        self._stocks = None
        self._deadline = None

    @staticmethod
    def _symbol(symbol):
        value = str(symbol).strip().upper()
        match = re.fullmatch(r"(SH|SZ|BJ):(\d{6})", value)
        if match:
            value = f"{match[2]}.{match[1]}"
        if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", value):
            raise ValueError("Tushare requires a six-digit symbol and SH, SZ or BJ exchange")
        return value

    def _call(self, api, params, fields):
        try:
            result = self._transport(api, params, fields)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("Invalid response")
            return result.copy()
        except Exception:
            # Do not chain upstream exceptions: request bodies can contain tokens.
            raise ConnectionError(
                "Tushare request failed; check connectivity, credentials and API permissions"
            ) from None

    def _request(self, api, params, fields):
        if not self._token:
            raise ConnectionError("TUSHARE_TOKEN is required")
        for attempt in range(2):
            remaining = self._deadline - time.monotonic() if self._deadline is not None else 20.0
            if remaining <= 0:
                raise ConnectionError("Tushare request deadline exceeded")
            timeout = min(8.0, remaining)
            retry = False
            try:
                response = requests.post(
                    self.ENDPOINT,
                    json={
                        "api_name": api,
                        "token": self._token,
                        "params": params,
                        "fields": fields,
                    },
                    timeout=(timeout / 4, timeout * 3 / 4),
                    allow_redirects=False,
                )
                retry = response.status_code == 429 or response.status_code >= 500
                if not retry:
                    response.raise_for_status()
                    payload = response.json()
                    # Tushare rate-limit messages share a general failure code.
                    message = str(payload.get("msg", ""))
                    retry = payload.get("code") != 0 and any(
                        s in message.lower() for s in ("频率", "每分钟", "每小时", "rate limit")
                    )
                    if not retry:
                        if payload.get("code") != 0:
                            raise ConnectionError("Tushare API rejected the request")
                        data = payload["data"]
                        return pd.DataFrame(data["items"], columns=data["fields"])
            except (requests.Timeout, requests.ConnectionError):
                retry = True
            if not retry or attempt == 1:
                break
            time.sleep(0.25 * (2**attempt))
        raise ConnectionError("Tushare temporarily unavailable")

    def _pages(self, api, params, fields):
        frames = []
        for offset in range(0, self.MAX_BARS, self.PAGE_SIZE):
            limit = min(self.PAGE_SIZE, self.MAX_BARS - offset)
            page = self._call(api, {**params, "limit": limit, "offset": offset}, fields)
            if page.empty:
                break
            frames.append(page.iloc[:limit])
            if len(page) < limit:
                break
        if not frames:
            return pd.DataFrame(columns=fields.split(","))
        frame = pd.concat(frames, ignore_index=True)
        if frame.trade_date.duplicated().any():
            raise ValueError("Tushare returned duplicate dates or ignored pagination")
        return frame.sort_values("trade_date")

    def search(self, query):
        """Search the cached listed-stock directory by code or name."""
        self._deadline = time.monotonic() + 20.0
        if self._stocks is None:
            stocks = self._call(
                "stock_basic",
                {"exchange": "", "list_status": "L", "limit": 10000},
                "ts_code,name,exchange",
            )
            self._stocks = [
                dict(
                    symbol=self._symbol(row.ts_code),
                    name=str(row["name"]),
                    exchange={"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}[str(row.ts_code)[-2:]],
                    type="stock",
                )
                for _, row in stocks.iterrows()
            ]
        needle = str(query).strip().casefold()
        return [
            dict(s)
            for s in self._stocks
            if needle in s["symbol"].casefold() or needle in s["name"].casefold()
        ][:100]

    stocksearch = search

    def resolve(self, symbol):
        canonical = self._symbol(symbol)
        matches = self.search(canonical)
        if not matches:
            raise ValueError("Unknown listed Tushare stock")
        return matches[0]

    @staticmethod
    def _date(value):
        try:
            date = pd.Timestamp(value)
            if pd.isna(date):
                raise ValueError()
            if date.tzinfo is not None:
                date = date.tz_convert("Asia/Shanghai").tz_localize(None)
            return date.normalize()
        except Exception:
            raise ValueError("Invalid Tushare date") from None

    def history_ohlc(self, symbol, start=None, end=None, freq="1d", adjust="none", anchor=None):
        self._deadline = time.monotonic() + 20.0
        canonical = self._symbol(symbol)
        if freq not in ("1d", "1w", "1M"):
            raise ValueError("Tushare supports only 1d, 1w and 1M frequencies")
        if adjust not in ("none", "pre", "post"):
            raise ValueError("Unsupported Tushare adjustment")
        today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        end_date = self._date(end) if end is not None else today
        anchor_date = self._date(anchor) if anchor is not None else end_date
        start_date = self._date(start) if start is not None else end_date - pd.DateOffset(years=20)
        if end_date > today or anchor_date > today or start_date > end_date:
            raise ValueError("Tushare dates must be ordered and cannot be in the future")
        if start_date < max(end_date, anchor_date) - pd.DateOffset(years=25):
            raise ValueError("Tushare requests are limited to 25 years")
        params = dict(
            ts_code=canonical,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        raw = self._pages("daily", params, "ts_code,trade_date,open,high,low,close,vol,amount")
        raw = raw[
            (raw.trade_date >= params["start_date"]) & (raw.trade_date <= params["end_date"])
        ].copy()
        if "ts_code" in raw and not raw.empty and not raw.ts_code.eq(canonical).all():
            raise ValueError("Tushare returned an unexpected symbol")
        if adjust != "none" and not raw.empty:
            factor_params = {**params, "start_date": str(raw.trade_date.min())}
            factors = self._pages("adj_factor", factor_params, "ts_code,trade_date,adj_factor")
            factor = pd.to_numeric(
                factors.set_index("trade_date").adj_factor.reindex(raw.trade_date), errors="coerce"
            ).to_numpy()
            if not np.isfinite(factor).all() or (factor <= 0).any():
                raise ValueError("Missing or invalid adjustment factor coverage")
            denominator = 1.0
            if adjust == "pre":
                anchor_rows = self._call(
                    "adj_factor",
                    {
                        **factor_params,
                        "start_date": (anchor_date - pd.DateOffset(years=25)).strftime("%Y%m%d"),
                        "end_date": anchor_date.strftime("%Y%m%d"),
                        "limit": 1,
                        "offset": 0,
                    },
                    "ts_code,trade_date,adj_factor",
                )
                anchor_rows = anchor_rows[
                    anchor_rows.trade_date <= anchor_date.strftime("%Y%m%d")
                ].sort_values("trade_date")
                if anchor_rows.empty:
                    raise ValueError("Missing adjustment factor at anchor")
                denominator = float(anchor_rows.iloc[-1].adj_factor)
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError("Invalid adjustment factor at anchor")
            for col in ("open", "high", "low", "close"):
                raw[col] = pd.to_numeric(raw[col]) * factor / denominator
        capped = len(raw) == self.MAX_BARS
        raw = raw.rename(columns={"trade_date": "timestamp", "vol": "volume"})
        raw["timestamp"] = pd.to_datetime(raw.timestamp, format="%Y%m%d", utc=True)
        raw["symbol"] = canonical
        raw["volume"] = pd.to_numeric(raw.volume) * 100.0
        raw["amount"] = pd.to_numeric(raw.amount) * 1000.0
        raw = raw[["timestamp", "symbol", "open", "high", "low", "close", "volume", "amount"]]
        # Validate daily observations before groupby can skip missing prices or
        # turn incomplete volume into an apparently complete weekly/monthly sum.
        ohlcv = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
        if not np.isfinite(ohlcv.to_numpy(dtype=float)).all():
            raise ValueError("Tushare OHLCV observations must be finite")
        raw[ohlcv.columns] = ohlcv
        if freq != "1d" and not raw.empty:
            periods = raw.timestamp.dt.tz_localize(None).dt.to_period(
                "W-SUN" if freq == "1w" else "M"
            )
            raw = (
                raw.groupby(periods, sort=True)
                .agg(
                    timestamp=("timestamp", "first"),
                    symbol=("symbol", "first"),
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                    amount=("amount", "sum"),
                )
                .reset_index(drop=True)
            )
        # Already adjusted; deliberately omit adj_factor to avoid normalize_bars scaling again.
        bars = normalize_bars(
            raw, market="CN", freq=freq, engine="tushare", source=self.ENDPOINT, adjust=adjust
        )
        bars.attrs.update(
            adjustment_anchor=anchor_date.strftime("%Y-%m-%d"),
            volume_unit="shares",
            amount_unit="CNY",
            partial_periods=True,
            daily_bar_limit=self.MAX_BARS,
            possibly_truncated=capped,
        )
        return bars
