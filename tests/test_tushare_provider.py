import importlib

import pandas as pd
import pytest


def provider(transport, **kwargs):
    return importlib.import_module("tradelearn.data.tushare").TushareProvider(
        transport=transport, **kwargs
    )


def daily(dates=("20240102", "20240103", "20240104")):
    return pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                trade_date=d,
                open=10,
                high=12,
                low=9,
                close=11,
                vol=2,
                amount=3,
            )
            for d in dates
        ]
    )


class Fixture:
    def __init__(self, rows=None, factors=None):
        self.rows = daily() if rows is None else rows
        self.factors = (
            pd.DataFrame(
                {"trade_date": ["20240102", "20240103", "20240104"], "adj_factor": [1.0, 2.0, 2.0]}
            )
            if factors is None
            else factors
        )
        self.calls = []

    def __call__(self, api, params, fields):
        self.calls.append((api, params.copy(), fields))
        if api == "stock_basic":
            return pd.DataFrame([dict(ts_code="600000.SH", name="浦发银行", exchange="SSE")])
        frame = self.rows if api == "daily" else self.factors
        frame = frame[
            (frame.trade_date >= params["start_date"]) & (frame.trade_date <= params["end_date"])
        ].sort_values("trade_date", ascending=False)
        return frame.iloc[
            params.get("offset", 0) : params.get("offset", 0) + params.get("limit", 5000)
        ].copy()


def test_daily_contract_units_and_search_cache():
    f = Fixture()
    p = provider(f)
    bars = p.history_ohlc("SH:600000", start="2024-01-02", end="2024-01-04")
    assert bars.index.names == ["timestamp", "symbol"]
    assert list(bars.index.get_level_values("timestamp")) == list(
        pd.date_range("2024-01-02", periods=3, tz="UTC")
    )
    assert set(bars.index.get_level_values("symbol")) == {"600000.SH"}
    assert list(bars.volume) == [200.0, 200.0, 200.0]
    assert list(bars.amount) == [3000.0, 3000.0, 3000.0]
    assert bars.attrs["adjust"] == "none"
    assert p.search("浦发") == [
        {"symbol": "600000.SH", "name": "浦发银行", "exchange": "SSE", "type": "stock"}
    ]
    assert p.resolve("600000.SH")["name"] == "浦发银行"
    assert sum(c[0] == "stock_basic" for c in f.calls) == 1


def test_adjustments_fixed_anchor_and_no_double_adjust():
    p = provider(Fixture())
    pre = p.history_ohlc(
        "600000.SH", start="2024-01-02", end="2024-01-03", adjust="pre", anchor="2024-01-04"
    )
    assert list(pre.close) == [5.5, 11.0]
    assert pre.attrs["adjust"] == "pre"
    post = p.history_ohlc("600000.SH", start="2024-01-02", end="2024-01-04", adjust="post")
    assert list(post.close) == [11.0, 22.0, 22.0]


def test_missing_factor_fails_closed():
    f = Fixture(
        factors=pd.DataFrame({"trade_date": ["20240102", "20240104"], "adj_factor": [1.0, 2.0]})
    )
    with pytest.raises(ValueError, match="factor"):
        provider(f).history_ohlc("600000.SH", start="2024-01-02", end="2024-01-04", adjust="pre")


def test_weekly_adjusts_before_aggregation_and_respects_cutoff():
    b = provider(Fixture()).history_ohlc(
        "600000.SH", start="2024-01-02", end="2024-01-03", freq="1w", adjust="pre"
    )
    assert len(b) == 1
    assert b.iloc[0].open == 5
    assert b.iloc[0].high == 12
    assert b.iloc[0].low == 4.5
    assert b.iloc[0].close == 11
    assert b.iloc[0].volume == 400
    assert b.index[0][0] <= pd.Timestamp("2024-01-03", tz="UTC")


def test_pagination_is_bounded_and_chronological():
    dates = pd.date_range("2005-01-01", periods=5500).strftime("%Y%m%d")
    f = Fixture(rows=daily(dates))
    p = provider(f)
    p.PAGE_SIZE = 1000
    b = p.history_ohlc("600000.SH", start="2005-01-01", end="2020-01-01")
    assert len(b) == 5000
    assert b.index.is_monotonic_increasing
    assert len(f.calls) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(freq="1m"),
        dict(adjust="bad"),
        dict(start="1900-01-01"),
        dict(anchor="2999-01-01", adjust="pre"),
    ],
)
def test_invalid_or_unbounded_requests_fail_before_transport(kwargs):
    f = Fixture()
    with pytest.raises(ValueError):
        provider(f).history_ohlc("600000.SH", end="2024-01-04", **kwargs)
    assert not f.calls


def test_transport_exception_never_exposes_secrets():
    def broken(*args):
        raise RuntimeError("secret-token https://example?token=secret-token")

    with pytest.raises(ConnectionError) as exc:
        provider(broken).history_ohlc("600000.SH", start="2024-01-01", end="2024-01-04")
    assert "secret-token" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_anchor_can_precede_forward_preload_end():
    b = provider(Fixture()).history_ohlc(
        "600000.SH", start="2024-01-02", end="2024-01-04", anchor="2024-01-02", adjust="pre"
    )
    assert list(b.close) == [11, 22, 22]


def test_monthly_aggregation_and_empty_history():
    b = provider(Fixture()).history_ohlc(
        "600000.SH", start="2024-01-02", end="2024-01-04", freq="1M"
    )
    assert len(b) == 1 and b.iloc[0].volume == 600
    b = provider(Fixture()).history_ohlc("600000.SH", start="2023-01-02", end="2023-01-04")
    assert b.empty and b.index.names == ["timestamp", "symbol"]


def test_default_transport_retries_are_bounded_and_redacted(monkeypatch):
    mod = importlib.import_module("tradelearn.data.tushare")
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        raise mod.requests.Timeout("SECRET")

    monkeypatch.setattr(mod.requests, "post", request)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    with pytest.raises(ConnectionError, match="Tushare") as exc:
        mod.TushareProvider(token="SECRET").history_ohlc(
            "600000.SH", start="2024-01-01", end="2024-01-04"
        )
    assert "SECRET" not in str(exc.value)
    assert len(calls) == 2
    assert all(c[0] == "https://api.tushare.pro" and sum(c[1]["timeout"]) <= 8 for c in calls)


@pytest.mark.parametrize("freq", ["1d", "1w", "1M"])
@pytest.mark.parametrize("column", ["open", "high", "low", "close", "vol"])
def test_missing_daily_observations_cannot_be_hidden_by_aggregation(freq, column):
    rows = daily()
    rows.loc[1, column] = float("nan")
    with pytest.raises(ValueError, match="OHLCV"):
        provider(Fixture(rows=rows)).history_ohlc(
            "600000.SH",
            start="2024-01-02",
            end="2024-01-04",
            freq=freq,
        )
