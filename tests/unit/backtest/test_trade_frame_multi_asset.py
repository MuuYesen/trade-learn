import pandas as pd

from tradelearn.backtest.engine import _trades_frame


def test_trades_frame_tracks_open_positions_per_asset() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2024-01-01", tz="UTC"),
                "data": "AAPL",
                "size": 10.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": pd.Timestamp("2024-01-02", tz="UTC"),
                "data": "MSFT",
                "size": 5.0,
                "price": 200.0,
                "commission": 0.0,
            },
            {
                "datetime": pd.Timestamp("2024-01-03", tz="UTC"),
                "data": "MSFT",
                "size": -5.0,
                "price": 220.0,
                "commission": 0.0,
            },
            {
                "datetime": pd.Timestamp("2024-01-04", tz="UTC"),
                "data": "AAPL",
                "size": -10.0,
                "price": 110.0,
                "commission": 0.0,
            },
        ]
    )

    trades = _trades_frame(fills)
    closed = trades[trades["isclosed"].astype(bool)]

    assert closed["data"].tolist() == ["MSFT", "AAPL"]
    assert closed["pnl"].tolist() == [100.0, 100.0]
