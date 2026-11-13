"""Exact fixed BT valid values and expiry, distinct from Lite relative seconds."""

from datetime import date, datetime, timedelta

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl


CASES = [None, 0, 10.9, timedelta(0), timedelta(seconds=10.9),
         datetime(2026, 1, 1, 9, 30, 10, 123000), date(2026, 1, 2),
         official.date2num(datetime(2026, 1, 1, 9, 30, 20))]


def observations(api, kind, valid):
    frame = pd.DataFrame(
        {name: [10.] * 4 for name in ('open', 'high', 'low', 'close', 'volume')},
        index=pd.to_datetime(['2026-01-01 09:30:00', '2026-01-01 09:30:05',
                              '2026-01-01 09:30:15', '2026-01-01 09:30:30']),
    )
    rows = []

    class Business(api.Strategy):
        def next(self):
            if len(self) == 1:
                self.order = self.buy(size=1, exectype=getattr(api.Order, kind),
                                      valid=valid, **({'price': 9} if kind == 'Limit' else {}))
        def notify_order(self, order):
            rows.append(order.getstatusname())

    engine = api.Cerebro(stdstats=False)
    engine.adddata(api.feeds.PandasData(dataname=frame))
    engine.addstrategy(Business)
    business = engine.run()[0]
    return business.order.valid, business.order.getstatusname(), rows


@pytest.mark.parametrize('valid', CASES)
@pytest.mark.parametrize('kind', ['Market', 'Limit'])
def test_order_valid_number_and_terminal_status_match_fixed_bt(kind, valid):
    expected_valid, expected_status, _ = observations(official, kind, valid)
    actual_valid, actual_status, _ = observations(tl, kind, valid)
    assert actual_status == expected_status
    if expected_valid is None:
        assert actual_valid is None
    else:
        assert actual_valid == pytest.approx(expected_valid, abs=1e-10, rel=0)


def test_public_order_day_is_bt_duration_sentinel_without_changing_lite_tif():
    from tradelearn.backtest.models import Order as CoreOrder
    assert tl.Order.DAY == official.Order.DAY == timedelta(0)
    assert CoreOrder.DAY == 'day'
    assert not hasattr(tl.Order, 'GTC') and not hasattr(tl.Order, 'IOC')
    expected = observations(official, 'Limit', official.Order.DAY)
    actual = observations(tl, 'Limit', tl.Order.DAY)
    assert actual[0] == pytest.approx(expected[0], abs=1e-10, rel=0)
    assert actual[1] == expected[1]
    # The Engine view recognizes the same callback projections, no second ledger.
    projection = CoreOrder(ref=1, data=None, ordtype=CoreOrder.Buy, size=1)
    assert isinstance(projection, tl.Order)
    assert issubclass(type(projection), tl.Order)
