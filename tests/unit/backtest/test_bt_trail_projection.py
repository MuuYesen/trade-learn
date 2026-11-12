"""Expected vectors verified against frozen Backtrader 1.9.78.123 order objects."""
from types import SimpleNamespace as NS
import pytest
from tradelearn.backtest.models import Order


@pytest.mark.parametrize('side,controls,limit,expected',[
    (Order.Sell,{'trailamount':1},98,[(99,97),(109,107),(109,107),(109,107)]),
    (Order.Buy,{'trailamount':1},102,[(101,103),(101,103),(91,93),(91,93)]),
    (Order.Sell,{'trailpercent':.1},98,[(90,88),(99,97),(99,97),(99,97)]),
    (Order.Buy,{'trailpercent':.1},102,[(110,112),(110,112),(99,101),(99,101)]),
])
def test_created_trailing_prices_and_public_adjustment_keep_original_bt_delta(side,controls,limit,expected):
    order=Order(1,NS(close=[100]),side,1,price=100,pricelimit=limit,
        exectype=Order.StopTrailLimit,**controls)
    order._initialize_clock(NS())
    actual=[(order.created.price,order.created.pricelimit)]
    for close in [110,90,100]:
        order.trailadjust(close)
        actual.append((order.created.price,order.created.pricelimit))
    for got,want in zip(actual,expected,strict=True):assert got==pytest.approx(want)
    assert order.price==100 and order.pricelimit==limit


def test_unspecified_limit_uses_original_reference_then_moves_with_trigger():
    order=Order(1,NS(close=[100]),Order.Sell,1,price=100,
        exectype=Order.StopTrailLimit,trailamount=5)
    order._initialize_clock(NS())
    assert (order.created.price,order.created.pricelimit)==(95,95)
    order.trailadjust(110)
    assert (order.created.price,order.created.pricelimit)==(105,105)
