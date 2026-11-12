"""Actual fixed-BT calls, including parameter profiles and default/error behavior."""
from datetime import datetime
from types import SimpleNamespace as NS
import backtrader as official
import pytest
from tradelearn import engine as bt

PROFILES=[{}, {'commission':.1,'percabs':False},
    {'commission':2.,'margin':100.,'mult':10.,'leverage':2.,'interest':.05,'interest_long':True},
    {'commission':.001,'stocklike':True,'commtype':0,'leverage':1.5,'automargin':.1}]
CALLS=[('cashadjust',(2.,10.,12.)),('confirmexec',(2.,10.)),
    ('get_credit_interest',(None,NS(size=-2.,price=10.,datetime=datetime(2026,1,1)),datetime(2026,1,4))),
    ('get_leverage',()),('get_margin',(10.,)),('getcommission',(-2.,10.)),
    ('getoperationcost',(-2.,10.)),('getsize',(11.,105.)),
    ('getvalue',(NS(size=-2.,price=10.),12.)),('getvaluesize',(-2.,10.)),
    ('profitandloss',(-2.,10.,12.))]


@pytest.mark.parametrize('name',['CommInfoBase','CommissionInfo'])
@pytest.mark.parametrize('parameters',PROFILES)
@pytest.mark.parametrize('method,args',CALLS)
def test_commission_math_calls_match_fixed_bt_without_claiming_native_fees(name,parameters,method,args):
    expected=getattr(getattr(official,name)(**parameters),method)(*args)
    actual=getattr(getattr(bt,name)(**parameters),method)(*args)
    if expected is None:assert actual is None
    else:assert actual==pytest.approx(expected)


def outcome(call):
    try:return ('value',call())
    except Exception as error:return ('exception',type(error).__name__)


@pytest.mark.parametrize('timeframe',range(1,10))
@pytest.mark.parametrize('compression',[None,1,2])
def test_timeframe_default_and_explicit_calls_match(timeframe,compression):
    assert outcome(lambda:bt.TimeFrame.getname(timeframe,compression))==outcome(lambda:official.TimeFrame.getname(timeframe,compression))
    assert bt.TimeFrame.TName(timeframe)==official.TimeFrame.TName(timeframe)
    assert bt.TimeFrame.TFrame(bt.TimeFrame.TName(timeframe))==official.TimeFrame.TFrame(official.TimeFrame.TName(timeframe))


@pytest.mark.parametrize('size',[0,2,-2,0.0,2.0,.5,-.5])
def test_position_public_length_truth_and_price_match_same_bt_inputs(size):
    ours=bt.Position(size=size,price=10.)
    theirs=official.Position(size=size,price=10.)
    assert bool(ours)==bool(theirs)
    assert outcome(lambda:len(ours))==outcome(lambda:len(theirs))
    assert ours.price==theirs.price
