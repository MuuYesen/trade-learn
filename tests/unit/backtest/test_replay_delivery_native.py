"""Explicit replay delivery identity/date projection in the same native engine."""
import pytest
from tradelearn._rust import RustBacktestEngine


def engine():
    return RustBacktestEngine([],[],[],[],[],[],1000.,0.,False,False,False,0.,0.,True,True,False)


def payload():
    return (['A'],[100],[20.],[21.],[19.],[21.],[10.],[30.],[31.],[29.],[31.],False)


def test_native_replay_view_gate_preserves_raw_fill_source_time_and_single_account():
    value=engine();oid=value.submit_order_for_symbol('A','buy','market',.25)
    value.configure_market_eligibility(oid,10)
    first=value.step_bars_with_quote_views_compact(*payload(),[('A',1,10)])
    assert first[0] is None and first[1]==1000.
    second=value.step_bars_with_quote_views_compact(*payload(),[('A',2,20)])
    assert second[0] is not None and second[1]==992.5


@pytest.mark.parametrize('views',[[('foreign',1,10)],[('A',0,10)],[('A',1,10),('A',2,20)]])
def test_invalid_replay_view_scope_and_duplicate_identity_are_atomic(views):
    value=engine()
    with pytest.raises(ValueError):value.step_bars_with_quote_views_compact(*payload(),views)
    assert value.get_cash()==1000.


def test_old_quote_ffi_and_empty_view_ffi_are_identical():
    old,new=engine(),engine()
    old.submit_order_for_symbol('A','buy','market',.25)
    new.submit_order_for_symbol('A','buy','market',.25)
    assert old.step_bars_with_quotes_compact(*payload())==new.step_bars_with_quote_views_compact(*payload(),[])


def test_replay_close_phase_uses_captured_creation_close_without_changing_quote():
    value=engine()
    value.step_bars_with_quote_views_compact(*payload(),[('A',1,10)])
    oid=value.submit_order_for_symbol('A','buy','market',.25)
    value.configure_market_close_reference(oid,21.)
    close=list(payload());close[-1]=True
    fill=value.step_bars_with_quote_views_compact(*close,[('A',1,10)])
    assert fill[0] is not None and fill[1]==994.75
    assert value.get_equity()==1000.


def test_clone_cursor_withdrawal_is_a_new_delivery_but_not_a_new_market_date():
    value=engine();value.step_bars_with_quote_views_compact(*payload(),[('A',1,10)])
    oid=value.submit_order_for_symbol('A','buy','market',.25);value.configure_market_eligibility(oid,10)
    withdrawn=value.step_bars_with_quote_views_compact(*payload(),[('A',2,9)])
    assert withdrawn[0] is None and withdrawn[1]==1000.
    assert value.step_bars_with_quote_views_compact(*payload(),[('A',3,20)])[1]==992.5
