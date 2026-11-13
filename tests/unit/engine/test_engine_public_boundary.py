import importlib
import pytest
import tradelearn.engine as bt


def test_engine_has_no_non_bt_facade_exports():
    for name in ['StrategyRuntime','RuntimeContext','DataFeed','ExecutedInfo','Params',
                 'FixedSize','PercentSizer','AllInSizer','GridSearchResult','grid_search',
                 'TradeLogStrategy','BarRangeSlippage','CNAStockCommission','FixedCommission',
                 'FixedSlippage','PercentCommission','PercentSlippage','TargetOrderConstraints','pta','tdx','tv']:
        assert not hasattr(bt,name), name


def test_nautilus_store_has_lazy_public_entry_and_keeps_internal_implementation():
    assert 'NautilusStore' in dir(bt.stores)
    assert 'NautilusStore' in bt.stores.__all__
    integration=importlib.import_module('tradelearn_runtime')
    assert 'NautilusStore' in dir(integration)
    assert 'nautilus_trader' not in integration.__dict__


def test_engine_strategy_has_no_framework_current_datetime_alias():
    from tradelearn.engine import Strategy
    assert not hasattr(Strategy,'current_datetime')
    assert callable(Strategy._decision_datetime)
