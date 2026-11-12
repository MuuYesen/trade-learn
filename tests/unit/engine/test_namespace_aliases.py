"""Existing indicator/observer modules use Backtrader's documented aliases."""
from tradelearn import engine as bt


def test_ind_and_obs_alias_the_existing_common_modules():
    assert bt.ind is bt.indicators
    assert bt.ind.Indicator is bt.Indicator
    assert bt.obs is bt.observers
    assert {'ind','indicators','obs','observers'} <= set(bt.__all__)
