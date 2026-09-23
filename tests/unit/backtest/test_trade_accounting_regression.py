"""Closed trades must account for the complete position lifecycle."""
import math
import pandas as pd
import pytest
from tradelearn.backtest.engine import _trades_frame, _summary_trade_metrics


def closed(rows):
    fills = pd.DataFrame(rows, columns=['data', 'size', 'price', 'commission'])
    fills['datetime'] = pd.date_range('2026-01-01', periods=len(rows), tz='UTC')
    trades = _trades_frame(fills)
    return trades[trades.isclosed]


@pytest.mark.parametrize('rows,pnl,net,fees', [
    ([('A',10,100,1),('A',-5,110,1),('A',-5,120,1)], [150], [147], [3]),
    ([('A',10,100,1),('A',-10,110,1)], [100], [98], [2]),
    ([('A',-10,100,1),('A',5,90,1),('A',5,80,1)], [150], [147], [3]),
    ([('A',10,100,1),('A',10,120,1),('A',-5,130,1),('A',-15,100,1)], [-50], [-54], [4]),
    ([('A',10,100,1),('A',-15,110,3),('A',5,100,1)], [100,50], [97,48], [3,2]),
    ([('A',10,100,1),('B',10,200,2),('A',-10,110,1),('B',-10,190,2)], [100,-100], [98,-104], [2,4]),
])
def test_complete_lifecycle(rows, pnl, net, fees):
    trades = closed(rows)
    assert trades.pnl.tolist() == pytest.approx(pnl)
    assert trades.pnlcomm.tolist() == pytest.approx(net)
    assert trades.commission.tolist() == pytest.approx(fees)


def test_partial_exit_is_not_a_completed_trade():
    assert closed([('A',10,100,1),('A',-5,110,1)]).empty


def test_all_winners_have_infinite_profit_factor():
    trades = closed([('A',10,100,1),('A',-10,110,1)])
    assert math.isinf(_summary_trade_metrics(trades)['profit_factor'])


def test_mixed_profit_factor_uses_net_cycle_pnl():
    trades = closed([('A',10,100,1),('A',-10,110,1),('A',10,100,1),('A',-10,95,1)])
    assert _summary_trade_metrics(trades)['profit_factor'] == pytest.approx(98/52)


@pytest.mark.parametrize('mode', ['full', 'lazy'])
def test_real_engine_returns_complete_net_profit_in_both_stats_modes(mode):
    from tradelearn.engine import Cerebro, Strategy

    class PartialExit(Strategy):
        def next(self):
            if len(self.data) == 1:
                self.buy(size=10)
            elif len(self.data) in (2, 3):
                self.sell(size=5)

    prices = [90.0, 100.0, 110.0, 120.0, 120.0]
    bars = pd.DataFrame({key: prices for key in ['open','high','low','close']},
                        index=pd.date_range('2026-01-01', periods=5, tz='UTC'))
    bars['volume'] = 1000.0
    engine = Cerebro(stats_mode=mode)
    engine.setcash(10000)
    engine.broker.setcommission(commission=0.01)
    engine.adddata(bars)
    engine.addstrategy(PartialExit)
    [strategy] = engine.run()
    stats = strategy.stats
    assert stats.fills.price.tolist() == [100,110,120]
    trade = stats.trades[stats.trades.isclosed].iloc[0]
    assert trade.pnl == pytest.approx(150)
    assert trade.pnlcomm == pytest.approx(128.5)
    assert stats.summary['total_trades'] == 1
    assert stats.summary['win_rate_pct'] == 100
    assert stats.summary['expectancy'] == pytest.approx(128.5)
    assert stats.summary['final_value'] == pytest.approx(10128.5)
    assert math.isinf(stats.summary['profit_factor'])
