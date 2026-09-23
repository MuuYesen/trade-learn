import pandas as pd
import pytest
from tradelearn.engine import Cerebro, Strategy

@pytest.mark.parametrize('symbol_count', [1, 2])
@pytest.mark.parametrize('cash_notifications', [False, True])
def test_rejected_buy_clears_pending_before_next_target(symbol_count, cash_notifications):
    observed = []
    class Target(Strategy):
        def next(self):
            i = len(self.data) - 1
            if i == 0:
                self.order_target_size(data=self.data, target=100)
            elif i == 1:
                observed.append(self._pending_size.get(self.data, 0))
                self.order_target_size(data=self.data, target=99)
    if cash_notifications:
        Target.notify_cashvalue = lambda self, cash, value: None
    prices = [100., 101., 101., 101.]
    frame = pd.DataFrame({k: prices for k in ['open', 'high', 'low', 'close']},
                         index=pd.date_range('2024-01-01', periods=4))
    frame['volume'] = 1000.
    engine = Cerebro(match_mode='exact', trade_on_close=False, stdstats=False)
    engine.broker.setcash(10000)
    for i in range(symbol_count):
        engine.adddata(frame.copy(), name=str(i))
    engine.addstrategy(Target)
    strategy = engine.run()[0]
    assert observed == [0]
    assert strategy.stats.fills['size'].tolist() == [99.0]
    assert strategy.stats.orders['status'].tolist() == ['Margin', 'Completed']
