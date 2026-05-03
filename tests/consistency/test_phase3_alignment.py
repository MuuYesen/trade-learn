
import tradelearn.engine as bt
import pandas as pd
import numpy as np

class TestStrategy(bt.Strategy):
    params = (('buy_bar', 5), ('sell_bar', 10))

    def __len__(self):
        return len(self.data)

    def next(self):
        print(f"Bar {len(self)}: close={self.data.close[0]}")
        if len(self) == self.p.buy_bar:
            print(f"BUY at bar {len(self)}")
            self.buy()
        elif len(self) == self.p.sell_bar:
            print(f"SELL at bar {len(self)}")
            self.sell()

def run_test_case(coc=False, sizer_stake=10):
    # Create fake data: 20 bars of upward trend
    dates = pd.date_range('2023-01-01', periods=20)
    data_df = pd.DataFrame({
        'open': np.linspace(100, 120, 20),
        'high': np.linspace(102, 122, 20),
        'low': np.linspace(98, 118, 20),
        'close': np.linspace(101, 121, 20),
        'volume': [1000] * 20
    }, index=dates)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_df)
    cerebro.addstrategy(TestStrategy)
    cerebro.addsizer(bt.FixedSize, stake=sizer_stake)
    
    if coc:
        cerebro.set_coc(True)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    
    strategies = cerebro.run()
    return strategies[0]

def test_coc_alignment():
    print("\nTesting Cheat-On-Close (COC)...")
    # Case 1: NO COC (Execute on NEXT bar open)
    strat_no_coc = run_test_case(coc=False)
    fills_no_coc = strat_no_coc.broker.fills_frame()
    buy_fill_no_coc = fills_no_coc.iloc[0]
    # Buy issued at bar 5, should execute at bar 6 open
    print(f"NO COC - Buy Fill Price: {buy_fill_no_coc['price']}, Date: {buy_fill_no_coc['datetime']}")
    
    # Case 2: WITH COC (Execute on SAME bar close)
    strat_coc = run_test_case(coc=True)
    fills_coc = strat_coc.broker.fills_frame()
    buy_fill_coc = fills_coc.iloc[0]
    print(f"WITH COC - Buy Fill Price: {buy_fill_coc['price']}, Date: {buy_fill_coc['datetime']}")
    
    assert buy_fill_coc['price'] != buy_fill_no_coc['price']
    assert buy_fill_coc['datetime'] < buy_fill_no_coc['datetime']

def test_sizer_alignment():
    print("\nTesting Sizer...")
    strat = run_test_case(sizer_stake=50)
    fills = strat.broker.fills_frame()
    assert abs(fills.iloc[0]['size']) == 50
    print(f"Sizer Stake 50 - Fill Size: {fills.iloc[0]['size']}")

def test_analyzer_alignment():
    print("\nTesting Analyzers...")
    strat = run_test_case()
    analysis = strat.analyzers.sharperatio.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    
    print(f"Sharpe Ratio: {analysis['sharperatio']}")
    print(f"Max Drawdown: {dd['maxdrawdown']}%")
    
    assert analysis['sharperatio'] is not None
    assert 'maxdrawdown' in dd

if __name__ == "__main__":
    test_coc_alignment()
    test_sizer_alignment()
    test_analyzer_alignment()
    print("\nAll Phase 3 alignment tests PASSED!")
