
import tradelearn as bt
import pandas as pd
import numpy as np

class AlignmentStrategy(bt.Strategy):
    def __init__(self):
        # 1. Test Auto Min Period (SMA 10)
        self.sma = bt.indicators.SMA(self.data, period=10)
        
        # 2. Test Delayed Line (SMA-1)
        self.sma_prev = self.sma(-1)
        
        # 3. Test Multi-line Indicator (MACD)
        self.macd = bt.indicators.MACD(self.data)
        
        self.prenext_count = 0
        self.next_count = 0
        self.notified_cash = []

    def notify_cashvalue(self, cash, value):
        self.notified_cash.append(value)

    def prenext(self):
        self.prenext_count += 1
        
    def next(self):
        self.next_count += 1
        
        # 4. Test Data Aliases
        close_via_alias = self.data_close[0]
        close_direct = self.data.close[0]
        
        # 5. Test Multi-line access
        macd_line = self.macd.macd[0]
        signal_line = self.macd.signal[0]
        
        # 6. Test Delayed line value
        if self.next_count > 1:
            actual_prev = self.sma[-1]
            delayed_val = self.sma_prev[0]
            assert abs(actual_prev - delayed_val) < 1e-6, f"Delayed line mismatch: {actual_prev} != {delayed_val}"

        # Debug prints
        if self.next_count == 1:
            print(f"First next() at bar {len(self.data)}")
            print(f"MACD[0]: {macd_line:.4f}, Signal[0]: {signal_line:.4f}")

def test_full_alignment():
    # 100 bars of data
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'open': np.linspace(100, 200, 100),
        'high': np.linspace(101, 201, 100),
        'low': np.linspace(99, 199, 100),
        'close': np.linspace(100.5, 200.5, 100),
        'volume': [100] * 100
    }, index=dates)

    cerebro = bt.Cerebro()
    data = bt.DataFeed(df)
    cerebro.adddata(data)
    cerebro.addstrategy(AlignmentStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Drawdown)
    
    strategies = cerebro.run()
    strat = strategies[0]
    
    # Assertions
    print(f"Prenext count: {strat.prenext_count}")
    print(f"Next count: {strat.next_count}")
    
    # EMA 26 + EMA 9 - 1 = 34. index 33 is the 34th bar.
    # So prenext should be 33.
    assert strat.prenext_count == 33, f"Expected 33 prenext calls, got {strat.prenext_count}"
    assert strat.next_count == 100 - 33, f"Expected {100-33} next calls, got {strat.next_count}"
    assert len(strat.notified_cash) == 100, "notify_cashvalue should be called every bar"
    
    # Analyzer Tests
    sharpe_analysis = strat.analyzers['sharperatio'].get_analysis()
    drawdown_analysis = strat.analyzers['drawdown'].get_analysis()
    
    print(f"Sharpe Analysis: {sharpe_analysis}")
    print(f"Drawdown Analysis: {drawdown_analysis}")
    
    assert "sharperatio" in sharpe_analysis, "Sharpe ratio missing from analysis"
    assert "maxdrawdown" in drawdown_analysis, "Max drawdown missing from analysis"

    print("All alignment tests PASSED!")

if __name__ == "__main__":
    test_full_alignment()
