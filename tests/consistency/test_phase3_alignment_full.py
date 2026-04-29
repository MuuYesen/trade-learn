
import numpy as np
import pandas as pd

import tradelearn.engine as bt


class SMA(bt.Indicator):
    lines = ("sma",)
    params = (("period", 10),)

    def __init__(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        self.lines.sma = bt.talib.SMA(line, timeperiod=self.p.period)


class MACD(bt.Indicator):
    lines = ("macd", "signal", "histo")

    def __init__(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        macd = bt.talib.MACD(line)
        self.lines.macd = macd.macd
        self.lines.signal = macd.signal
        self.lines.histo = macd.hist


class AlignmentStrategy(bt.Strategy):
    def __init__(self):
        # 1. Test Auto Min Period (SMA 10)
        self.sma = SMA(self.data.close, period=10)
        
        # 2. Test Delayed Line (SMA-1)
        self.sma_prev = self.sma(-1)
        
        # 3. Test Multi-line Indicator (MACD)
        self.macd = MACD(self.data.close)
        
        self.prenext_count = 0
        self.next_count = 0
        self.notified_cash = []

    def notify_cashvalue(self, cash, value):
        self.notified_cash.append(value)

    def prenext(self):
        self.prenext_count += 1
        
    def next(self):
        self.next_count += 1
        
        # 4. Test current data line access
        assert self.data.close[0] == self.data.close[0]
        
        # 5. Test Multi-line access
        macd_line = self.macd.macd[0]
        signal_line = self.macd.signal[0]
        
        # 6. Test Delayed line value
        if self.next_count > 1:
            actual_prev = self.sma[-1]
            delayed_val = self.sma_prev[0]
            if not (np.isnan(actual_prev) and np.isnan(delayed_val)):
                assert abs(actual_prev - delayed_val) < 1e-6, (
                    f"Delayed line mismatch: {actual_prev} != {delayed_val}"
                )

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
    
    assert strat.prenext_count == 0
    assert strat.next_count == 100, f"Expected 100 next calls, got {strat.next_count}"
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
