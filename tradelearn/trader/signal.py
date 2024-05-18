import tradelearn.trader as bt

import pandas as pd

class Signal(bt.Indicator):
    lines = ('signal',)
    signal_df = None

    def set_signal(self, signal_df):
        self.signal_df = signal_df

    def align_signal(self, base_line):
        signal_df = pd.merge(self.signal_df, pd.DataFrame(base_line.index), on=['date'], how='right')
        signal_df = signal_df.set_index(['date'])
        self.lines.signal.array.extend(signal_df.values.reshape(-1).tolist())
