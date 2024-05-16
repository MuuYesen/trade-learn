import tradelearn.trader as bt

class Signal(bt.Indicator):
    lines = ('signal',)

    def set_signal(self, signal_list: list):
        self.lines.signal.array.extend(signal_list)
