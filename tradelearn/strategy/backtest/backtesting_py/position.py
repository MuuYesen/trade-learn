import numpy as np


class Position:
    """
    Currently held asset position, available as
    `minitrade.backtest.core.backtesting.Strategy.position` within
    `minitrade.backtest.core.backtesting.Strategy.next`.
    Can be used in boolean contexts, e.g.

        if self.position():
            ...  # we have a position, either long or short
    """

    def __init__(self, broker: '_Broker', ticker: str):
        self.__broker = broker
        self.__ticker = ticker

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades[self.__ticker])

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in cash units."""
        return sum(trade.pl for trade in self.__broker.trades[self.__ticker])

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position in percent."""
        weights = np.abs([trade.size for trade in self.__broker.trades[self.__ticker]])
        weights = weights / weights.sum()
        pl_pcts = np.array([trade.pl_pct for trade in self.__broker.trades[self.__ticker]])
        return (pl_pcts * weights).sum()

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    def close(self, portion: float = 1.):
        """
        close portion of position by closing `portion` of each active trade. See `Trade.close`.
        """
        for trade in self.__broker.trades[self.__ticker]:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ({len(self.__broker.trades[self.__ticker])} trades)>'
