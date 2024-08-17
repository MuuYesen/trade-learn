import sys
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .libs._util import _as_str, _Data, try_
from .allocation import Allocation
from .broker import Broker


class Strategy(ABC):
    """
    A trading strategy base class. Extend this class and
    override methods
    `minitrade.backtest.core.backtesting.Strategy.init` and
    `minitrade.backtest.core.backtesting.Strategy.next` to define
    your own strategy.
    """

    def __init__(self, broker, data, params):
        self._indicators = []
        self._broker: Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)
        self._alloc = Allocation(data.tickers)
        self._data_index = data.index.copy()
        self._records = {}
        self._start_on_day = 0

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
                                                        map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with.")
            setattr(self, k, v)
        return params

    def I(self,  # noqa: E743
          funcval: Union[pd.DataFrame, pd.Series, Callable], *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          ** kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Declare an indicator. An indicator is just an array of values,
        but one that is revealed gradually in
        `minitrade.backtest.core.backtesting.Strategy.next` much like
        `minitrade.backtest.core.backtesting.Strategy.data` is.
        Returns DataFrame in `init()` and `np.ndarray` of indicator values in `next()`.

        `funcval` is either a function that returns the indicator array(s) of
        same length as `minitrade.backtest.core.backtesting.Strategy.data`, or
        the indicator array(s) itself as a DataFrame, Series, or arrays.

        In the plot legend, the indicator is labeled with function name,
        DataFrame column name, or Series name, unless `name` overrides it.

        If `plot` is `True`, the indicator is plotted on the resulting
        `minitrade.backtest.core.backtesting.Backtest.plot`.

        If `overlay` is `True`, the indicator is plotted overlaying the
        price candlestick chart (suitable e.g. for moving averages).
        If `False`, the indicator is plotted standalone below the
        candlestick chart.

        `color` can be string hex RGB triplet or X11 color name.
        By default, the next available color is assigned.

        If `scatter` is `True`, the plotted indicator marker will be a
        circle instead of a connected line segment (default).

        Additional `*args` and `**kwargs` are passed to `func` and can
        be used for parameters.

        For example, using simple moving average function from TA-Lib:

            def init():
                self.sma = self.I(ta.SMA, self.data.close, self.n_sma)
        """
        if callable(funcval):
            if name is None:
                params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
                func_name = _as_str(funcval)
                name = (f'{func_name}({params})' if params else f'{func_name}')
            else:
                name = name.format(*map(_as_str, args),
                                   **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))))
            try:
                value = funcval(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f'Indicator "{funcval}" error') from e
        else:
            value = funcval

        if isinstance(value, (pd.DataFrame, pd.Series)):
            if not value.index.equals(self._data.index):
                raise ValueError(
                    'Indicators of pd.DataFrame or pd.Series must have the same index as'
                    f' `data` (data shape: {len(self._data)}; indicator shape: {len(value)}.\n'
                    f'`data` index: {self._data.index}\n'
                    f'Indicator index: {value.index}\n')
            value = value.copy()
        else:
            if value is not None:
                value = try_(lambda: np.asarray(value, order='C'), None)
            is_arraylike = bool(value is not None and value.shape)

            # Optionally flip the array if the long side of array is not on the 1st dimension
            if is_arraylike and np.argmin(value.shape) == 0:
                value = value.T

            if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[0] != len(self._data):
                raise ValueError(
                    'Indicators of numpy.ndarray must have the same '
                    f'length as `data` (data shape: {len(self._data)}; indicator "{name}" '
                    f'shape: {getattr(value, "shape" , "")}, returned value: {value})')
            elif value.ndim == 1:
                value = pd.Series(value, index=self._data.index, name=name)
            else:
                value = pd.DataFrame(value, index=self._data.index)

        # Use an experimental feature to save DataFrame/Series metadata
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html
        value.attrs.update({'name': name, 'plot': plot, 'overlay': overlay,
                           'color': color, 'scatter': scatter, **kwargs})
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `minitrade.backtest.core.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `minitrade.backtest.core.backtesting.lib`,
        make sure to call: `super().init()`
        """

    @abstractmethod
    def next(self):
        """
        Main strategy runtime method, called as each new
        `minitrade.backtest.core.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `minitrade.backtest.core.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `minitrade.backtest.core.backtesting.lib`,
        make sure to call: `super().next()`
        """

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(self, *,
            ticker: str = None,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None):
        """
        Place a new long order. For explanation of parameters, see `Order` and its properties.

        For single asset strategy, `ticker` can be left as None.

        See `Position.close()` and `Trade.close()` for closing existing positions.

        See also `Strategy.sell()`.
        """
        assert 0 < size < 1 or round(size) == size, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(ticker, size, limit, stop, sl, tp, tag)

    def sell(self, *,
             ticker: str = None,
             size: float = _FULL_EQUITY,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None):
        """
        Place a new short order. For explanation of parameters, see `Order` and its properties.

        For single asset strategy, `ticker` can be left as None.

        See also `Strategy.buy()`.

        .. note::
            If you merely want to close an existing long position,
            use `Position.close()` or `Trade.close()`.
        """
        assert 0 < size < 1 or round(size) == size, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(ticker, -size, limit, stop, sl, tp, tag)

    def rebalance(self, force: bool = False, rtol: float = 0.01, atol: int = 0, cash_reserve: float = 0.1):
        """
        Rebalance the portfolio according to the current weight allocation.

        If the weight allocation is not changed from the previous cycle, the rebalance is skipped. This behavior can be
        overridden by setting `force` to `True`, which will force rebalance even if the weight allocation is unchanged.
        This is useful when the actual portfolio value deviates from the target value due to price changes and should
        be corrected.

        When a rebalance should be performed, the difference between the target and actual portfolio, defined as the sum
        of absolute difference of individual assets, is calculated. If the difference is too small compared to the
        relative tolerance `rtol` or the absolute tolerance `atol`, the rebalance is again skipped. This can be used
        to avoid unnecessary transaction cost. An exception is when the target weight of an asset is zero, in which case
        the position of the asset, if exists, is always closed.

        `cash_reserve` is the ratio of total equity reserved as cash to account for order quantity rounding and sudden
        price changes between order placement and execution. It is recommended to set this value to a small positive
        number to avoid order rejection due to insufficient cash. The minimum value may depend on the volatility of the
        assets.

        Args:
            force: If True, rebalance will be performed even if the current weight allocation
                is not changed from the previous.
            rtol: Relative tolerance of the total absolute value difference between current
                and previous allocation vs. total portfolio value. If the difference is smaller
                than `rtol`, rebalance will not be performed.
            atol: Absolute tolerance of the total absolute value difference between current
                and previous allocation. If the difference is smaller than `atol`, rebalance
                will not be performed.
            cash_reserve: Ratio of total equity reserved as cash to account for order
                quantity rounding and sudden price changes between order placement and
                execution.
        """
        self._broker.rebalance(alloc=self._alloc, force=force, rtol=rtol, atol=atol, cash_reserve=cash_reserve)

    def record(self, name: str = None, plot: bool = True, overlay: bool = None, color: str = None, scatter: bool = False, **kwargs):
        """
        Record arbitrary key-value pairs as time series. This can be used for diagnostic
        data collection or for plotting custom data.

        Values to be recorded should be passed as keyword arguments. The value can be a scalar, a dictionary, or a
        pandas Series. If a dictionary or a Series is passed, its keys will be used as names for time series.

        Example:
        ```python
        # Record a scalar value
        self.record(my_key=42)

        # Record a dictionary
        self.record(my_dict={'a': 1, 'b': 2})

        # Record a pandas Series
        self.record(my_series=pd.Series({'a': 1, 'b': 2}))
        ```

        Args:
            name: Name of the time series. If not provided, the name will be the same as the keyword argument.
            plot: If True, the time series will be plotted on the resulting `minitrade.backtest.core.backtesting.Backtest.plot`.
            overlay: If True, the time series will be plotted overlaying the price candlestick chart. If False, the time series
                will be plotted standalone below the candlestick chart.
            color: Color of the time series. If not provided, the next available color will be assigned.
            scatter: If True, the plotted time series marker will be a circle instead of a connected line segment.
        """
        for k, v in kwargs.items():
            if isinstance(v, dict) or isinstance(v, pd.Series):
                v = dict(v)
                if k not in self._records:
                    self._records[k] = pd.DataFrame(index=self._data_index, columns=v.keys())
                self._records[k].loc[self._broker.now, list(v.keys())] = list(v.values())
            else:
                if k not in self._records:
                    self._records[k] = pd.Series(index=self._data_index)
                self._records[k].iloc[len(self._data)-1] = v
            self._records[k].name = name or k
            self._records[k].attrs.update({'name': name or k, 'plot': plot, 'overlay': overlay,
                                           'color': color, 'scatter': scatter})

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity()

    @property
    def data(self) -> _Data:
        """
        Price data, roughly as passed into
        `minitrade.backtest.core.backtesting.Backtest.__init__`,
        but with two significant exceptions:

        * `data` is _not_ a DataFrame, but a custom structure
          that serves customized numpy arrays for reasons of performance
          and convenience. Besides OHLCV columns, `.index` and length,
          it offers `.pip` property, the smallest price unit of change.
        * Within `minitrade.backtest.core.backtesting.Strategy.init`, `data` arrays
          are available in full length, as passed into
          `minitrade.backtest.core.backtesting.Backtest.__init__`
          (for precomputing indicators and such). However, within
          `minitrade.backtest.core.backtesting.Strategy.next`, `data` arrays are
          only as long as the current iteration, simulating gradual
          price point revelation. In each call of
          `minitrade.backtest.core.backtesting.Strategy.next` (iteratively called by
          `minitrade.backtest.core.backtesting.Backtest` internally),
          the last array value (e.g. `data.close[-1]`)
          is always the _most recent_ value.
        * If you need data arrays (e.g. `data.close`) to be indexed
          **Pandas Series or DataFrame**, you can call their `.df` accessor
          (e.g. `data.close.df`). If you need the whole of data
          as a **DataFrame**, use `.df` accessor (i.e. `data.df`).
        """
        return self._data

    @property
    def storage(self) -> dict | None:
        """Storage is a dictionary for saving custom data across backtest runs
        when used in the context of automated trading in incremental mode.

        If backtest finishes successfully, any modification to the dictionary
        is persisted and can be accessed in future runs. If backtest fails due
        to any error, the modification is not saved. If backtest runs in dryrun
        mode, the modification is not saved.

        No storage is provided when trading in "strict" mode, in which case `storage`
        is None.
        """
        return self._broker._storage

    def position(self, ticker: str = None) -> 'Position':
        """Instance of `minitrade.backtest.core.backtesting.Position`.

        For single asset strategy, `ticker` can be left as None, which returns
        the position of the only asset.
        """
        ticker = ticker or self._data.the_ticker
        return self._broker.positions[ticker]

    @property
    def orders(self) -> 'List[Order]':
        """List of orders (see `Order`) waiting for execution."""
        return self._broker.orders

    def trades(self, ticker: str = None) -> 'Tuple[Trade, ...]':
        """List of active trades (see `Trade`)."""
        return tuple(self._broker.trades[ticker] if ticker else self._broker.all_trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """List of settled trades (see `Trade`)."""
        return tuple(self._broker.closed_trades)

    @property
    def alloc(self) -> Allocation:
        """`Allocation` instance that manages the weight allocation among assets."""
        return self._alloc

    def start_on_day(self, n: int):
        """Hint to start the backtest on a specific day.

        This can be used to define a warm-up period, ensuring at least `n` days of data
        are available when `next()` is called for the first time.

        When the backtest starts depends both on `n` and on the availability of indicators.
        If indicators are defined, the backtest will start when all indicators have
        valid data or on the `n`-th day, whichever comes later.

        This method should be called in `init()`.

        Args:
            n: Day index to start on. Must be within [0, len(data)-1].
        """
        assert 0 <= n < len(self._data), f"day must be within [0, {len(self._data)-1}]"
        self._start_on_day = n

    @classmethod
    def prepare_data(cls, tickers: 'List[str]', start: str) -> pd.DataFrame | None:
        """Prepare data for trading.

        This class method can be overridden in a `Strategy` implementation to provide
        data for trading. The can be useful when the data is not provided externally
        and the strategy wants to bring its own data, e.g. from a database.

        Args:
            tickers: List of tickers to fetch data for.
            start: Start date of the data to fetch.

        Returns:
            A `pd.DataFrame` with 2-level columns as required by `Backtest()` or None.
        """
        return None