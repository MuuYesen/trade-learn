import multiprocessing as mp
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache, partial
from itertools import compress, product, repeat
from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq

from .backtesting_py.libs._plotting import plot
from .backtesting_py.libs._stats import compute_stats
from .backtesting_py.libs._util import _Data, _Indicator
from .backtesting_py.strategy import Strategy
from .backtesting_py.broker import Broker
from .backtesting_py.order import Order

class _OutOfMoneyError(Exception):
    pass

class Backtest:
    """
    Backtest a particular (parameterized) strategy
    on particular data.

    Upon initialization, call method
    `minitrade.backtest.core.backtesting.Backtest.run` to run a backtest
    instance, or `minitrade.backtest.core.backtesting.Backtest.optimize` to
    optimize it.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 10_000,
                 holding: dict = {},
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 trade_start_date=None,
                 lot_size=1,
                 fail_fast=True,
                 storage: dict | None = None,
                 ):
        """
        Initialize a backtest. Requires data and a strategy to test.

        `data` is a `pd.DataFrame` with 2-level columns:
        1st level is a list of tickers, and
        2nd level is `open`, `high`, `low`, `close`, and `volume`.
        If the strategy works only on one asset, the 1st level can be dropped.
        If any columns are missing, set them to what you have available,
        e.g.

            df['open'] = df['high'] = df['low'] = df['close']
            df['volumn'] = 0

        The passed data frame can contain additional columns that
        can be used by the strategy (e.g. sentiment info).
        DataFrame index can be either a datetime index (timestamps)
        or a monotonic range index (i.e. a sequence of periods).

        `strategy` is a `minitrade.backtest.core.backtesting.Strategy`
        _subclass_ (not an instance).

        `cash` is the initial cash to start with.

        `holding` is a mapping of preexisting assets and their sizes before
        backtest begins, e.g.

            {'AAPL': 10, 'MSFT': 5}

        `spread` is the the constant bid-ask spread rate (relative to the price).
        E.g. set it to `0.0002` for commission-less forex
        trading where the average spread is roughly 0.2‰ of the asking price.

        `commission` is the commission rate. E.g. if your broker's commission
        is 1% of order value, set commission to `0.01`.
        The commission is applied twice: at trade entry and at trade exit.
        Besides one single floating value, `commission` can also be a tuple of floating
        values `(fixed, relative)`. E.g. set it to `(100, .01)`
        if your broker charges minimum $100 + 1%.
        Additionally, `commission` can be a callable
        `func(order_size: int, price: float) -> float`
        (note, order size is negative for short orders),
        which can be used to model more complex commission structures.
        Negative commission values are interpreted as market-maker's rebates.

        .. note::
            Before v0.4.0, the commission was only applied once, like `spread` is now.
            If you want to keep the old behavior, simply set `spread` instead.

        `margin` is the required margin (ratio) of a leveraged account.
        No difference is made between initial and maintenance margins.
        To run the backtest using e.g. 50:1 leverge that your broker allows,
        set margin to `0.02` (1 / leverage).

        If `trade_on_close` is `True`, market orders will be filled
        with respect to the current bar's closing price instead of the
        next bar's open.

        If `hedging` is `True`, allow trades in both directions simultaneously.
        If `False`, the opposite-facing orders first close existing trades in
        a [FIFO] manner.

        If `exclusive_orders` is `True`, each new order auto-closes the previous
        trade/position, making at most a single trade (long or short) in effect
        at each time.

        If `trade_start_date` is not None, orders generated before the date are
        surpressed and ignored in backtesting.

        `lot_size` is the minimum increment of shares you buy in one order. Order
        size will be rounded to integer multiples during rebalance.

        `fail_fast`, when True, instructs the backtester to bail out when
        cash is not enough to cover an order. This can be used in live trading
        to detect issues early. If False, backtesting will ignore the order and
        continue, which can be convenient during algorithm research.

        `storage`, when not None, is a dictionary that contains saved states from
        past runs. Modification to storage is persisted and can be made available
        for future runs.

        [FIFO]: https://www.investopedia.com/terms/n/nfa-compliance-rule-2-43b.asp
        """

        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float value, percent of '
                            'entry order price')
        if not isinstance(commission, (Number, tuple)) and not callable(commission):
            raise TypeError('`commission` must be a float percent of order value, '
                            'a tuple of `(fixed, relative)` commission, '
                            'or a function that takes `(order_size, price)`'
                            'and returns commission dollar value')

        data = data.copy(deep=False)
        ohlc = ['open', 'high', 'low', 'close']
        ohlcv = ['open', 'high', 'low', 'close'] + ['volume']

        # Convert single asset data into 2-level column index
        if data.columns.nlevels == 1:
            data.columns = pd.MultiIndex.from_product([['Asset'], data.columns])

        # Convert index to datetime index
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            # Numeric index with most large numbers
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass
        if not set(data.columns.levels[1]).issuperset(set(ohlc)):
            raise ValueError("`data` must be a pandas.DataFrame containing columns 'open', 'high', 'low', 'close'")
        if len(data) == 0:
            raise ValueError("`data` cannot be empty")
        if np.any(data.xs('close', axis=1, level=1) > cash):
            warnings.warn('Some prices are larger than initial cash value. Note that fractional '
                          'trading is not supported. If you want to trade Bitcoin, '
                          'increase initial cash, or trade μBTC or satoshis instead (GH-134).',
                          stacklevel=2)
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.',
                          stacklevel=2)
            data = data.sort_index()
        if data.loc[:, (slice(None), ohlc)].apply(lambda s: s.loc[s.first_valid_index():].isna().sum()).sum() > 0:
            raise ValueError('Some OHLC values are missing (NaN). '
                             'Please strip those lines with `df.dropna()` or '
                             'fill them in with `df.interpolate()` or whatever.')
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.',
                          stacklevel=2)
        data.index.name = 'date'

        self._data = data
        self._broker = partial(
            Broker, cash=cash, holding=holding, spread=spread, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders,
            trade_start_date=datetime.strptime(trade_start_date, '%Y-%m-%d') if trade_start_date else None,
            lot_size=lot_size, fail_fast=fail_fast, storage=storage,
        )
        self._strategy = strategy
        self._results: Optional[pd.Series] = None

        # equal weighed average, as if buy and hold an equal weighed portfolio
        weights = 1 / self._data.xs('close', axis=1, level=1).iloc[0]
        weighted_data = self._data.copy()
        weighted_data = weighted_data.loc[:, (slice(None), ohlcv)]
        for ticker in weights.index:
            weighted_data[ticker] = weighted_data[ticker] * weights[ticker]
        weighted_data = weighted_data.T.groupby(level=1).agg('sum').T / weights.sum()
        self._ohlcv_ref_data = weighted_data

    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                     93.9944
            Equity Final [$]                      51959.9
            Equity Peak [$]                       75787.4
            Return [%]                            419.599
            Buy & Hold Return [%]                 703.458
            Return (Ann.) [%]                      21.328
            Volatility (Ann.) [%]                 36.5383
            Sharpe Ratio                         0.583718
            Sortino Ratio                         1.09239
            Calmar Ratio                         0.444518
            Max. Drawdown [%]                    -47.9801
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   65
            Win Rate [%]                          46.1538
            Best Trade [%]                         53.596
            Worst Trade [%]                      -18.3989
            Avg. Trade [%]                        2.35371
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.08802
            Expectancy [%]                        8.79171
            SQN                                  0.916893
            Kelly Criterion                        0.6134
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            _orders                              Ticke...
            _positions                           {'GOO...
            _trade_start_bar                           0
            dtype: object

        .. warning::
            You may obtain different results for different strategy parameters.
            E.g. if you use 50- and 200-bar SMA, the trading simulation will
            begin on bar 201. The actual length of delay is equal to the lookback
            period of the `Strategy.I` indicator which lags the most.
            Obviously, this can affect results.
        """
        data = _Data(self._data.copy(deep=False))
        broker: Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)
        processed_orders: List[Order] = []
        final_positions = None

        try:
            strategy.init()
        except Exception as e:
            print('Strategy initialization throws exception', e)
            print(traceback.format_exc())
            return

        # Indicators used in Strategy.next()
        indicator_attrs = {attr: indicator for attr, indicator in strategy.__dict__.items()
                           if any([indicator is item for item in strategy._indicators])}

        # Skip first few candles where indicators are still "warming up"
        start = max((indicator.isna().any(axis=1).argmin() if isinstance(indicator, pd.DataFrame)
                     else indicator.isna().argmin() for indicator in indicator_attrs.values()), default=0)
        start = max(start, strategy._start_on_day)

        # Preprocess indicators to numpy array for better performance
        def deframe(df): return df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(df.columns) == 1 else df
        indicator_attrs_np = {attr: deframe(indicator).to_numpy() for attr, indicator in indicator_attrs.items()}

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid='ignore'):

            for i in range(start, len(self._data)):
                # Prepare data and indicators for `next` call
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs_np.items():
                    setattr(strategy, attr,
                            _Indicator(
                                array=indicator[: i + 1],
                                df=partial(_Indicator.lazy_indexing, indicator_attrs[attr], i + 1)))

                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                strategy.next()

                # take note of the orders generated
                processed_orders.extend(broker.orders)
            else:

                # take note of the final positions
                final_positions = ({t: p.size for t, p in broker.positions.items()}
                                   | {'Cash': int(broker.margin_available)})

                if start < len(self._data):
                    broker.finalize()

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            data._set_length(len(self._data))

            equity = pd.DataFrame(broker._equity, index=data.index,
                                  columns=['Equity', *data.tickers, 'Cash']).bfill().fillna(broker._cash)

            self._results = compute_stats(
                orders=processed_orders,
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._ohlcv_ref_data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
                positions=final_positions,
                trade_start_bar=start,
            )
        print(self._results)

        return self._results.copy()

    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `minitrade.backtest.core.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"skopt"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: \
            https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="skopt"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'skopt'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [scikit-optimize]\
        [plotting tools].

        [OptimizeResult]: \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        [scikit-optimize]: https://scikit-optimize.github.io
        [plotting tools]: https://scikit-optimize.github.io/stable/modules/plots.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer random seed.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                              constraint=lambda p: p.sma1 < p.sma2)

        .. TODO::
            Improve multiprocessing/parallel execution on Windos with start method 'spawn'.
        """
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            stats = self._results if self._results is not None else self.run()
            if maximize not in stats:
                raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                 'result of backtest.run()')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError('`maximize` must be str (a field of backtest.run() result '
                            'Series) or a function that accepts result Series '
                            'and returns a number; the higher the better')
        assert callable(maximize), maximize

        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")
        assert callable(constraint), constraint

        if return_optimization and method != 'skopt':
            raise ValueError("return_optimization=True only valid if method='skopt'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no "
                                 f"optimization values: {k}={v}")

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = default_rng(random_state).random
            grid_frac = (1 if max_tries is None else
                         max_tries if 0 < max_tries <= 1 else
                         max_tries / _grid_size())
            param_combos = [dict(params)  # back to dict so it pickles
                            for params in (AttrDict(params)
                                           for params in product(*(zip(repeat(k), _tuple(v))
                                                                   for k, v in kwargs.items())))
                            if constraint(params)  # type: ignore
                            and rand() <= grid_frac]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 1000:
                warnings.warn(f'Searching for best of {len(param_combos)} configurations.',
                              stacklevel=2)

            heatmap = pd.Series(np.nan,
                                name=maximize_key,
                                index=pd.MultiIndex.from_tuples(
                                    [p.values() for p in param_combos],
                                    names=next(iter(param_combos)).keys()))

            def _batch(seq):
                n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
                for i in range(0, len(seq), n):
                    yield seq[i:i + n]

            # Save necessary objects into "global" state; pass into concurrent executor
            # (and thus pickle) nothing but two numbers; receive nothing but numbers.
            # With start method "fork", children processes will inherit parent address space
            # in a copy-on-write manner, achieving better performance/RAM benefit.
            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            Backtest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
            try:
                # If multiprocessing start method is 'fork' (i.e. on POSIX), use
                # a pool of processes to compute results in parallel.
                # Otherwise (i.e. on Windos), sequential computation will be "faster".
                if mp.get_start_method(allow_none=False) == 'fork':
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(Backtest._mp_task, backtest_uuid, i)
                                   for i in range(len(param_batches))]
                        for future in _tqdm(as_completed(futures), total=len(futures),
                                            desc='Backtest.optimize'):
                            batch_index, values = future.result()
                            for value, params in zip(values, param_batches[batch_index]):
                                heatmap[tuple(params.values())] = value
                else:
                    if os.name == 'posix':
                        warnings.warn("For multiprocessing support in `Backtest.optimize()` "
                                      "set multiprocessing start method to 'fork'.")
                    for batch_index in _tqdm(range(len(param_batches))):
                        _, values = Backtest._mp_task(backtest_uuid, batch_index)
                        for value, params in zip(values, param_batches[batch_index]):
                            heatmap[tuple(params.values())] = value
            finally:
                del Backtest._mp_backtests[backtest_uuid]

            best_params = heatmap.idxmax()

            if pd.isnull(best_params):
                # No trade was made in any of the runs. Just make a random
                # run so we get some, if empty, results
                stats = self.run(**param_combos[0])
            else:
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_skopt() -> Union[pd.Series,
                                       Tuple[pd.Series, pd.Series],
                                       Tuple[pd.Series, pd.Series, dict]]:
            try:
                from skopt import forest_minimize
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
                from skopt.space import Categorical, Integer, Real
                from skopt.utils import use_named_args
            except ImportError:
                raise ImportError("Need package 'scikit-optimize' for method='skopt'. "
                                  "pip install scikit-optimize") from None

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                         max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                         max_tries)

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in 'mM':  # timedelta, datetime64
                    # these dtypes are unsupported in skopt, so convert to raw int
                    # TODO: save dtype and convert back later
                    values = values.astype(int)

                if values.dtype.kind in 'iumM':
                    dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
                elif values.dtype.kind == 'f':
                    dimensions.append(Real(low=values.min(), high=values.max(), name=key))
                else:
                    dimensions.append(Categorical(values.tolist(), name=key, transform='onehot'))

            # Avoid recomputing re-evaluations:
            # "The objective has been evaluated at this point before."
            # https://github.com/scikit-optimize/scikit-optimize/issues/302
            memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

            # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
            INVALID = 1e300
            progress = iter(_tqdm(repeat(None), total=max_tries, desc='Backtest.optimize'))

            @ use_named_args(dimensions=dimensions)
            def objective_function(**params):
                next(progress)
                # Check constraints
                # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                if not constraint(AttrDict(params)):
                    return INVALID
                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', 'The objective has been evaluated at this point before.')

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                    acq_func='LCB',
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator='lhs',  # 'sobel' requires n_initial_points ~ 2**N
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap = heatmap[heatmap != -INVALID]
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                valid = res.func_vals != INVALID
                res.x_iters = list(compress(res.x_iters, valid))
                res.func_vals = res.func_vals[valid]
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method == 'skopt':
            output = _optimize_skopt()
        else:
            raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
        return output

    @ staticmethod
    def _mp_task(backtest_uuid, batch_index):
        bt, param_batches, maximize_func = Backtest._mp_backtests[backtest_uuid]
        return batch_index, [maximize_func(stats) if stats['# Trades'] else np.nan
                             for stats in (bt.run(**params)
                                           for params in param_batches[batch_index])]

    _mp_backtests: Dict[float, Tuple['Backtest', List, Callable]] = {}

    def plot(self, *, results: pd.Series = None, filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=False, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = False,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True,
             plot_allocation=False, relative_allocation=True,
             plot_indicator=True):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `minitrade.backtest.core.backtesting.Backtest.run` or
        `minitrade.backtest.core.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.
        By default, a strategy/parameter-dependent file is created in the
        current working directory.

        `plot_width` is the width of the plot in pixels. If None (default),
        the plot is made to span 100% of browser width. The height is
        currently non-adjustable.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity (initial cash plus assets) graph section. This is the same
        as `plot_return` plus initial 100%.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section. This is the same
        as `plot_equity` minus initial 100%.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `plot_trades` is `True`, the stretches between trade entries
        and trade exits are marked by hash-marked tractor beams.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times,
        unaffected by any interim asset volatility.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart. Default downsampling rule is:
        monthly for daily data, daily for hourly data, hourly for minute data,
        and minute for (sub-)second data.
        `superimpose` can also be a valid [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to superimpose.
        Note, this only works for data with a datetime index.

        If `resample` is `True`, the OHLC data is resampled in a way that
        makes the upper number of candles for Bokeh to plot limited to 10_000.
        This may, in situations of overabundant data,
        improve plot's interactive performance and avoid browser's
        `Javascript Error: Maximum call stack size exceeded` or similar.
        Equity & dropdown curves and individual trades data is,
        `resample` can also be a [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to resample, overriding above numeric limitation.
        Note, all this only works for data with a datetime index.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        [Pandas offset string]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.

        If `plot_allocation` is `True`, the resulting plot will contain
        an equity allocation graph section.

        If `relative_allocation` is `True`, scale and label equity allocation graph axis
        with return percent, not absolute cash-equivalent values.

        If `plot_indicator` is `True`, the resulting plot will contain
        a section for each indicator used in the strategy.
        """
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        indicators = results._strategy._indicators + list(results._strategy._records.values())

        return plot(
            results=results,
            data=self._data,
            baseline=self._ohlcv_ref_data,
            indicators=indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
            plot_allocation=plot_allocation,
            relative_allocation=relative_allocation,
            plot_indicator=plot_indicator)