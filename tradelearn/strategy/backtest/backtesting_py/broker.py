import warnings
from math import copysign
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .libs._util import _Data
from .order import Order
from .trade import Trade
from .position import Position
from .allocation import Allocation


class _OutOfMoneyError(Exception):
    pass


class Broker:
    def __init__(self, *, data: _Data, cash, holding, spread, commission, margin, trade_on_close, hedging, exclusive_orders,
                 trade_start_date, lot_size, fail_fast, storage):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data = data
        self._cash = cash
        self._holding = holding
        if callable(commission):
            self._commission = commission
        else:
            try:
                self._commission_fixed, self._commission_relative = commission
            except TypeError:
                self._commission_fixed, self._commission_relative = 0, commission
            assert self._commission_fixed >= 0, 'Need fixed cash commission in $ >= 0'
            assert -.1 <= self._commission_relative < .1, \
                ("commission should be between -10% "
                 f"(e.g. market-maker's rebates) and 10% (fees), is {self._commission_relative}")
            self._commission = self._commission_func

        self._spread = spread
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders
        self._trade_start_date = trade_start_date   # datetime with no tz
        self._lot_size = lot_size
        self._fail_fast = fail_fast
        self._storage = storage

        self._equity = np.tile(np.nan, (len(data.index), len(data.tickers)+2))
        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {ticker: [] for ticker in self._data.tickers}
        self._trade_start_bar = min(
            (self._data.index.tz_localize(None) < self._trade_start_date).sum(),
            len(self._data)-1) if self._trade_start_date else 0
        # Handle preexisting positions as if they are acquired on the first bar but
        # at the close price of trade_start_date, so that the portfolio return is 0
        # between backtest start date and trade_start_date.
        if self._holding:
            for ticker, size in self._holding.items():
                if size:
                    self.trades[ticker].append(Trade(self, ticker=ticker, size=size, entry_price=self._data[
                        ticker, 'close'][self._trade_start_bar], entry_bar=0, tag='preexisting'))
                    # add the cost for preexisting positions to initial cash
                    self._cash += size * self._data[ticker, 'close'][self._trade_start_bar]
        self.positions: Dict[str, Position] = {ticker: Position(self, ticker) for ticker in self._data.tickers}
        self.closed_trades: List[Trade] = []

    def _commission_func(self, order_size, price):
        return self._commission_fixed + abs(order_size) * price * self._commission_relative
    
    def __repr__(self):
        pos = ','.join([f'{k}:{p.size}' for k, p in self.positions.items()])
        return f'<Broker: margin_available:{self.margin_available:.0f},{pos} ({len(self.all_trades)} trades)>'

    def rebalance(self, alloc: Allocation, force: bool = False, rtol: float = 0.01, atol: int = 0, cash_reserve: float = 0.1):
        assert 0 <= cash_reserve < 1, "cash_reserve should be between 0 and 1"
        assert 0 <= rtol < 1, "rtol should be between 0 and 1"
        assert 0 <= atol, "atol should be non-negative"

        # ignore any trade actions before trade_start_date
        if self._trade_start_date and self.now.replace(tzinfo=None) < self._trade_start_date:
            alloc._clear()
            return
        # rebalance if force rebalance is true or portfolio weights have changed
        if force or alloc.modified:
            # money value of current portfolio
            total_equity = self.equity()
            # desired values for each ticker excluding cash reserve that is not to be allocated
            value_allocation = alloc.weights * total_equity * (1 - cash_reserve)
            # calculate the amount to buy or sell
            current_value = pd.Series([self.equity(ticker)
                                       for ticker in self._data.tickers], index=self._data.tickers)
            value_diff = value_allocation - current_value
            value_diff_abs = value_diff.abs().sum()
            value_diff_rel = value_diff_abs / total_equity
            # sort in ascending order so that sell orders are placed first then buy orders to make sure that cash
            # balance is always positive in simulation
            for ticker in value_diff.sort_values().index:
                if alloc.weights.loc[ticker] == 0:
                    # this may generate multiple orders for the same ticker if multiple long positions are opened
                    # for the same ticker previously over time
                    for trade in self.trades[ticker]:
                        trade.close()
                else:
                    # rebalance if the current value deviate too much from the desired value
                    # this is to avoid tiny orders triggered by ticker price fluctuation
                    if value_diff[ticker] and (atol and value_diff_abs > atol or value_diff_rel > rtol):
                        # calculate number of shares to buy respecting lot_size
                        # implicitly this forces order in whole share, fractional share not supported for now
                        size = value_diff[ticker] // self.last_price(ticker) // self._lot_size * self._lot_size
                        if size != 0:
                            self.new_order(ticker=ticker, size=size)
        alloc._next()

    def new_order(self,
                  ticker: str,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None):
        """
        Argument size indicates whether the order is long or short
        """
        ticker = ticker or self._data.the_ticker

        # ignore any trade actions before trade_start_date
        if self._trade_start_date and self.now.replace(tzinfo=None) < self._trade_start_date:
            return

        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        adjusted_price = self._adjusted_price(ticker, size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})")

        order = Order(self, ticker, size, limit, stop, sl, tp, trade, self.now, tag=tag)
        # Put the new order in the order queue,
        # inserting SL/TP/trade-closing orders in-front
        if trade:
            self.orders.insert(0, order)
        else:
            # If exclusive orders (each new order auto-closes previous orders/position),
            # cancel all non-contingent orders and close all open trades beforehand
            if self._exclusive_orders:
                for o in self.orders:
                    if not o.is_contingent:
                        o.cancel()
                for t in self.trades[ticker]:
                    t.close()

            self.orders.append(order)

        return order

    def last_price(self, ticker) -> float:
        """ Price at the last (current) close. """
        return self._data[ticker, 'close'][-1]

    def _adjusted_price(self, ticker: str, size=None, price=None) -> float:
        """
        Long/short `price`, adjusted for spread.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or self.last_price(ticker)) * (1 + copysign(self._spread, size))

    def equity(self, ticker: str = None) -> float:
        if ticker:
            # return current value of the asset
            return sum(trade.value for trade in self.trades[ticker])
        else:
            return self._cash + sum(trade.pl for trade in self.all_trades)

    @property
    def margin_available(self) -> float:
        # From https://github.com/QuantConnect/Lean/pull/3768
        margin_used = sum(abs(trade.value) / self._leverage for trade in self.all_trades)
        return max(0, self.equity() - margin_used)

    @property
    def all_trades(self) -> List[Trade]:
        return [trade for _, trades in self.trades.items() for trade in trades]

    @property
    def now(self):
        return self._data.now

    def finalize(self):
        # Ignore any unprocessed orders in broker.orders since they don't have chance
        # to be executed before the end of backtest. This is not strictly
        # true since market order can still execute if trade_on_close=True.
        # But we ignore this since it won't affect the strategy performance.

        # close any remaining open trades so they produce some stats
        final_orders = [trade.close(finalize=True) for trade in self.all_trades]
        for order in final_orders:
            price = self.last_price(order.ticker)
            time_index = len(self._data) - 1
            trade = order.parent_trade
            _prev_size = trade.size
            size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
            if trade in self.trades[order.ticker]:
                self._reduce_trade(trade, price, size, time_index)
                assert order.size != -_prev_size or trade not in self.trades[order.ticker]

    def next(self):
        i = len(self._data) - 1
        self._process_orders()

        # Log account equity for the equity curve
        total_equity = self.equity()
        ticker_equity = [self.equity(ticker) for ticker in self._data.tickers]
        equity = [total_equity, *ticker_equity, self.margin_available]
        self._equity[i] = equity

        # If equity is negative, set all to 0 and stop the simulation
        if equity[0] <= 0:
            assert self.margin_available <= 0
            for trade in self.all_trades:
                self._close_trade(trade, self.last_price(trade.ticker), i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError

    def _process_orders(self):
        i = len(self._data) - 1
        reprocess_orders = False

        # Process orders
        for order in list(self.orders):  # type: Order

            data = self._data
            open_, high, low = (
                data[order.ticker, 'open'][-1],
                data[order.ticker, 'high'][-1],
                data[order.ticker, 'low'][-1])
            prev_close = data[order.ticker, 'close'][-2]

            # Related SL/TP order was already removed
            if order not in self.orders:
                continue

            # Check if stop condition was hit
            stop_price = order.stop
            if stop_price:
                is_stop_hit = ((high > stop_price) if order.is_long else (low < stop_price))
                if not is_stop_hit:
                    continue

                # > When the stop price is reached, a stop order becomes a market/limit order.
                # https://www.sec.gov/fast-answers/answersstopordhtm.html
                order._replace(stop_price=None)

            # Determine purchase price.
            # Check if limit order can be filled.
            if order.limit:
                is_limit_hit = low < order.limit if order.is_long else high > order.limit
                # When stop and limit are hit within the same bar, we pessimistically
                # assume limit was hit before the stop (i.e. "before it counts")
                is_limit_hit_before_stop = (is_limit_hit and
                                            (order.limit < (stop_price or -np.inf)
                                             if order.is_long
                                             else order.limit > (stop_price or np.inf)))
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                # stop_price, if set, was hit within this bar
                price = (min(stop_price or open_, order.limit)
                         if order.is_long else
                         max(stop_price or open_, order.limit))
            else:
                # Market-if-touched / market order
                price = prev_close if self._trade_on_close else open_
                price = (max(price, stop_price or -np.inf)
                         if order.is_long else
                         min(price, stop_price or np.inf))

            # Determine entry/exit bar index
            is_market_order = not order.limit and not stop_price
            time_index = (i - 1) if is_market_order and self._trade_on_close else i

            # If order is a SL/TP order, it should close an existing trade it was contingent upon
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                # If order.size is "greater" than trade.size, this order is a trade.close()
                # order and part of the trade was already closed beforehand

                adjusted_price = self._adjusted_price(order.ticker, order.size, price)  #
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                # If this trade isn't already closed (e.g. on multiple `trade.close(.5)` calls)
                if trade in self.trades[order.ticker]:
                    self._reduce_trade(trade, adjusted_price, size, time_index)  #
                    # self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades[order.ticker]
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders  # Removed when trade was closed
                else:
                    # It's a trade.close() order, now done
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # Else this is a stand-alone trade

            # Adjust price to include commission (or bid-ask spread).
            # In long positions, the adjusted price is a fraction higher, and vice versa.
            adjusted_price = self._adjusted_price(order.ticker, order.size, price)
            adjusted_price_plus_commission = \
                adjusted_price + self._commission(order.size, price) / abs(order.size)

            # If order size was specified proportionally,
            # precompute true size in units, accounting for margin and spread/commissions
            size = order.size
            if -1 < size < 1:
                size = copysign(int((self.margin_available * self._leverage * abs(size))
                                    // adjusted_price_plus_commission), size)
                # Not enough cash/margin even for a single unit
                if not size:
                    # XXX: The order is canceled by the broker?
                    self.orders.remove(order)
                    continue
                else:
                    # replace relative size with calculated size
                    order.size = int(size)
            assert size == round(size)
            need_size = int(size)

            if not self._hedging:
                # Fill position by FIFO closing/reducing existing opposite-facing trades.
                # Existing trades are closed at unadjusted price, because the adjustment
                # was already made when buying.
                for trade in list(self.trades[order.ticker]):
                    if trade.is_long == order.is_long:
                        continue
                    assert trade.size * order.size < 0

                    # Order size greater than this opposite-directed existing trade,
                    # so it will be closed completely
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        # The existing trade is larger than the new order,
                        # so it will only be closed partially
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0

                    if not need_size:
                        break

            # If we don't have enough liquidity to cover for the order, abort the backtest
            if abs(need_size) * adjusted_price_plus_commission > self.margin_available * self._leverage:
                if self._fail_fast:
                    raise RuntimeError(
                        f'Not enough liquidity for {order}, has {int(self.margin_available * self._leverage)},'
                        f' needs {int(abs(need_size) * adjusted_price)}, aborting')
                else:
                    self.orders.remove(order)
                    continue

            # open a new trade
            if need_size:
                self._open_trade(order.ticker, adjusted_price, need_size, order.sl, order.tp, time_index, order.tag)

                # We need to reprocess the SL/TP orders newly added to the queue.
                # This allows e.g. SL hitting in the same bar the order was open.
                # See https://github.com/kernc/backtesting.py/issues/119
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif (low <= (order.sl or -np.inf) <= high or
                          low <= (order.tp or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning)

            # Order processed
            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # Reduce existing trade ...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ... by closing a reduced copy of it
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades[trade.ticker].append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.trades[trade.ticker].remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self._cash += trade.pl - self._commission(trade.size, price)

    def _open_trade(self, ticker: str, price: float, size: int,
                    sl: Optional[float], tp: Optional[float], time_index: int, tag):
        trade = Trade(self, ticker, size, price, time_index, tag)
        self.trades[ticker].append(trade)
        # Apply broker commission at trade open
        self._cash -= self._commission(size, price)
        # Create SL/TP (bracket) orders.
        # Make sure SL order is created first so it gets adversarially processed before TP order
        # in case of an ambiguous tie (both hit within a single bar).
        # Note, sl/tp orders are inserted at the front of the list, thus order reversed.
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl