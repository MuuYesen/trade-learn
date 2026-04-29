"""Tradelearn 1.x-style grid trading strategy for the Lite API."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradelearn.lite import Strategy


class OneXGridTrade(Strategy):
    title = "Long"
    prepare_days = 30
    distance = 0.01
    initial_position_ratio = 0.0
    stop_ratio = 0.5

    def init(self) -> None:
        warmup = self.data.close.df.iloc[: self.prepare_days]
        self.max_bound = warmup.max()
        self.min_bound = warmup.min()
        self.initial_cash = self.equity
        self.initial_enter_price = warmup.mean()
        self.initial_size = (
            self.initial_cash * self.initial_position_ratio // self.initial_enter_price
        )
        short_bounds = np.arange(
            0,
            (self.max_bound - self.initial_enter_price) / self.initial_enter_price,
            self.distance,
        )
        long_bounds = np.arange(
            0,
            (self.min_bound - self.initial_enter_price) / self.initial_enter_price,
            -self.distance,
        )
        self.band_list = (
            np.unique(np.concatenate((long_bounds, short_bounds))) + 1
        ) * self.initial_enter_price
        self.label_list = np.arange(1, len(self.band_list))
        self.initial_label = len(long_bounds)
        grid_count = max(len(short_bounds), len(long_bounds)) - 1
        self.size = (
            self.initial_cash
            * (1 - self.initial_position_ratio)
            // (self.initial_enter_price * max(1, grid_count))
        )
        self.last_grid = -1
        self.last_grid_change = None
        self.is_end = False
        self.start_on_bar(self.prepare_days)

    def next(self) -> None:
        if self.is_end:
            return

        if self.equity / self.initial_cash < self.stop_ratio:
            self.position().close()
            self.is_end = True
            return

        close = self.data.close[0]
        grid = pd.cut([close], self.band_list, labels=self.label_list)[0]
        if close > self.band_list[-1]:
            grid = len(self.band_list)
        if close < self.band_list[0]:
            grid = 0
        if self.title == "Long" and grid > self.initial_label:
            grid = self.initial_label
        if self.title == "Short" and grid < self.initial_label:
            grid = self.initial_label

        if self.last_grid == -1:
            if grid == self.initial_label and self.initial_size > 0:
                self.buy(size=self.initial_size)
            self.last_grid = self.initial_label
            return

        grid_change_new = [self.last_grid, grid]
        if grid_change_new == self.last_grid_change:
            self.last_grid = grid
            return

        grid_change = grid - self.last_grid
        if grid_change < 0:
            self.buy(size=self.size * -grid_change)
        if grid_change > 0:
            self.sell(size=self.size * grid_change)

        self.last_grid = grid
        self.last_grid_change = grid_change_new
