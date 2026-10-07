"""Charting market structure algorithm; pivots include confirmation indices.

Geometric pivot indices describe where a swing happened, not when it became
known. Entries absent from confirmations are provisional terminal points.
Run on an already truncated, closed-bar prefix for historical replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SMCResult:
    candle_set: object
    lev: list
    test_array: list
    demand_zone_values: list
    supply_zone_values: list
    bos_hh: list
    bos_ll: list
    imp_hh: list
    imp_ll: list
    confirmations: dict = field(default_factory=dict)


class SMCAnalyzer:
    def __init__(self, depth):
        self.depth = depth
        self.confirmations = {}
        self.known_at = depth
        self.lev = []
        self.test_array = []
        self.demand_zone_values = []
        self.supply_zone_values = []
        self.bos_ll = []
        self.bos_hh = []
        self.imp_ll = []
        self.imp_hh = []

    @staticmethod
    def checkcandles(candle):
        if candle["Open"] < candle["Close"]:
            return "Green"
        if candle["Open"] > candle["Close"]:
            return "Red"
        return None

    def runforlows(self, candle_set, running_index):
        bosfill = 0
        cl = 0
        final_low = candle_set["Low"][running_index]
        fl_i = running_index
        for i in range(running_index, len(candle_set) - 1):
            if len(self.lev) >= 2 and self.lev[-2][1] > candle_set["Low"][i] and bosfill != 1:
                self.breakofstructurell(candle_set, candle_set.iloc[i], i)
                bosfill = 1
            if final_low > candle_set["Low"][i] or final_low == candle_set["Low"][i]:
                final_low = candle_set["Low"][i]
                fl_i = i
                cl = 0
            else:
                cl += 1
            if cl == self.depth:
                if fl_i != self.lev[len(self.lev) - 1][0]:
                    self.lev.append((fl_i, final_low))
                    self.known_at = max(self.known_at, int(i) + 1)
                    self.confirmations[int(fl_i)] = self.known_at
                    self.test_array.append((fl_i, final_low, "LOW"))
                    return fl_i + 1
                cl -= 1
        return None

    def runforhigs(self, candle_set, running_index):
        bosfill = 0
        ch = 0
        final_high = candle_set["High"][running_index]
        fl_h = running_index
        for i in range(running_index, len(candle_set)):
            if len(self.lev) >= 2 and self.lev[-2][1] < candle_set["High"][i] and bosfill != 1:
                self.breakofstructurehh(candle_set, candle_set.iloc[i], i)
                bosfill = 1
            if final_high < candle_set["High"][i] or final_high == candle_set["High"][i]:
                final_high = candle_set["High"][i]
                fl_h = i
                ch = 0
            else:
                ch += 1
            if ch == self.depth:
                if fl_h != self.lev[len(self.lev) - 1][0]:
                    self.lev.append((fl_h, final_high))
                    self.known_at = max(self.known_at, int(i))
                    self.confirmations[int(fl_h)] = self.known_at
                    self.test_array.append((fl_h, final_high, "HIGH"))
                    return fl_h + 1
                ch -= 1
        return None

    def breakofstructurehh(self, candle_set, candle, i):
        del candle
        self.getdemandzones(candle_set, self.lev[-1][0], i)
        self.getimpulsivemovement(candle_set, self.lev[-1][0], "Buy", i)
        self.bos_hh.append((self.lev[-2][0], self.lev[-2][1], i, self.lev[-2][1]))

    def breakofstructurell(self, candle_set, candle, i):
        del candle
        self.getsupplyzones(candle_set, self.lev[-1][0], i)
        self.getimpulsivemovement(candle_set, self.lev[-1][0], "Sell", i)
        self.bos_ll.append((self.lev[-2][0], self.lev[-2][1], i, self.lev[-2][1]))

    def getimpulsivemovement(self, candle_set, scan_start_index, direction, bos_index):
        upper = min(bos_index, len(candle_set) - 2)
        if direction == "Buy":
            for i in range(scan_start_index, upper):
                if candle_set["High"][i] < candle_set["Low"][i + 2]:
                    self.imp_hh.append((i, candle_set["High"][i], i + 2, candle_set["Low"][i + 2]))
        else:
            for i in range(scan_start_index, upper):
                if candle_set["Low"][i] > candle_set["High"][i + 2]:
                    self.imp_ll.append((i, candle_set["Low"][i], i + 2, candle_set["High"][i + 2]))

    def getdemandzones(self, candle_set, bcslow_index, j):
        brcheck = 0
        if self.checkcandles(candle_set.iloc[bcslow_index]) == "Red":
            for i in range(j + 1, len(candle_set)):
                if candle_set["High"][bcslow_index] > candle_set["Low"][i] and brcheck != 1:
                    self.demand_zone_values.append(
                        (
                            bcslow_index,
                            candle_set["High"][bcslow_index],
                            i - 1,
                            candle_set["Low"][bcslow_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.demand_zone_values.append(
                    (
                        bcslow_index,
                        candle_set["High"][bcslow_index],
                        len(candle_set) - 1,
                        candle_set["Low"][bcslow_index],
                    )
                )
        elif (
            bcslow_index > 0
            and self.checkcandles(candle_set.iloc[bcslow_index]) is not None
            and self.checkcandles(candle_set.iloc[bcslow_index - 1]) == "Red"
            and brcheck != 1
        ):
            for i in range(j + 1, len(candle_set)):
                if candle_set["High"][bcslow_index - 1] > candle_set["Low"][i]:
                    self.demand_zone_values.append(
                        (
                            bcslow_index - 1,
                            candle_set["High"][bcslow_index - 1],
                            i - 1,
                            candle_set["Low"][bcslow_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.demand_zone_values.append(
                    (
                        bcslow_index - 1,
                        candle_set["High"][bcslow_index - 1],
                        len(candle_set) - 1,
                        candle_set["Low"][bcslow_index],
                    )
                )
        else:
            for i in range(j + 1, len(candle_set)):
                if candle_set["High"][bcslow_index] > candle_set["Low"][i] and brcheck != 1:
                    self.demand_zone_values.append(
                        (
                            bcslow_index,
                            candle_set["High"][bcslow_index],
                            i - 1,
                            candle_set["Low"][bcslow_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.demand_zone_values.append(
                    (
                        bcslow_index,
                        candle_set["High"][bcslow_index],
                        len(candle_set) - 1,
                        candle_set["Low"][bcslow_index],
                    )
                )

    def getsupplyzones(self, candle_set, bcshigh_index, j):
        brcheck = 0
        if self.checkcandles(candle_set.iloc[bcshigh_index]) == "Green":
            for i in range(j + 1, len(candle_set)):
                if candle_set["Low"][bcshigh_index] < candle_set["High"][i] and brcheck != 1:
                    self.supply_zone_values.append(
                        (
                            bcshigh_index,
                            candle_set["Low"][bcshigh_index],
                            i - 1,
                            candle_set["High"][bcshigh_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.supply_zone_values.append(
                    (
                        bcshigh_index,
                        candle_set["Low"][bcshigh_index],
                        len(candle_set) - 1,
                        candle_set["High"][bcshigh_index],
                    )
                )
        elif (
            bcshigh_index > 0
            and self.checkcandles(candle_set.iloc[bcshigh_index]) is not None
            and self.checkcandles(candle_set.iloc[bcshigh_index - 1]) == "Green"
        ):
            for i in range(j + 1, len(candle_set)):
                if candle_set["Low"][bcshigh_index - 1] < candle_set["High"][i] and brcheck != 1:
                    self.supply_zone_values.append(
                        (
                            bcshigh_index - 1,
                            candle_set["Low"][bcshigh_index - 1],
                            i - 1,
                            candle_set["High"][bcshigh_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.supply_zone_values.append(
                    (
                        bcshigh_index - 1,
                        candle_set["Low"][bcshigh_index - 1],
                        len(candle_set) - 1,
                        candle_set["High"][bcshigh_index],
                    )
                )
        else:
            for i in range(j + 1, len(candle_set)):
                if candle_set["Low"][bcshigh_index] < candle_set["High"][i] and brcheck != 1:
                    self.supply_zone_values.append(
                        (
                            bcshigh_index,
                            candle_set["Low"][bcshigh_index],
                            i - 1,
                            candle_set["High"][bcshigh_index],
                        )
                    )
                    brcheck = 1
                    break
            if brcheck == 0:
                self.supply_zone_values.append(
                    (
                        bcshigh_index,
                        candle_set["Low"][bcshigh_index],
                        len(candle_set) - 1,
                        candle_set["High"][bcshigh_index],
                    )
                )

    def getmarketstructure(self, candle_set):
        last_index = 0
        runs = "H"
        if len(candle_set) > self.depth:
            last5_low_candles = np.array(candle_set["Low"])[: self.depth]
            last5_high_candles = np.array(candle_set["High"])[: self.depth]
            last5_low = np.min(last5_low_candles)
            last5_low_index = np.where(last5_low_candles == last5_low)[0][0]
            last5_high = np.max(last5_high_candles)
            last5_high_index = np.where(last5_high_candles == last5_high)[0][0]

            if last5_high_index < last5_low_index:
                self.lev.append((last5_high_index, last5_high))
                self.confirmations[int(last5_high_index)] = self.depth
                self.test_array.append((last5_high_index, last5_high, "HIGH"))
                last_index = last5_high_index
                runs = "L"
            else:
                self.lev.append((last5_low_index, last5_low))
                self.confirmations[int(last5_low_index)] = self.depth
                self.test_array.append((last5_low_index, last5_low, "LOW"))
                last_index = last5_low_index
                runs = "H"
            i = 0
            while i < len(candle_set) - 1:
                if runs == "H":
                    last_index = self.runforhigs(candle_set, last_index)
                    runs = "L"
                else:
                    last_index = self.runforlows(candle_set, last_index)
                    runs = "H"
                if last_index is not None:
                    i = last_index
                else:
                    if runs == "H":
                        last5_high_candles = np.array(candle_set["High"])[-self.depth :]
                        last5_high = np.max(last5_high_candles)
                        last5_high_index = np.where(last5_high_candles == last5_high)[0][0]
                        if last5_high != self.lev[-1][1]:
                            self.lev.append(
                                (len(candle_set) - self.depth + last5_high_index, last5_high)
                            )
                            self.test_array.append(
                                (
                                    len(candle_set) - self.depth + last5_high_index,
                                    last5_high,
                                    "HIGH",
                                )
                            )
                        runs = "L"
                    else:
                        last5_low_candles = np.array(candle_set["Low"])[-self.depth :]
                        last5_low = np.min(last5_low_candles)
                        last5_low_index = np.where(last5_low_candles == last5_low)[0][0]
                        if last5_low != self.lev[-1][1]:
                            self.lev.append(
                                (len(candle_set) - self.depth + last5_low_index, last5_low)
                            )
                            self.test_array.append(
                                (len(candle_set) - self.depth + last5_low_index, last5_low, "LOW")
                            )
                        runs = "H"
                    break

    def run(self, candle_set):

        self.getmarketstructure(candle_set)
        return SMCResult(
            candle_set=candle_set,
            lev=list(self.lev),
            test_array=list(self.test_array),
            demand_zone_values=list(self.demand_zone_values),
            supply_zone_values=list(self.supply_zone_values),
            bos_hh=list(self.bos_hh),
            bos_ll=list(self.bos_ll),
            imp_hh=list(self.imp_hh),
            imp_ll=list(self.imp_ll),
            confirmations=dict(self.confirmations),
        )


def main(candle_set, depth):
    return SMCAnalyzer(depth).run(candle_set)
