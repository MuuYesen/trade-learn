"""Finite recursive state reused by fixed MyTT adapters."""

import math


class EWM:
    def __init__(self, n):
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("period must be a positive integer")
        self.alpha = 2 / (n + 1)
        self.value = float("nan")
        self.weight = 1.0

    def add(self, x):
        if not math.isnan(self.value):
            self.weight *= 1 - self.alpha
            if not math.isnan(x):
                self.value = (self.weight * self.value + self.alpha * x) / (
                    self.weight + self.alpha
                )
                self.weight = 1.0
        elif not math.isnan(x):
            self.value = x
        return self.value
