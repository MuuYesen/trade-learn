"""Small event adapter for dependency-ordered persistent indicator nodes.

Sequence is delivery order, not a timestamp. Revisions require reset/replay;
this adapter never rewrites committed bars. State and snapshots are trusted
in-process Python objects, not untrusted serialized code.
"""

from __future__ import annotations

import math
import numbers
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same(x, y) for x, y in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            return False
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            return all(_same(x, y) for x, y in zip(left.flat, right.flat, strict=True))
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        return left == right or (math.isnan(left) and math.isnan(right))
    return bool(left == right)


def _available(value):
    if isinstance(value, dict):
        return all(_available(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(_available(x) for x in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, numbers.Real):
        return math.isfinite(value)
    return value is not None


@dataclass
class _Node:
    backend: Any
    dependencies: tuple[str, ...]
    configuration: Any
    count: int = 0


class StreamGraph:
    """One graph per strategy/feed/symbol/timeframe and fixed parameters."""

    def __init__(self, *, identity):
        self.identity = tuple(identity)
        self._nodes = {}
        self._values = {}
        self._last = None
        self._failed = False

    @property
    def values(self):
        return deepcopy(self._values)

    def add(self, name, backend, *, dependencies=(), configuration=None):
        """Register in topological order before any event is consumed."""
        if self._last is not None:
            raise ValueError("cannot add nodes after streaming starts")
        if name in self._nodes:
            raise ValueError("duplicate node")
        if any(node.backend is backend for node in self._nodes.values()):
            raise ValueError(
                "persistent backend already registered; reference its dependency instead"
            )
        dependencies = tuple(dependencies)
        if any(dep not in self._nodes for dep in dependencies):
            raise ValueError("dependency must already be registered")
        self._nodes[name] = _Node(backend, dependencies, deepcopy(configuration))

    def ready(self, name):
        """Warmup is the maximum dependency path, not a fabricated zero value."""
        node = self._nodes[name]
        return (
            node.count >= getattr(node.backend, "warmup", 1)
            and name in self._values
            and _available(self._values[name])
            and all(self.ready(dep) for dep in node.dependencies)
        )

    def update(self, sequence, event_id, inputs):
        """Commit a new delivery; exact last-event duplicates reuse the result."""
        if self._failed:
            raise RuntimeError("graph failed; reset/replay required")
        payload = deepcopy(inputs)
        if self._last is not None:
            previous, previous_id, previous_inputs = self._last
            if sequence == previous:
                if event_id != previous_id or not _same(payload, previous_inputs):
                    raise ValueError("unsupported current-bar revision or sequence conflict")
                return self.values
            if sequence < previous:
                raise ValueError("out-of-order delivery")
            if event_id == previous_id:
                raise ValueError("unsupported event revision")
        try:
            for name, node in self._nodes.items():
                self._values[name] = deepcopy(
                    node.backend.update(
                        payload, {dep: self._values[dep] for dep in node.dependencies}
                    )
                )
                node.count += 1
        except Exception:
            self._failed = True
            raise
        self._last = (sequence, deepcopy(event_id), payload)
        return self.values

    def reset(self):
        """Release committed results and reset every persistent node."""
        for node in self._nodes.values():
            node.backend.reset()
            node.count = 0
        self._values.clear()
        self._last = None
        self._failed = False

    def _signature(self):
        return (
            self.identity,
            tuple(
                (
                    name,
                    type(n.backend).__module__,
                    type(n.backend).__qualname__,
                    n.dependencies,
                    n.configuration,
                )
                for name, n in self._nodes.items()
            ),
        )

    def snapshot(self):
        """Capture explicitly supported backend state; opaque native handles reject."""
        if self._failed:
            raise RuntimeError("cannot snapshot a failed graph")
        return deepcopy(
            dict(
                version=1,
                signature=self._signature(),
                values=self._values,
                last=self._last,
                nodes={name: (n.count, n.backend.snapshot()) for name, n in self._nodes.items()},
            )
        )

    def restore(self, snapshot):
        """Refuse identity/topology/configuration mismatch before mutation."""
        snapshot = deepcopy(snapshot)
        if snapshot.get("version") != 1 or not _same(snapshot.get("signature"), self._signature()):
            raise ValueError("incompatible graph snapshot")
        if (
            set(snapshot.get("nodes", {})) != set(self._nodes)
            or set(snapshot.get("values", {})) not in (set(), set(self._nodes))
            or any(type(item[0]) is not int or item[0] < 0 for item in snapshot["nodes"].values())
        ):
            raise ValueError("invalid graph snapshot nodes/counts")
        try:
            for name, (count, state) in snapshot["nodes"].items():
                self._nodes[name].backend.restore(state)
                self._nodes[name].count = count
        except Exception:
            self._failed = True
            raise
        self._values, self._last = snapshot["values"], snapshot["last"]
        self._failed = False
