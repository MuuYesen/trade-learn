"""Progress helper with an optional tqdm backend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(iterable: Iterable[T], **kwargs: object) -> Iterable[T]:
    """Wrap ``iterable`` with tqdm when installed, otherwise return it."""

    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def iter_progress(iterable: Iterable[T], **kwargs: object) -> Iterator[T]:
    """Yield values from ``progress`` for call sites that need an iterator."""

    yield from progress(iterable, **kwargs)
