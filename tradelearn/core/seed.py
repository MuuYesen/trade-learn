"""Global seed helpers for deterministic Stage 0 behavior."""

from __future__ import annotations

import os
import random

import numpy as np

from tradelearn.core.errors import ConfigurationError

ENV_SEED = "TRADELEARN_SEED"


def get_seed() -> int | None:
    """Read ``TRADELEARN_SEED`` as an integer when it is set."""

    raw = os.environ.get(ENV_SEED)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_SEED} must be an integer, got {raw!r}") from exc


def set_global_seed(seed: int | None = None) -> int | None:
    """Set Python and NumPy random seeds and persist the env value."""

    if seed is None:
        seed = get_seed()
    if seed is None:
        return None
    seed = int(seed)
    os.environ[ENV_SEED] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
