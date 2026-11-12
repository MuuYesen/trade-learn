"""One raw observation clock for TL and framework loggers; no I/O or calculations."""
import time

_POSIX = hasattr(time, 'CLOCK_MONOTONIC') and callable(getattr(time, 'clock_gettime_ns', None))
CLOCK_DOMAIN = 'POSIX_MONOTONIC' if _POSIX else 'PYTHON_MONOTONIC'


def monotonic_ns():
    """Read the shared raw monotonic domain without converting to wall-clock time."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC) if _POSIX else time.monotonic_ns()
