"""Smoke-test the installed native wheel, without importing project source files."""

from tradelearn import __version__, _rust


def engine():
    return _rust.RustBacktestEngine(
        [1, 2],
        [10.0, 8.0],
        [12.0, 9.0],
        [8.0, 7.0],
        [10.0, 8.0],
        [1000.0, 1000.0],
        10000.0,
        0.0,
        False,
        False,
        False,
        0.0,
        0.0,
        True,
        True,
        False,
    )


assert __version__ == _rust.tradelearn_rust_version() == "0.2.6"
e = engine()
ref = e.submit_order("buy", "stop", 1.0, None, 11.0)
assert e.step(0)[0][3] == 11.0
e = engine()
ref = e.submit_order("buy", "limit", 1.0, 9.0)
assert e.cancel_order(ref)
assert e.step(0) == []
e = engine()
ref = e.submit_order("buy", "limit", 1.0, 9.0)
e.configure_order(ref, 0, None)
assert e.step(0) == []
assert e.drain_order_events() == [(ref, "expired")]
e = engine()
a = e.submit_order("sell", "stop", 1.0, None, 9.0)
b = e.submit_order("sell", "limit", 1.0, 11.0)
e.configure_order(b, None, a)
assert len(e.step(0)) == 1
assert e.step(1) == []
print("trade-learn 0.2.6 native lifecycle smoke passed")
