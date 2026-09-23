"""Native trailing order regressions; exercise state across real engine bars."""
import pytest

from tradelearn import _rust


def engine_for(bars, smart):
    opens, highs, lows, closes = map(list, zip(*bars))
    return _rust.RustBacktestEngine(
        list(range(1, len(bars) + 1)), opens, highs, lows, closes,
        [1000.0] * len(bars), 10000.0, 0.0, False, False, False,
        0.0, 0.0, True, True, False, smart_matching=smart,
    )


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("side,reference,limit,bars,expected", [
    ("sell", 100.0, 97.0, [(90, 94, 89, 92), (99, 101, 98, 100)], 99.0),
    ("buy", 100.0, 103.0, [(110, 111, 106, 108), (101, 102, 99, 100)], 101.0),
])
def test_trailing_limit_respects_limit_and_remains_triggered(smart, side, reference, limit, bars, expected):
    engine = engine_for(bars, smart)
    ref = engine.submit_order(side, "stop_trail_limit", 1.0, limit, reference, 5.0)
    assert engine.step(0) == []
    fills = engine.step(1)
    assert len(fills) == 1
    assert fills[0][0] == ref
    assert fills[0][3] == expected


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("side,bars", [
    ("sell", [(97, 98, 96, 97), (96, 97, 94, 95)]),
    ("buy", [(103, 104, 102, 103), (104, 106, 103, 105)]),
])
def test_initial_reference_never_moves_in_unfavorable_direction(smart, side, bars):
    engine = engine_for(bars, smart)
    engine.submit_order(side, "stop_trail", 1.0, None, 100.0, 5.0)
    assert engine.step(0) == []
    fills = engine.step(1)
    assert len(fills) == 1
    assert fills[0][3] == (95.0 if side == "sell" else 105.0)


@pytest.mark.parametrize("smart", [False, True])
def test_limit_crossing_before_trigger_cannot_fill_trailing_limit(smart):
    # Bearish path: 100 -> 110 -> 90 -> 92. Limit 102 is visited before stop 95.
    engine = engine_for([(100, 110, 90, 92), (103, 104, 101, 102)], smart)
    engine.submit_order("sell", "stop_trail_limit", 1.0, 102.0, 100.0, 5.0)
    assert engine.step(0) == []
    assert engine.step(1)[0][3] == 103.0


@pytest.mark.parametrize("smart", [False, True])
def test_current_bar_high_only_updates_trail_after_matching(smart):
    engine = engine_for([(100, 120, 99, 115), (114, 115, 110, 111)], smart)
    engine.submit_order("sell", "stop_trail", 1.0, None, 100.0, 5.0)
    assert engine.step(0) == []
    assert engine.step(1)[0][3] == 114.0


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("side,bars,limit,expected", [
    ("sell", [(100, 101, 90, 91)], 94.0, 95.0),
    ("buy", [(100, 110, 99, 109)], 106.0, 105.0),
])
def test_trailing_limit_fills_at_trigger_not_earlier_open(smart, side, bars, limit, expected):
    engine = engine_for(bars, smart)
    engine.submit_order(side, "stop_trail_limit", 1.0, limit, 100.0, 5.0)
    assert engine.step(0)[0][3] == expected


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("side,bars,expected", [
    ("sell", [(100, 120, 99, 115), (110, 112, 107, 109)], 108.0),
    ("buy", [(100, 101, 80, 85), (85, 89, 83, 87)], 88.0),
])
def test_percentage_trail_tracks_favorable_prior_bar(smart, side, bars, expected):
    engine = engine_for(bars, smart)
    engine.submit_order(side, "stop_trail", 1.0, None, 100.0, None, 0.1)
    assert engine.step(0) == []
    assert engine.step(1)[0][3] == expected


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("side,bars,limit,expected", [
    ("sell", [(90, 100, 89, 99)], 97.0, 97.0),
    ("buy", [(110, 111, 100, 101)], 103.0, 103.0),
])
def test_gap_trigger_can_fill_limit_on_later_segment(smart, side, bars, limit, expected):
    engine = engine_for(bars, smart)
    engine.submit_order(side, "stop_trail_limit", 1.0, limit, 100.0, 5.0)
    assert engine.step(0)[0][3] == expected
