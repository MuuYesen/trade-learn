"""Project declared native time-bar periods onto existing BT feed metadata."""
from tradelearn.engine.base import TimeFrame


def bar_period(bar_type):
    """Return unknown for non-time bars; source event spacing is never inferred."""
    mapping = {
        'MILLISECOND': (TimeFrame.MicroSeconds, 1000),
        'SECOND': (TimeFrame.Seconds, 1),
        'MINUTE': (TimeFrame.Minutes, 1),
        'HOUR': (TimeFrame.Minutes, 60),
        'DAY': (TimeFrame.Days, 1),
        'WEEK': (TimeFrame.Weeks, 1),
        'MONTH': (TimeFrame.Months, 1),
        'YEAR': (TimeFrame.Years, 1),
    }
    declared = mapping.get(bar_type.spec.aggregation.name)
    if declared is None:
        return dict(timeframe=None, compression=None)
    timeframe, scale = declared
    return dict(timeframe=timeframe, compression=int(bar_type.spec.step) * scale)


def feed_period_metadata(instruments, subscriptions):
    """Keep only unambiguous declarations for each explicitly bound feed."""
    result = {}
    for symbol, instrument in instruments.items():
        periods = [bar_period(bar) for bar in subscriptions if bar.instrument_id == instrument]
        if periods and all(value == periods[0] for value in periods):
            result[symbol] = periods[0]
    return result
