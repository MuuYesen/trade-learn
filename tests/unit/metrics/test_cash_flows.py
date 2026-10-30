import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradelearn import metrics


def test_cash_deposit_is_not_investment_return():
    assert metrics.cash_flow_return(100,160,net_flow=50)==pytest.approx(.1)


def test_modified_dietz_weights_actual_timestamps():
    start=datetime(2026,9,1,15,tzinfo=timezone.utc)
    end=datetime(2026,9,2,15,tzinfo=timezone.utc)
    event=datetime(2026,9,2,3,tzinfo=timezone.utc)
    assert metrics.cash_flow_return(100,160,events=[(event,50)],start=start,end=end,
                                   method='modified_dietz')==pytest.approx(.08)


def test_intraday_flow_cannot_claim_end_of_day():
    start=datetime(2026,9,1,15,tzinfo=timezone.utc)
    end=datetime(2026,9,2,15,tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        metrics.cash_flow_return(100,160,events=[(start,50)],start=start,end=end)


def test_insolvency_is_explicit_and_does_not_hide_a_finite_loss():
    with pytest.raises(ValueError, match='invalid adjusted return'):
        metrics.cash_flow_return(100, -10, net_flow=0, method='modified_dietz')
    assert metrics.cash_flow_return(
        100, -10, net_flow=0, method='modified_dietz', allow_insolvency=True
    ) == pytest.approx(-1.1)


def test_insolvency_does_not_bypass_invalid_capital_or_nonfinite_values():
    with pytest.raises(ValueError, match='positive invested capital'):
        metrics.cash_flow_return(0, -10, net_flow=0, allow_insolvency=True)
    with pytest.raises(ValueError, match='nonfinite amount'):
        metrics.cash_flow_return(100, float('inf'), net_flow=0, allow_insolvency=True)


def test_scalar_cash_flow_import_requires_no_vector_dependencies():
    root = Path(__file__).resolve().parents[3]
    program = '''import sys
sys.path.insert(0, sys.argv[1])
from tradelearn.metrics.cash_flows import cash_flow_return
from tradelearn.metrics import cash_flow_return as public_return
assert cash_flow_return is public_return
assert cash_flow_return(100, 90, net_flow=0) == -0.1
assert 'pandas' not in sys.modules and 'numpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-S', '-c', program, str(root)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_modified_dietz_does_not_overflow_an_individually_finite_weight():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc)
    event = datetime(2026, 9, 1, 0, 0, 30, tzinfo=timezone.utc)
    assert metrics.cash_flow_return(
        100, 100, events=[(event, 1e308)], start=start, end=end,
        method='modified_dietz', allow_insolvency=True
    ) == pytest.approx(-2)


def test_modified_dietz_rejects_nonfinite_invested_capital():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc)
    event = datetime(2026, 9, 1, 0, 0, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='nonfinite cash flow arithmetic'):
        metrics.cash_flow_return(
            1.5e308, 1.5e308, events=[(event, 1e308)], start=start, end=end,
            method='modified_dietz', allow_insolvency=True
        )


def test_cash_flow_rejects_incomplete_and_inconsistent_inputs():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    midday = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    cases = [
        ({"begin": True, "finish": 100, "net_flow": 0}, "boolean is not an amount"),
        ({"begin": 100, "finish": 100, "net_flow": 0, "allow_insolvency": 1}, "boolean"),
        ({"begin": 100, "finish": 100, "net_flow": 0, "method": "unknown"}, "unsupported"),
        ({"begin": 100, "finish": 100, "events": [(end, 1)]}, "aware valuation"),
        ({"begin": 100, "finish": 100, "events": [(midday, 1)],
          "start": start, "end": end}, "intraday"),
        ({"begin": 100, "finish": 100, "events": [(end, 1)], "net_flow": 2,
          "start": start, "end": end}, "disagrees"),
        ({"begin": 100, "finish": 100, "net_flow": 1,
          "method": "modified_dietz"}, "timed events"),
        ({"begin": 100, "finish": 100}, "explicit cash flow"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            metrics.cash_flow_return(**kwargs)


def test_metrics_public_names_are_discoverable():
    assert "cash_flow_return" in dir(metrics)
