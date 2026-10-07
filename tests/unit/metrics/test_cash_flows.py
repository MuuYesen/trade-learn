from datetime import datetime, timezone

import pytest

from tradelearn import metrics


def test_cash_deposit_is_not_investment_return():
    assert metrics.cash_flow_return(100, 160, net_flow=50) == pytest.approx(0.1)


def test_modified_dietz_weights_actual_timestamps():
    start = datetime(2026, 9, 1, 15, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, 15, tzinfo=timezone.utc)
    event = datetime(2026, 9, 2, 3, tzinfo=timezone.utc)
    assert metrics.cash_flow_return(
        100, 160, events=[(event, 50)], start=start, end=end, method="modified_dietz"
    ) == pytest.approx(0.08)


def test_intraday_flow_cannot_claim_end_of_day():
    start = datetime(2026, 9, 1, 15, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        metrics.cash_flow_return(100, 160, events=[(start, 50)], start=start, end=end)
