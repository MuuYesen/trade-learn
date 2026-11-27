"""Small reference checks for research backtests."""

from math import isfinite, sqrt


def annual_sharpe_se(annual_sharpe: float, observations: float, periods_per_year: float) -> float:
    """Approximate SE; observations must match the period used for annualization."""
    if not isfinite(annual_sharpe):
        raise ValueError("annual_sharpe must be finite")
    if not isfinite(observations) or observations <= 1:
        raise ValueError("observations must be finite and greater than one")
    if not isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    years = observations / periods_per_year
    return sqrt((1 + annual_sharpe**2 / (2 * periods_per_year)) / years)


def needs_fill_review(position_change_expected: bool, fill_count: int) -> bool:
    """Flag a window with expected orders and no fills for diagnosis."""
    return position_change_expected and fill_count == 0
