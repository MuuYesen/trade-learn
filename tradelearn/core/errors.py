"""Shared exception hierarchy for trade-learn."""


class TradelearnError(Exception):
    """Base class for all trade-learn errors."""


class ContractError(TradelearnError):
    """Raised when data violates a documented core contract."""


class ConfigurationError(TradelearnError):
    """Raised when configuration or environment values are invalid."""


class DataError(TradelearnError):
    """Raised when input or provider data cannot be used."""


class GoldenDataError(TradelearnError):
    """Raised when golden baseline generation cannot proceed."""
