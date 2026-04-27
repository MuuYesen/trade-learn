"""Standard strategy library."""

import importlib

def _import_strategy(module_name: str, class_name: str):
    """Helper to import from files with numeric prefixes."""
    module = importlib.import_module(f".{module_name}", package=__package__)
    return getattr(module, class_name)

QuickstartSmaCross = _import_strategy("01_quickstart", "QuickstartSmaCross")
SmaCross = _import_strategy("02_sma_cross", "SmaCross")
RandomForestRotation = _import_strategy("03_rf_rotation", "RandomForestRotation")
Alpha101GBMStrategy = _import_strategy("04_ml_gbm", "Alpha101GBMStrategy")
MigratedSmaCross = _import_strategy("05_migration", "MigratedSmaCross")
Turtle = _import_strategy("06_turtle", "Turtle")
EnhancedRSI = _import_strategy("07_rsi_enhanced", "EnhancedRSI")
BetterMA = _import_strategy("08_better_ma", "BetterMA")
MacdTharp = _import_strategy("09_macd_settings", "MacdTharp")
OrderExecutionStrategy = _import_strategy("10_order_execution", "OrderExecutionStrategy")

__all__ = [
    "SmaCross",
    "Alpha101GBMStrategy",
    "RandomForestRotation",
    "QuickstartSmaCross",
    "MigratedSmaCross",
    "Turtle",
    "EnhancedRSI",
    "BetterMA",
    "MacdTharp",
    "OrderExecutionStrategy",
]
