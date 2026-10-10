"""Strategy report facade; rendering dependencies load only when requested."""

from tradelearn.report.sections import ReportContext, ReportSection

__all__ = ["Reporter", "ReportContext", "ReportSection"]


def __getattr__(name: str):
    if name == "Reporter":
        from tradelearn.report.reporter import Reporter

        return Reporter
    raise AttributeError(f"module 'tradelearn.report' has no attribute {name!r}")
