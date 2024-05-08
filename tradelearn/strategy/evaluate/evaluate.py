from quantstats import reports
from .common.pyfolio import create_full_tear_sheet


class Evaluate:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(pyfoliozer, baseline, engine='quantstats'):
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.index = returns.index.tz_convert(None)

        benchmark_ret = baseline.close.pct_change()

        if engine == 'pyfolio':
            create_full_tear_sheet(returns,
                                   benchmark_rets=benchmark_ret,
                                   positions=positions,
                                   transactions=transactions)
        if engine == 'quantstats':
            reports.html(returns, benchmark=benchmark_ret, output='stats.html', title='Stock Sentiment')
