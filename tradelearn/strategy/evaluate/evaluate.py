from quantstats import reports
from .pyfolio import create_full_tear_sheet


class Evaluate:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(strat, baseline, path='./', engine='quantstats'):

        pyfoliozer = strat.analyzers.getbyname('_Pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.index = returns.index.tz_convert(None)

        baseline = baseline.set_index(['date'])
        benchmark_ret = baseline.close.pct_change()

        if engine == 'pyfolio':
            html = create_full_tear_sheet(returns,
                                   benchmark_rets=benchmark_ret,
                                   positions=positions,
                                   transactions=transactions)
            with open(path + 'full_tearsheet.html', 'w') as file:
                file.write(html)

        if engine == 'quantstats':
            reports.html(returns, benchmark=benchmark_ret, output=path+'stats.html', title='Stock Sentiment')
