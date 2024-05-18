import pandas as pd
from quantstats import reports
from .pyfolio import create_full_tear_sheet


class Evaluate:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(strat: dict, baseline: pd.DataFrame, filename: str = None, engine: str = 'quantstats'):
        tn_begin_date = strat.data.lines.datetime.date(-(strat.data.buflen() - 1))
        tn_end_date = strat.data.lines.datetime.date((0))

        baseline = baseline.query(f"date >= '{tn_begin_date}' and date <= '{tn_end_date}'")

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
            if filename is None:
                filename = './pyfolio.html'
            with open(filename, 'w') as file:
                file.write(html)

        if engine == 'quantstats':
            if filename is None:
                filename = './quantstats.html'
            reports.html(returns, benchmark=benchmark_ret, output=filename, title='Stock Sentiment')
