import pandas as pd
from quantstats import reports
from tradelearn.strategy.evaluate.pyfolio import create_returns_tear_sheet


class Evaluate:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(stats: dict, baseline: pd.DataFrame, filename: str = None, engine: str = 'quantstats'):
        strat_rets = stats._equity_curve.Equity.pct_change()
        benchmark_ret = baseline.close.pct_change()

        if engine == 'pyfolio':
            html = create_returns_tear_sheet(returns=strat_rets, benchmark_rets=benchmark_ret)

            if filename is None:
                filename = './pyfolio.html'
            with open(filename, 'w') as file:
                file.write(html)

        if engine == 'quantstats':
            if filename is None:
                filename = './quantstats.html'
            reports.html(strat_rets, benchmark=benchmark_ret, output=filename, title='Stock Sentiment')
