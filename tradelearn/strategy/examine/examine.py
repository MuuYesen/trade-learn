import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from tradelearn.strategy.examine.alphalens import performance
from tradelearn.strategy.examine.alphalens import utils
from tradelearn.strategy.examine.alphalens import tears

class Examine:

    def __init__(self):
        pass

    @staticmethod
    def single_factor(data: pd.DataFrame, col: str, filename: str = './examine.html'):
        close = data[['date', 'code', 'close']].pivot(index='date', columns='code', values='close')

        data = data.set_index(['date', 'code'], drop=True)
        data.sort_index(inplace=True)
        factor_data = utils.get_clean_factor_and_forward_returns(data[col], close, quantiles=5,
                                                                 periods=(1, 5, 10), max_loss=1)
        html = tears.create_full_tear_sheet(factor_data,
                                     long_short=True,
                                     group_neutral=False,
                                     by_group=False)

        with open(filename, 'w+', encoding='utf8') as file:
            file.write(html)

    @staticmethod
    def factor_compare(data: pd.DataFrame, ind: str = None, cir: str = None, f_col: str = None):
        if f_col:
            data = data[f_col]

        close = data[['date', 'code', 'close']].pivot(index='date', columns='code', values='close')

        data = data.set_index(['date', 'code'], drop=True)
        data.sort_index(inplace=True)

        eval_list = []
        for col in data.columns:
            if not col.startswith('alpha'):
                continue
            try:
                factor_data = utils.get_clean_factor_and_forward_returns(data[col], close, quantiles=5,
                                                                            periods=(5,), max_loss=1)
                mean_quant_ret, std_quantile = performance.mean_return_by_quantile( \
                    factor_data,
                    by_group=False
                )
                mean_quant_rateret = mean_quant_ret.apply(
                    utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
                )

                ic = performance.factor_information_coefficient(factor_data)

                def regress_t_value(data, fac, ind, cir):
                    industry_dummies = None
                    log_cir = None
                    if ind:
                        industry_dummies = pd.get_dummies(data[ind], drop_first=True, columns=[ind], dtype=int)
                    if cir:
                        log_cir = data[cir].apply(np.log)
                    x = pd.concat([industry_dummies, log_cir, data[fac]], axis=1)
                    y = data['return']
                    result = sm.OLS(y, x).fit()
                    return result.params[fac], result.tvalues[fac]

                t_stat, p_value = stats.ttest_1samp(ic['5D'], 0)

                res_series = data.groupby('date').apply(lambda x: regress_t_value(x, col, ind, cir))\
                                         .apply(pd.Series, index=['c_series', 't_series'])
                c_series, t_series = res_series['c_series'], res_series['t_series']

                res_dict = {
                    'name': col.split('.')[0],
                    'return_max': mean_quant_rateret['5D'].iloc[-1] * 10000,
                    'return_min': mean_quant_rateret['5D'].iloc[0] * 10000,
                    'ic_mean': ic['5D'].mean(),
                    'ic_std': ic['5D'].std(),
                    'risk_adjusted_ic(ir)': ic['5D'].mean() / ic['5D'].std(),
                    'ic_skew': stats.skew(ic['5D']),
                    'ic_kurtosis': stats.kurtosis(ic['5D']),
                    't_stat(IC)': t_stat,
                    'p_value(IC)': p_value,
                    't_mean': t_series.mean(),
                    't_std': t_series.std(),
                    'return_ratio_mean': c_series.mean(),
                    'return_ratio_std': c_series.std(),
                }

                print(res_dict)
                eval_list.append(res_dict)
            except Exception as e:
                print(f'{col} has error!', e)

        return pd.DataFrame(eval_list).set_index('name')