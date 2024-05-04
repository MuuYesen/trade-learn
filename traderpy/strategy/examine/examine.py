import pandas as pd


class Examine:

    def __init__(self):
        pass

    @staticmethod
    def factor_compare(data):
        close = data[['date', 'code', 'close']].pivot(index='date', columns='code', values='close')

        data = data.set_index(['date', 'code'], drop=True)
        data.sort_index(inplace=True)

        eval_list = []
        for col in data.columns:
            if not col.startswith('alpha'):
                continue
            try:
                factor_data = al.utils.get_clean_factor_and_forward_returns(data[col], close, quantiles=5,
                                                                            periods=(5,), max_loss=1)
                mean_quant_ret, std_quantile = al.performance.mean_return_by_quantile( \
                    factor_data,
                    by_group=False
                )
                mean_quant_rateret = mean_quant_ret.apply(
                    al.utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
                )

                ic = al.performance.factor_information_coefficient(factor_data)

                res_dict = {
                    'name': col.split('.')[0],
                    'return_max': mean_quant_rateret['5D'].iloc[-1] * 10000,
                    'return_min': mean_quant_rateret['5D'].iloc[0] * 10000,
                    'ic_mean': ic['5D'].mean(),
                    'ic_std': ic['5D'].std(),
                    'ir': ic['5D'].mean() / ic['5D'].std()
                }
                print(res_dict)
                eval_list.append(res_dict)
            except:
                print(f'{col} has error!')

        return pd.DataFrame(eval_list)

if __name__ == '__main__':
    s_data = pd.read_csv('../../database/data/000300SH.csv', parse_dates=['date'], dtype={'code': str},
                         low_memory=True, encoding='utf_8_sig')
    s_data.rename(columns={'open': 'alpha_open'}, inplace=True)
    res = Examine.factor_compare(s_data)
    print(res)
