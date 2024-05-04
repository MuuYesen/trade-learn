import numpy as np
import pandas as pd
import statsmodels.api as sm


class Neutralize:

    def __init__(self):
        pass

    @staticmethod
    def neutralize(data, ind, cir):
        '''
        市值行业中性化，对某一时间截面的因子对市值及行业哑变量线性回归，取残差作为新的因子值
        :param factor_data: 某一时间界面的因子数据
        :param stockList: 交易标的
        :param industry_code: 用哪些行业划分
        :param date: 当前时间点
        :return: 中性化处理后的因子数据
        '''

        for col in data.columns:
            x = pd.concat([np.log(cir),
                           pd.get_dummies(ind, drop_first=True, columns=[ind.columns[0]])], axis=1)
            y = data[col]
            model = sm.OLS(y, x).fit()
            data[col] = model.resid

        return data
