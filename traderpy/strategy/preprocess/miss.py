import numpy as np
import pandas as pd


class Miss:

    def __init__(self):
        pass

    @staticmethod
    def replace_nan_indu(data, ind):
        '''
        缺失值处理：得到新的因子暴露度序列后，将因子暴露度缺失的地方设为申万一级行业相同个股的平均值。
        :param factor_data: 因子df，columns为因子，raw为symbol
        :param stockList: 代码list
        :param industry_code: 聚宽的industry list
        :param date: 日期
        :return: 缺失值处理后的factor df
        '''

        data = pd.merge(data, ind, how='inner', on=['code'])
        industry_mean_df = data.groupby('ind_code').mean().apply(lambda x: x.fillna(x.mean()), axis=0)

        for index in data.index:
            for col in data.columns.drop('ind_code'):
                if np.isnan(data.loc[index, col]):
                    data.loc[index, col] = industry_mean_df.loc[data.loc[index, 'ind_code'], col]
        data = data.drop(columns=['ind_code'])
        return data