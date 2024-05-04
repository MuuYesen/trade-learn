

class Outlier:

    def __init__(self):
        pass

    @staticmethod
    def winorize_med(data, scale, axis=0):
        '''
        中位数去极值：设第 T 期某因子在所有个股上的暴露度序列为𝐷𝑖，𝐷𝑀为该序列中位数，𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数，
        则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 +5𝐷𝑀1，将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 −5𝐷𝑀1；
        :param factor_data: 因子df，columns为因子，raw为symbol
        :param scale: 几倍标准差
        :param axis: 默认columns为因子，raw为symbol
        :return: 去极值后的factor df
        '''

        def func(col):
            med = col.median()
            med1 = abs(col - med).median()
            col[col > med + scale * med1] = med + scale * med1
            col[col < med - scale * med1] = med - scale * med1
            return col

        win_factor_data = data.apply(func, axis=axis)
        return win_factor_data
