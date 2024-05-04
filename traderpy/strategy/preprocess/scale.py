

class Scale:

    def __init__(self):
        pass

    @staticmethod
    def standardlize(data):
        '''
        标准化，原数据减去均值除以标准差，得到近似正态序列
        :param factor_data: 因子数据
        :return: 处理后序列
        '''
        data = (data-data.mean())/data.std()
        return data
