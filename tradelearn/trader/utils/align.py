import pandas as pd


class Align:

    def __int__(self):
        pass

    @staticmethod
    def transform(data, baseline):
        res = None
        for symbol in data['code'].unique():
            temp = data.query(f"code == '{symbol}'")
            temp = pd.merge(temp, baseline['date'], on=['date'], how='right')
            temp['is_fake'] = temp.isnull().all(axis=1)
            temp = temp.fillna(method='ffill')
            temp = temp.fillna(method='bfill')
            res = pd.concat([res, temp], axis=0)
        return res