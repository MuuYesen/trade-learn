import pandas as pd

class Label:

    def __init__(self):
        pass

    @staticmethod
    def label_by_percent(data, rt, pos_prercent: float = 0.3, neg_percent: float = 0.3):
        data = pd.merge(data, rt, how='inner', on=['code'])

        data.loc[
            data['return'] > data['return'].quantile(1 - pos_prercent), 'label'] = 1
        data.loc[
            data['return'] < data['return'].quantile(neg_percent), 'label'] = 0

        data = data.drop(columns=['return'])
        return data