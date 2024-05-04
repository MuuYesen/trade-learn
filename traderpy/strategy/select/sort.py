import pandas as pd


class SortSelect:

    def __init__(self):
        pass

    @staticmethod
    def sort_by_factor(data: pd.Series, low_perc: int, high_perc: int):
        stock_list = data[(data >= data.quantile(low_perc))
                          & (data <= data.quantile(high_perc))].index.tolist()
        return stock_list

    @staticmethod
    def sort_by_factor_with_groups():
        pass


