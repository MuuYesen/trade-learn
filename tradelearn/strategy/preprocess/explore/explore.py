import pandas as pd

from tradelearn.strategy.preprocess.explore.template.common.ts_plot import TsPlot
from tradelearn.strategy.preprocess.explore.template.common.mi_plot import MiPlot
from tradelearn.strategy.preprocess.explore.template.common.report import Report

import os
cur_dir_path = os.path.abspath(os.path.dirname(__file__))

import matplotlib.pyplot as plt


class Explore:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(data: pd.DataFrame, filename: str = './explore.html'):
        html = Report.analysis_report(data)
        with open(filename, 'w+', encoding='utf8') as file:
            file.write(html)

    @staticmethod
    def cross_sectional_data(data, path):
        pass

    @staticmethod
    def time_series_data(data, path='./'):  # need no missgno
        vis = TsPlot(data, 'date', theme_name='light')
        for ts_fea in data.select_dtypes(include='number'):
            try:
                vis.full_statistics_plot(ts_fea, save=True)
                vis.save_plot(file_name=f'{ts_fea}_tsplot', save_path=path)
            except Exception:
                pass

    @staticmethod
    def missing_data(data, path='./'):
        MiPlot.matrix(data, labels=True)
        plt.show()
        plt.savefig(path + "miss_matrix.png")
        MiPlot.bar(data)
        plt.show()
        plt.savefig(path + "miss_bar.png")


