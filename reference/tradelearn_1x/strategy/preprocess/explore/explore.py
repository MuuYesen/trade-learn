import pandas as pd

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



