from ydata_profiling import ProfileReport
from traderpy.strategy.preprocess.explore.libs.ts_plot import TsPlot
from traderpy.strategy.preprocess.explore.libs.mi_plot import MiPlot


import os
cur_dir_path = os.path.abspath(os.path.dirname(__file__))

import matplotlib.pyplot as plt


class Explore:

    def __init__(self):
        pass

    @staticmethod
    def cross_sectional_data(data, path):
        config = {
        }
        report = ProfileReport(df=data,
                               title="Example Dataset Profile",
                               dark_mode=True,
                               config=None,
                               minimal=False,
                               interactions=None,
                               )
        report.to_file(path)

    @staticmethod
    def time_series_data(data, path):  # need no missgno
        vis = TsPlot(data, 'date', theme_name='light')
        for ts_fea in data.select_dtypes(include='number'):
            try:
                vis.full_statistics_plot(ts_fea, save=True, save_path=path)

                from jinja2 import Environment, FileSystemLoader
                env = Environment(loader=FileSystemLoader(os.path.join(cur_dir_path, './libs/')))
                template = env.get_template('template.html')
                result = template.render(name=ts_fea+"_Statistics_plot.png")
                print(result)
                with open('result.html', 'w') as file:
                    file.write(result)
            except:
                continue

    @staticmethod
    def missing_data(data):
        MiPlot.matrix(data, labels=True)
        plt.show()
        MiPlot.bar(data)
        plt.show()
        MiPlot.heatmap(data)
        plt.show()
        MiPlot.dendrogram(data)
        plt.show()


