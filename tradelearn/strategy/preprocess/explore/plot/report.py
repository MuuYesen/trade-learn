import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
cur_dir_path = os.path.abspath(os.path.dirname(__file__))

from jinja2 import Environment, FileSystemLoader

class Report:

    def __init__(self):
        pass

    @staticmethod
    def analysis_report(data):
        def template(template_name: str):
            env = Environment(loader=FileSystemLoader(os.path.join(cur_dir_path, '..')))
            return env.get_template(template_name)

        data1 = data.copy()
        # PART0
        non_missing_count = data.notnull().sum().tolist()
        missing_count = data.isnull().sum().tolist()

        data_info_str = f"<center>the dataset include {data.shape[0]} rows and {data.shape[1]} columns. </center><br>"
        data_info_str += "<table><tr><th>column name</th><th>data type</th><th>non_missing_count</th><th>missing_count</th><th>Total sample size</th></tr>"
        for i in range(data.shape[1]):
            data_info_str += f"<tr><td>{data.columns[i]}</td><td>{data.dtypes[i]}</td><td>{non_missing_count[i]}</td><td>{missing_count[i]}</td><td>{data.shape[0]}</td></tr>"
        data_info_str += "</table>"

        section_content = '<div><style type="text/css">p{ text-indent:2em;}</style>' + data_info_str + '</div>'
        part0_html = template('section.html').render(section_content=section_content, section_title='Data Overview',
                                                     section_anchor_id='part0')

        data = data.drop(['code', 'date'], axis=1)

        # PART1
        descriptive_stats = data.describe().to_html()
        part1_html = template('section.html').render(section_content=descriptive_stats, section_title='Descriptive Statistics',
                                                     section_anchor_id='part1')

        # PART2
        ncols = 4  # 每行4个图
        nrows = math.ceil(len(data.columns)/ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3*nrows))  # 调整宽度以适应4列
        for i, col in enumerate(data.columns):
            row = i // ncols
            col_idx = i % ncols
            sns.histplot(data[col], kde=True, ax=axes[row, col_idx])
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('value')
            # 调整字体大小
            axes[row, col_idx].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col_idx].set_xlabel(col, fontsize=8)
            axes[row, col_idx].set_ylabel('value', fontsize=8)
        
        # 隐藏多余的子图
        for i in range(len(data.columns), nrows * ncols):
            row = i // ncols
            col_idx = i % ncols
            axes[row, col_idx].set_visible(False)
            
        plt.tight_layout()
        hist_buffer = BytesIO()
        plt.savefig(hist_buffer, format='png', bbox_inches='tight', dpi=100)
        hist_plot = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
        hist_buffer.close()
        hist_plot = f"<img src='data:image/png;base64,{hist_plot}' alt='Histogram' style='max-width:100%;'>"

        from tradelearn.strategy.preprocess.explore.template.common.mi_plot import MiPlot
        plt.figure(figsize=(8, 6))
        MiPlot.matrix(data, color=(0.537, 0.812, 0.941))
        plt.tight_layout()
        hist_buffer = BytesIO()
        plt.savefig(hist_buffer, format='png', bbox_inches='tight', dpi=100)
        hist_plot2 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
        hist_buffer.close()
        hist_plot2 = f"<br><div><img src='data:image/png;base64,{hist_plot2}' alt='Time Series Statistics' style='max-width:100%;'></div>"

        from tradelearn.strategy.preprocess.explore.template.common.pa_plot import PaPlot
        plt.figure(figsize=(8, 6))
        PaPlot.corr_plot(data)
        plt.tight_layout()
        hist_buffer = BytesIO()
        plt.savefig(hist_buffer, format='png', bbox_inches='tight', dpi=100)
        hist_plot3 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
        hist_buffer.close()
        hist_plot3 = f"<br><div><img src='data:image/png;base64,{hist_plot3}' alt='Time Series Statistics' style='max-width:100%;'></div>"

        section_content = '' + str(hist_plot) + str(hist_plot2) + str(hist_plot3)
        part2_html = template('section.html').render(section_content=section_content, section_title='Variable Overview',
                                                     section_anchor_id='part2')

        # PART3
        plot_html_str = ''
        for col in data.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(data1['date'], data[col])
            plt.ylabel('some numbers')
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            curve_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            curve_plot1 = f"<img src='data:image/png;base64,{curve_plot1}' alt='Curve Plot' style='width:100%; max-width:500px;'>"

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=data[col])
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            box_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            box_plot1 = f"<img src='data:image/png;base64,{box_plot1}' alt='Box Plot' style='width:100%; max-width:500px;'>"

            plots_container = '<center><div style="display: flex; justify-content: space-between; gap: 20px; margin: 20px 0;">' \
                '<div style="flex: 1;">' + curve_plot1 + '</div>' \
                '<div style="flex: 1;">' + box_plot1 + '</div>' \
                '</div></center>' 
            
            description_str2 = '<h2>variable name: ' + col + '</h2></br></br>' \
                               '<table style="border-collapse: collapse; width: 100%;">' \
                               '<tr><td>Mean</td><td>Std</td><td>Min</td><td>Max</td><td>Skew</td><td>Kurt</td></tr>'\
                               '<tr><td>' + str(format(data[col].mean(), '.4f')) + '</td>'\
                               '<td>' + str(format(data[col].std(), '.4f')) + '</td>'\
                               '<td>' + str(format(data[col].min(), '.4f')) + '</td>'\
                               '<td>' + str(format(data[col].max(), '.4f')) + '</td>'\
                               '<td>' + str(format(data[col].skew(), '.4f')) + '</td>'\
                               '<td>' + str(format(data[col].kurt(), '.4f')) + '</td></tr>' \
                               '</table>'

            if col == 'open':
                from tradelearn.strategy.preprocess.explore.template.common.ts_plot import TsPlot
                vis = TsPlot(data1.copy(), 'date')
                vis._apply_theme('light')
                fig = vis.full_statistics_plot('open', display=False, save=False)
                hist_buffer = BytesIO()
                fig.savefig(hist_buffer, format='png', bbox_inches='tight', dpi=100)
                hist_plot2 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
                hist_buffer.close()
                plt.close()
                hist_plot2 = f"<br><div><img src='data:image/png;base64,{hist_plot2}' alt='Time Series Statistics' style='max-width:100%;'></div>"
                plots_container += hist_plot2

            plot_html_str += description_str2 + plots_container + "<hr style='border:1px solid #d0d0d5; height:1px'>"

        part3_html = template('section.html').render(section_content=plot_html_str, section_title='Variable Details',
                                                     section_anchor_id='part3')

        # PRIME
        html = template('wrapper.html').render(title='Exploratory Analysis Report',
                                               part0_html=part0_html,
                                               part1_html=part1_html,
                                               part2_html=part2_html,
                                               part3_html=part3_html,
                                               p1=len(part1_html) > 0,
                                               p2=len(part2_html) > 0,
                                               p3=len(part3_html) > 0)

        return html