import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

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

        # PART0
        non_missing_count = data.notnull().sum().tolist()
        missing_count = data.isnull().sum().tolist()

        data_info_str = f"data include {data.shape[0]} rows and {data.shape[1]} columns。<br><br>"
        data_info_str += "<table><tr><th>column name</th><th>data type</th><th>non_missing_count</th><th>missing_count</th><th>Total sample size</th></tr>"
        for i in range(data.shape[1]):
            data_info_str += f"<tr><td>{data.columns[i]}</td><td>{data.dtypes[i]}</td><td>{non_missing_count[i]}</td><td>{missing_count[i]}</td><td>{data.shape[0]}</td></tr>"
        data_info_str += "</table>"

        section_content = '<div><style type="text/css">p{ text-indent:2em;}</style> <br>' + data_info_str + '</div>'
        part0_html = template('section.html').render(section_content=section_content, section_title='Data Overview',
                                                     section_anchor_id='part0')

        # PART1
        descriptive_stats = data.describe().to_html()
        part1_html = template('section.html').render(section_content=descriptive_stats, section_title='Descriptive Statistics',
                                                     section_anchor_id='part1')

        # PART2
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(11, 10))
        for i, col in enumerate(data.columns.drop(['code', 'date'])):
            sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])
            axes[i//2, i%2].set_xlabel(col)
            axes[i//2, i%2].set_ylabel('value')
        plt.tight_layout()
        hist_buffer = BytesIO()
        plt.savefig(hist_buffer, format='png')
        hist_plot = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
        hist_buffer.close()
        hist_plot = f"<img src='data:image/png;base64,{hist_plot}' alt='Histogram'>"
        section_content = '' + str(hist_plot)
        part2_html = template('section.html').render(section_content=section_content, section_title='Variable Overview',
                                                     section_anchor_id='part2')

        # PART3
        plot_html_str = ''
        for col in data.columns.drop(['code', 'date']):
            plt.figure(figsize=(8, 6))
            plt.plot(data[col])
            plt.ylabel('some numbers')
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            curve_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            curve_plot1 = f"<img src='data:image/png;base64,{curve_plot1}' alt='Histogram'>"

            plt.figure(figsize=(8, 6))
            plt.hist(data[col])
            plt.ylabel('some numbers')
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            hist_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            hist_plot1 = f"<img src='data:image/png;base64,{hist_plot1}' alt='Histogram'>"

            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[col])
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            box_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            box_plot1 = f"<img src='data:image/png;base64,{box_plot1}' alt='Histogram'>"

            description_str2 = '<h2>variable name：' + col + '</h2>' \
                               '<style type="text/css">p{ text-indent:2em;}</style> <br>' \
                               '<p>1. mean：' + str(
                                format(data[col].mean(), '.4f')) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. min：' + str(
                                format(data[col].min(), '.4f')) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5. skew：' + str(format(data[col].skew(), '.4f')) + '</p>' \
                                                                                                                 '<p>2. std：' + str(
                                format(data[col].std(), '.4f')) + '&nbsp;&nbsp;&nbsp;&nbsp&nbsp&nbsp&nbsp&nbsp;&nbsp;&nbsp; 4. max：' + str(
                                format(data[col].max(), '.4f')) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6. kurt：' + str(format(data[col].kurt(), '.4f')) + '</p>'

            tab20_html = template('tabs.html').render(tab_id1='tab_id201' + col, tab_title1='curve_plot',
                                                      tab_content1=curve_plot1,
                                                      tab_id2='tab_id202' + col, tab_title2='hist_plot', tab_content2=hist_plot1,
                                                      tab_id3='tab_id203' + col, tab_title3='box_plot', tab_content3=box_plot1)

            plot_html_str = plot_html_str + description_str2 + tab20_html + "</br><hr style='border:1px solid #d0d0d5; height:1px'>"

        part3_html = template('section.html').render(section_content=plot_html_str, section_title='Variable Details',
                                                     section_anchor_id='part3')

        # BASE
        content = template('base.html').render(part0_html=part0_html,
                                               part1_html=part1_html,
                                               part2_html=part2_html,
                                               part3_html=part3_html)

        # PRIME
        html = template('wrapper.html').render(title='Exploratory Analysis Report',
                                               content=content,
                                               p1=len(part1_html) > 0,
                                               p2=len(part2_html) > 0,
                                               p3=len(part3_html) > 0)

        return html