import codecs
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

        data_info_str = f"数据包含 {data.shape[0]} 行和 {data.shape[1]} 列。<br><br>"
        data_info_str += "<table><tr><th>列名</th><th>数据类型</th><th>非空值数量</th><th>缺失样本量</th><th>总样本量</th></tr>"
        for i in range(data.shape[1]):
            data_info_str += f"<tr><td>{data.columns[i]}</td><td>{data.dtypes[i]}</td><td>{non_missing_count[i]}</td><td>{missing_count[i]}</td><td>{data.shape[0]}</td></tr>"
        data_info_str += "</table>"

        section_content = '<div><style type="text/css">p{ text-indent:2em;}</style> <br>' + data_info_str + '</div>'
        part0_html = template('section.html').render(section_content=section_content, section_title='数据概述',
                                                     section_anchor_id='part0')

        # PART1
        descriptive_stats = data.describe().to_html()
        part1_html = template('section.html').render(section_content=descriptive_stats, section_title='描述性统计',
                                                     section_anchor_id='part1')

        # PART2
        data.hist(bins=35, figsize=(10, 10))
        hist_buffer = BytesIO()
        plt.savefig(hist_buffer, format='png')
        hist_plot = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
        hist_buffer.close()
        hist_plot = f"<img src='data:image/png;base64,{hist_plot}' alt='Histogram'>"

        section_content = '' + str(hist_plot)

        part2_html = template('section.html').render(section_content=section_content, section_title='全局概览',
                                                     section_anchor_id='part2')

        # PART3
        plot_html_str = ''
        for col in data.columns.drop(['code', 'date']):
            plt.figure(figsize=(8, 4))
            plt.plot(data[col])
            plt.ylabel('some numbers')
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            curve_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            curve_plot1 = f"<img src='data:image/png;base64,{curve_plot1}' alt='Histogram'>"

            plt.figure(figsize=(8, 4))
            plt.hist(data[col])
            plt.ylabel('some numbers')
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            hist_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            hist_plot1 = f"<img src='data:image/png;base64,{hist_plot1}' alt='Histogram'>"

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=data[col])
            hist_buffer = BytesIO()
            plt.savefig(hist_buffer, format='png')
            box_plot1 = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
            hist_buffer.close()
            box_plot1 = f"<img src='data:image/png;base64,{box_plot1}' alt='Histogram'>"

            description_str2 = '<h2>变量名：' + col + '</h2>' \
                                                     '<style type="text/css">p{ text-indent:2em;}</style> <br>' \
                                                     '<p>1. 均值：' + str(
                data[col].mean()) + '&nbsp;&nbsp;&nbsp;&nbsp; 3. 最小值：' + str(
                data[col].min()) + '&nbsp;&nbsp;&nbsp;&nbsp; 5. 偏度：' + str(data[col].skew()) + '</p>' \
                                                                                                 '<p>2. 方差：' + str(
                data[col].std()) + '&nbsp;&nbsp;&nbsp;&nbsp; 4. 最大值：' + str(
                data[col].max()) + '&nbsp;&nbsp;&nbsp;&nbsp; 6. 峰度：' + str(data[col].kurt()) + '</p>'

            tab20_html = template('tabs.html').render(tab_id1='tab_id201' + col, tab_title1='曲线图',
                                                      tab_content1=curve_plot1,
                                                      tab_id2='tab_id202' + col, tab_title2='直方图',
                                                      tab_content2=hist_plot1,
                                                      tab_id3='tab_id203' + col, tab_title3='箱线图',
                                                      tab_content3=box_plot1)

            plot_html_str = plot_html_str + description_str2 + tab20_html + "</br><hr style='border:1px solid #d0d0d5; height:1px'>"

        part3_html = template('section.html').render(section_content=plot_html_str, section_title='变量详情',
                                                     section_anchor_id='part3')

        # BASE
        content = template('base.html').render(part0_html=part0_html,
                                               part1_html=part1_html,
                                               part2_html=part2_html,
                                               part3_html=part3_html)

        # PRIME
        html = template('wrapper.html').render(title='探索性分析报告',
                                        content=content,
                                        p1=len(part1_html) > 0,
                                        p2=len(part2_html) > 0,
                                        p3=len(part3_html) > 0)

        return html



