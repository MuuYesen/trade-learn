## trade-learn：使用 Python 搭建机器学习交易策略

trade-learn 是一个基于 alphalens、backtrader、pyfolio 和 quantstats 的机器学习策略研发工具包，提供因子采集、因子处理、因子评估、因果分析、模型定义和策略回测的全套策略研发流程，并支持可视化结果以 html 文件进行存档分享。

<img src="docs/img.png" alt="img" width="100%">

可视化图汇总：

<div align=center>
<img src="docs/plot_list.png" alt="img" width="70%">
</div>

## 主要特性

1. 内嵌美国量化交易平台 quantpian 开源的策略研发组件，如 empyrical、alphalens、pyfolio。
1. 提供「雅虎财经」的股票行情，以及相应的因子计算公式，包括 alpha101 和 alpha191。
2. 提供「通达信交易软件」的股票行情，以及配套的 30 个经验证的技术指标 tdx30，可直接对标通达信平台使用。
3. 使用信号驱动机制，令用户在信号编制中具有足够的自由度，并支持量价因子与机器学习模型信号，实现一步回测。
3. 接口调用方式简单，只需提供 ohlc 数据，同时信号计算以及发出统一在单个方法里完成。
1. 提供多个不同类型的策略模板，快速搭建相应策略回测，目前支持单标的的「投机策略」和多标的的「投资组合策略」。
2. 提供「因果图构建」和「因果特征选择」算法，为因子分析增加因果维度的探索，扩展 gplearn 的函数库，面向时序数据进行特征衍生。
6. 提供「探索性分析」和「最优模型选择」工具，迅速预览数据集的规律，以及常见模型在数据集的性能表现。
12. 裁剪 backtrader 回测框架，减少不必要的依赖安装，优化回测结果至 html 页面展示，拥有更友好的可视化互动。
14. 整个策略搭建过程，形成机器学习策略搭建的流程闭环，除了模型定义外，无需再引入其余第三方包。


## 下载方法

```bash
pip install trade-learn
```

```bash
pip install https://github.com/MuuYesen/trade-learn.git
```

## 简单例子

**使用量价指标进行单标的买卖**：
```python
from tradelearn.query.query import Query  # 导入数据查询模块
from tradelearn.strategy.preprocess.explore.explore import Explore  # 导入数据探索模块
from tradelearn.trader.utils.align import Align
from tradelearn.strategy.backtest.single import LongBacktest  # 导入单支股票回测模块
from tradelearn.strategy.evaluate.evaluate import Evaluate  # 导入策略评估模块

import numpy as np

import tradelearn.trader as bt


if __name__ == '__main__':
    # 定义数据起始日期和结束日期
    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    # 查询股票600520的历史数据作为基准
    baseline = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')

    # 获取原始数据并添加标签
    rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
    rawdata['label'] = rawdata['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)

    # 特征列表，去除标签和代码以及日期列
    fea_list = rawdata.columns.drop(['label', 'code', 'date']).tolist()

    # 数据探索
    Explore.analysis_report(rawdata)

    # 数据对齐
    rawdata = Align.transform(rawdata, baseline)

    # 定义随机森林指标类
    class RSI(bt.Indicator):

        lines = ("model_indi",)  # 指标线

        def __init__(self, stockid, fina_data, bt_begin_date, bt_end_date, fea_list):
            
            indi = Query.tec_indicator(fina_data, ['RSI']) # 计算相对强弱指标RSI

            # 生成信号
            def signal(x):
                if x < 20:
                    return True
                if x > 40:
                    return False
                return np.NAN
            indi = indi.set_index('date').map(signal)

            # 根据信号生成指标数据
            bt_indi = indi.query(f"date >= '{bt_begin_date}' and date < '{bt_end_date}'").values.reshape(-1)
            
            tmp_list = [np.NaN if fina_data['is_fake'].iloc[i] else bt_indi[i] for i in range(len(bt_indi))]
            self.lines.model_indi.array.extend(tmp_list)

    # 定义回测起始日期和结束日期
    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    # 运行回测
    res = LongBacktest.run(test_data=rawdata,
                           base_line=baseline,
                           model_class=RSI,
                           feature_list=fea_list,
                           begin_date=bt_begin_date,
                           end_date=bt_end_date)

    # 分析回测结果
    Evaluate.analysis_report(res, baseline, engine='quantstats')  # 使用quantstats引擎进行回测结果分析

```

**使用机器学习模型进行投资组合的搭建**：  
```python
from tradelearn.query.query import Query  # 导入数据查询模块
from tradelearn.strategy.preprocess.explore.explore import Explore  # 导入数据探索模块
from tradelearn.trader.utils.align import Align
from tradelearn.strategy.backtest.fund import LongBacktest  # 导入长周期回测模块
from tradelearn.strategy.evaluate.evaluate import Evaluate  # 导入策略评估模块

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import tradelearn.trader as bt
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器


if __name__ == '__main__':
    # 定义数据起始日期和结束日期
    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    # 查询沪深300指数的历史数据作为基准
    baseline = Query.history_ohlc(symbol='000001.SS', start=tn_begin_date, end=tn_end_date, engine='yahoo')  ## 两个接口都是右开区间，所有都是包括自定义

    rawdata = None
    # 循环查询多只股票的历史数据并进行处理
    for i in range(10):
        temp = Query.history_ohlc(symbol='60052' + str(i), start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
        if temp is None:
            continue

        # 标记涨跌标签
        temp['label'] = temp['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)
        rawdata = pd.concat([rawdata, temp], axis=0)

    # 特征列表，去除标签和代码以及日期列
    fea_list = rawdata.columns.drop(['label', 'code', 'date']).tolist()

    # 数据探索
    Explore.analysis_report(rawdata)

    # 数据对齐
    rawdata = Align.transform(rawdata, baseline)

    # 定义随机森林指标类
    class RandomForest(bt.Indicator):

        model_dict = {}  # 模型字典

        lines = ("model_indi",)  # 指标线

        def __init__(self, stockid, fina_data, bt_begin_date, bt_end_date, fea_list):

            if not RandomForest.model_dict:
                train_data = fina_data.query("is_fake == False")  # 过滤掉测试数据

                # 构建随机森林模型并保存到模型字典中
                for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                    bt_train_data = train_data.query(f"date >= '{date - relativedelta(months=12 * 3)}' and date < '{date}'")
                    bt_x_train, bt_y_train = bt_train_data[fea_list], bt_train_data['label']

                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    model.fit(bt_x_train, bt_y_train)
                    RandomForest.model_dict[date.year] = model

            indi_list = []
            # 使用模型进行预测
            for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                pos_data = fina_data.query(f"code == '{stockid}' and date >= '{date}' and date < '{date + relativedelta(months=12 * 1)}'")
                bt_x_test = pos_data.set_index(['date'])[fea_list]
                pre_proba = RandomForest.model_dict[date.year].predict_proba(bt_x_test)[:, 1]

                tmp_list = [np.NaN if pos_data['is_fake'].iloc[i] else pre_proba[i] for i in range(len(pre_proba))]
                indi_list.extend(tmp_list)

            self.lines.model_indi.array.extend(indi_list)

    # 定义回测起始日期和结束日期
    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    # 运行回测
    res = LongBacktest.run(test_data=rawdata,
                           base_line=baseline,
                           model_class=RandomForest,
                           feature_list=fea_list,
                           begin_date=bt_begin_date,
                           end_date=bt_end_date)

    # 分析回测结果
    Evaluate.analysis_report(res, baseline, engine='quantstats')  # 使用quantstats引擎进行回测结果分析

```

数据的形式假设：
code date xxx1 xxx2


## 方法指南


## 致谢

感谢以下仓库提供的支持： xxx


## 联系方式

知守溪的收纳屋。
muyes88@gmail.com

