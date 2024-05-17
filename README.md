## trade-learn：Building Trading Strategies in Python with Machine Learning 

<b>trade-learn</b> is a machine learning strategy development toolkit based on alphalens, backtrader, pyfolio, and quantstats. It provides a <b>complete strategy development process</b>, including factor collection, factor processing, factor evaluation, <b>causal analysis</b>, model definition, and strategy backtesting, and supports visualization results saved as <b>HTML files</b> for sharing.

<img src="docs/img.png" alt="img" width="100%">

Summary of visualizations:


<div align=center>
<img src="docs/plot_list.png" alt="img" width="70%">
</div>

## Key Features

1. Integrated with strategy development components from the Quantopian open-source platform, such as empyrical, alphalens, and pyfolio toolkits.
2. Provides stock quotes from "Yahoo Finance" and corresponding factor calculation formulas, including alpha101 and alpha191 factor sets.
3. Provides stock quotes from "Tongdaxin Trading Software" and 30 verified technical indicators (tdx30), directly usable on the Tongdaxin platform.
4. Signal-driven trading strategies with multiple templates to quickly build and backtest strategies, supporting both speculative and portfolio strategies.
5. Causal graph construction and causal feature selection algorithms, and extend the gplearn function library to achieve "feature derivation" for time-series data.
6. Exploratory analysis and optimal model selection tools to quickly preview data set patterns and common models' performance on the data set.
7. Trimmed backtrader backtesting framework to reduce unnecessary dependencies and optimize backtest results for HTML display, providing more user-friendly interactive visualization.
8. The entire strategy building process forms a complete loop for machine learning strategy development without introducing additional third-party packages except for model definition.


## Download

```bash
pip install trade-learn
```

```bash
pip install https://github.com/MuuYesen/trade-learn.git@master
```

## Usage Template
```python
from tradelearn.trader.signal import Signal
from tradelearn.strategy.backtest.single import LongBacktest

# Data retrieval
raw_data, base_line = "Target stock data", "Benchmark stock data"

# Define backtest start and end dates
bt_begin_date, bt_end_date = "Backtest start date", "Backtest end date"

# Define Signal class
class Example(Signal):

    def __init__(self, stockid, raw_data, bt_begin_date, bt_end_date, param_dict):
        signal_df = "Computed signal series containing True, False, and np.NAN values, with dates set as index"
        
        self.set_signal(signal_df)

# Signal class parameter dictionary
param_dict = {'fea_list': "Set of variable names used to generate signals"}

# Run backtest
res = LongBacktest.run(Example, param_dict, raw_data, base_line, bt_begin_date, bt_end_date)
```
## Simple Example

**Using volume and price indicators for single stock trading**：
```python
from tradelearn.query.query import Query  # Import data query module
from tradelearn.trader.signal import Signal  # Import strategy signal class
from tradelearn.strategy.backtest.single import LongBacktest  # Import single stock backtest module
from tradelearn.strategy.evaluate.evaluate import Evaluate  # Import strategy evaluation module

import numpy as np


if __name__ == '__main__':
    
    # Define data start and end dates
    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    # Query historical data for stock 600520 as the benchmark
    baseline = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')

    # Retrieve raw data and add labels
    rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
    rawdata['label'] = rawdata['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)

    # Define backtest start and end dates
    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'
    
    # Define RSI signal class
    class RSI(Signal):

        def __init__(self, stockid, raw_data, bt_begin_date, bt_end_date, param_dict):
            
            indi = Query.tec_indicator(raw_data, ['RSI']) # Calculate Relative Strength Index (RSI)

            # Generate signals for the entire period
            def signal(x):
                if x < 20:
                    return True
                if x > 40:
                    return False
                return np.NAN
            indi = indi.set_index('date').map(signal)

            # Retain signals for the backtest period
            bt_indi = indi.query(f"date >= '{bt_begin_date}' and date < '{bt_end_date}'")

            self.set_signal(bt_indi)
    
    param_dict = {}
    
    # Run backtest
    res = LongBacktest.run(RSI, param_dict, rawdata, baseline, bt_begin_date, bt_end_date)

    # Analyze backtest results
    Evaluate.analysis_report(res, baseline, engine='quantstats')
```

**Using machine learning models to build a portfolio**：  
```python
from tradelearn.query.query import Query  # Import data query module
from tradelearn.trader.signal import Signal  # Import strategy signal class
from tradelearn.strategy.backtest.fund import LongBacktest  # Import portfolio backtest module
from tradelearn.strategy.evaluate.evaluate import Evaluate  # Import strategy evaluation module

import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier


if __name__ == '__main__':
    
    # Define data start and end dates
    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    # Query historical data for the Shanghai Composite Index as the benchmark
    baseline = Query.history_ohlc(symbol='000001.SS', start=tn_begin_date, end=tn_end_date, engine='yahoo')

    rawdata = None
    # Loop to query historical data for multiple stocks and process
    for i in range(10):
        temp = Query.history_ohlc(symbol='60052' + str(i), start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
        if temp is None:
            continue

        # Label the data
        temp['label'] = temp['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)
        rawdata = pd.concat([rawdata, temp], axis=0)

    # Define backtest start and end dates
    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'
    
    # Define Random Forest indicator class and use rolling prediction to generate trading signals
    class RandomForest(Signal):

        model_dict = {}  # Model dictionary

        def __init__(self, stockid, raw_data, bt_begin_date, bt_end_date, param_dict):
            fea_list = param_dict['fea_list']
            
            if not RandomForest.model_dict:
                # Build Random Forest models and save to the model dictionary
                for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                    bt_train_data = raw_data.query(f"date >= '{date - relativedelta(months=12 * 3)}' and date < '{date}'")
                    bt_x_train, bt_y_train = bt_train_data[fea_list], bt_train_data['label']

                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    model.fit(bt_x_train, bt_y_train)
                    RandomForest.model_dict[date.year] = model

            # Use models for prediction
            indi_df = None
            for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                pos_data = raw_data.query(f"code == '{stockid}' and date >= '{date}' and date < '{date + relativedelta(months=12 * 1)}'")
                bt_x_test = pos_data.set_index(['date'])[fea_list]
                pre_proba = RandomForest.model_dict[date.year].predict_proba(bt_x_test)[:, 1]
                indi_df = pd.concat([indi_df, pd.DataFrame(pre_proba, index=pos_data['date'])])

            self.set_signal(indi_df)

    # Feature list, excluding labels and code and date columns
    fea_list = rawdata.columns.drop(['label', 'code', 'date']).tolist()
    param_dict = {'fea_list': fea_list}
    
    # Run backtest
    res = LongBacktest.run(RandomForest, param_dict, rawdata, baseline, bt_begin_date, bt_end_date)
    
    # Analyze backtest results
    Evaluate.analysis_report(res, baseline, engine='quantstats')
```
## Method Guide
### Retrieving Raw Data
```python
from tradelearn.query.query import Query

rawdata = Query.history_ohlc(symbol='600520', start='2017-01-01', end='2022-06-22', adjust='hfq',engine='tdx')
```
| Parameter Name   | Data Type	   | Notes                                        |
|--------|--------|-------------------------------------------|
| symbol | string | Stock ticker                                   |
| start  | string | Start date                                      |
| end    | string | End date                                      |
| adjust | string | Adjustment method, can choose forward or backward adjustment, corresponding to 'qfq' and 'hfq' respectively     |
| engine | string | Third-party data source, can choose Yahoo Finance or Tongdaxin, corresponding to 'yahoo' and 'tdx' respectively |

### Factor Generation
```python
from tradelearn.query.query import Query

res = Query.alphas101(stock_data=rawdata, alpha_name=['alpha001'])
res = Query.alphas191(stock_data=rawdata, alpha_name=['alpha001'])
res = Query.tec_indicator(stock_data=rawdata, alpha_name=['ATR', 'RSI'])
```
| Parameter Name       | Data Type	      | Notes                                            |
|------------|-----------|-----------------------------------------------|
| stock_data | DataFrame | Target market data, required to have columns: open, low, high, close, volume, vwap |
| alpha_name | list      | List of factor or indicator names                                   |

### Exploratory Analysis
```python
from tradelearn.strategy.preprocess.explore.explore import Explore

Explore.analysis_report(data=rawdata, filename='res/explore.html')
```

| Parameter Name     | Data Type	      | Notes                 |
|----------|-----------|--------------------|
| data     | DataFrame | Target market data             |
| filename | string    | Path and name of the saved HTML file, optional |
### Factor Derivation
```python
from tradelearn.strategy.preprocess.derive.derive import Derive

res = Derive.generic_generate(data=rawdata)
```
| Parameter Name     | Data Type	      | Notes                |
|----------|-----------|-------------------|
| data     | DataFrame | Target market data            |
### Single Factor Test
```python
from tradelearn.strategy.examine.examine import Examine

Examine.single_factor(data=data, col='alpha001_101', filename='res/examine.html')
```
| Parameter Name     | Data Type	      | Notes                        |
|----------|-----------|---------------------------|
| data     | DataFrame | Target market data, required to have two or more stocks       |
| col      | string    | Target factor name                    |
| filename | string    | Path and name of the saved HTML file, optional        |
### Multi-Factor Comparison
```python
from tradelearn.strategy.examine.examine import Examine

res = Examine.factor_compare(data=data, f_col=None, ind=None, cir=None)
```
| Parameter Name  | Data Type	      | Notes                                 |
|-------|-----------|------------------------------------|
| data  | DataFrame | Target market data, required to have two or more stocks                |
| f_col | string    | List of factor names to compare, if None, all variables will be compared |
| ind   | string    | Industry field name for t-test calculation, optional             |
| cir   | string    | Market capitalization field name for t-test calculation, optional             |
### Causal Feature Selection
```python
from tradelearn.causal.blanket.blanket import Blanket

Blanket.fit_causal(data=rawdata, method='iamb', target_name='volume', is_discrete=False)
```
| Parameter Name        | Data Type	      | Notes                                |
|-------------|-----------|-----------------------------------|
| data        | DataFrame | Target market data                            |
| method      | string    | Selected causal feature selection algorithm, options are 'iamb' and 'pcmb' |
| target      | string    | Dependent variable name                             |
| alpha       | float     | Confidence level, generally set to 0.05 or 0.01           |
| is_discrete | bool      | If data is discrete, set to True           |
### Causal Graph Construction

```python
from tradelearn.causal.graph.graph import Graph

Graph.fit_causal(data=rawdata, method='pc', is_discrete=False, filename='res/pc.png')
```
| Parameter Name        | Data Type      | Notes                        |
|-------------|-----------|---------------------------|
| data        | DataFrame | Target market data                    |
| method      | string    | Selected causal graph construction algorithm, options are 'pc' and 'ges' |
| is_discrete | bool      | If data is discrete, set to True   |
| filename    | string    | Path and name of the saved causal graph, optional            |
### Optimal Model Selection

```python
from tradelearn.automl.automl import AutoML

model = AutoML.lazy_predict(data=data)
```
| Parameter Name     | Data Type	      | Notes                |
|----------|-----------|-------------------|
| data     | DataFrame | Target market data            |
### Backtest Validation

```python
from tradelearn.strategy.backtest.single import LongBacktest  # Template call for single target speculative trading strategy, choose one of two
from tradelearn.strategy.backtest.fund import LongBacktest    # Template call for multi-target portfolio strategy, choose one of two

res = LongBacktest.run(model_class=Example, param_dict=param_dict, raw_data=rawdata, base_line=baseline,
                       begin_date=bt_begin_date, end_date=bt_end_date, show_source=True)
```
| Parameter Name        | Data Type	      | Notes                        |
|-------------|-----------|---------------------------|
| model_class | Signal    | Implementation of signal class, user-defined             |
| param_dict  | dict      | Dictionary of parameters to pass to signal class              |
| raw_data    | DataFrame | Target market data                    |
| base_line   | DataFrame | Baseline market data                    |
| begin_date  | string    | Start date of backtest                    |
| end_date    | string    | End date of backtest                   |
| show_source | bool      | Whether to show strategy source code in HTML file, default is True |
### Strategy Evaluation

```python
from tradelearn.strategy.evaluate.evaluate import Evaluate

Evaluate.analysis_report(strat=res, baseline=baseline, filename='./evaluate.html', engine='quantstats')
```
| Parameter Name        | Data Type	      | Notes                                                      |
|-------------|-----------|---------------------------------------------------------|
| strat | dict      | Data dictionary returned by LongBacktest.run()                              |
| baseline  | DataFrame | Baseline market data                                                  |
| filename    | string    | Path and name of the generated HTML file, optional                                    |
| engine   | string    | Evaluation engine, options are pyfolio or quantstats, corresponding to 'pyfolio' and 'quantstats' respectively |
## Acknowledgements

- [Quantopian](https://github.com/quantopian)
- [Trevor Stephens](https://github.com/trevorstephens)
- [PyWhy](https://github.com/py-why)
- [DRo](https://github.com/mementum)
- [happydasch](https://github.com/happydasch)
- [baobao1997](https://github.com/baobao1997)

## Contact Information

WeChat Official Account：知守溪的收纳屋  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email：muyes88@gmail.com

