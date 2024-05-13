import pandas as pd

import tradelearn.trader as bt
from dateutil.relativedelta import relativedelta

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

from tradelearn.trader.plot.tabs.log import init_log_tab
init_log_tab([__name__], logging.INFO)


class LongBacktest:

    @staticmethod
    def run(test_data, base_line, begin_date, end_date, model_class, feature_list, **kwargs):

        cerebro = bt.Cerebro()

        bt_base_data = base_line[['open', 'high', 'low', 'close', 'volume', 'date', 'code']]
        bt_base_data = bt_base_data.query(f"date >= '{begin_date}' and date < '{end_date}'").set_index(['date'])
        data = bt.feeds.PandasData(dataname=bt_base_data, name='baseline', datetime=None, open=0, high=1,
                                   low=2, close=3, volume=4, openinterest=-1)
        cerebro.adddata(data, name='baseline')

        bt_test_data = test_data[['open', 'high', 'low', 'close', 'volume', 'date', 'code']]
        bt_test_data = bt_test_data.query(f"date >= '{begin_date}' and date < '{end_date}'")
        for symbol in bt_test_data['code'].unique():
            data = bt_test_data.query(f"code == '{symbol}'")
            data = data.set_index(['date'])
            data = bt.feeds.PandasData(dataname=data, name=symbol, datetime=None, open=0, high=1,
                                       low=2, close=3, volume=4, openinterest=-1)  # 不断添加每个股票数据
            cerebro.adddata(data, name=symbol)

        cerebro.broker.setcash(10000000.0)
        cerebro.broker.setcommission(commission=0.001)  # 手续费

        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='_Pyfolio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')

        cerebro.addstrategy(Long, bt_data=test_data, model_class=model_class, feature_list=feature_list, begin_date=begin_date, end_date=end_date, **kwargs)

        print('初始资金: %.2f' % cerebro.broker.getvalue())
        results = cerebro.run()
        print('最终资金: %.2f' % cerebro.broker.getvalue())

        cerebro.plot(output_mode='show', style='bar', show_source=True)

        strat = results[0]
        print('夏普比率:', strat.analyzers._SharpeRatio.get_analysis())
        print('回撤指标:', strat.analyzers._DrawDown.get_analysis())

        return strat


class Long(bt.Strategy):

    def _getminperstatus(self):
        return -1

    def log(self, txt):
        datetime = self.data.datetime[0]
        dt = bt.num2date(datetime)
        logger.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self, bt_data, model_class, feature_list, begin_date, end_date, **kwargs):
        self.inds = {}
        for symbol in bt_data.query(f"date >= '{begin_date}' and date < '{end_date}'")['code'].unique():
            self.inds[symbol] = model_class(bt_begin_date=begin_date,
                                            bt_end_date=end_date,
                                            fina_data=bt_data,
                                            fea_list=feature_list,
                                            stockid=symbol,
                                            **kwargs)
        self.last = []
        self.order_list = []
        self.last_date = None

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买入执行, %.2f' % order.executed.price)
            elif order.issell():
                self.log('卖出执行, %.2f' % order.executed.price)

            self.bar_executed = len(self)
            self.log('=======订单, %.2f' % self.bar_executed)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单已取消/保证金/已拒绝')

        self.order = None

    def notify_trade(self, trade):

        self.trade = trade
        if trade.isclosed:
            self.log('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f' %(trade.pnl, trade.pnlcomm, trade.commission))

    def next(self):

        cur_date = self.datas[0].datetime.date(0)  # 确定当日投资组合

        if self.last_date is not None and self.last_date == cur_date:
            return

        if self.last_date is not None and cur_date < self.last_date + relativedelta(days=5):
            return

        for order in self.order_list:  # 取消未成交的订单
            self.cancel(order)
        self.order_list = []

        proba_dict = {}
        for symbol in self.inds.keys():
            proba_dict[symbol] = self.inds[symbol].lines.model_indi[0]
        data = pd.Series(proba_dict)
        long_list = data[(data >= data.quantile(0.8)) & (data <= data.quantile(1.0))].index.tolist()

        for data_id in self.last:  # 若上期股票未出现在本期交易列表中，则平仓
            if data_id not in long_list:
                order = self.close(data=data_id)
                self.order_list.append(order)

        if len(long_list):  # 如果存在做多信号，则开仓
            buy_weight = [(1 - 0.05) / len(long_list)]*len(long_list)

            for i, data_id in enumerate(long_list):
                target_value = buy_weight[i] * self.broker.get_value()
                data = self.getdatabyname(data_id)
                try:
                    size = int(abs(target_value / data.open[1] // 100 * 100))  # 下一天的开盘价(应为真实价格，而非处理后)买入，100的整数倍
                    price = data.open[1]
                except:
                    try:
                        size = int(abs(target_value / data.close[0] // 100 * 100))
                        price = data.close[0]
                    except:
                        size = 0
                order = self.order_target_size(data=data_id, target=size, price=price)

                self.order_list.append(order)

        self.last = long_list
        self.last_date = cur_date
