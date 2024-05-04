
class StopLoss:

    def __init__(self):
        pass

    @staticmethod
    def manage_risk(context, profit_threshold=0.2, stop_loss_threshold=0.05, trailing_stop_loss=0.03):
        if len(context.portfolio.positions) > 0:
            for stock in context.portfolio.positions.keys():
                position = context.portfolio.positions[stock]
                df = get_price(stock, start_date=position.init_time, end_date=context.previous_date, frequency='minute', fields=['high', 'low'], skip_paused=True)
                highest_price = df['high'].max()  # 从买入至今的最高价
                lowest_price = df['low'].min()  # 从买入至今的最低价
                current_price = position.price  # 持仓股票的当前价
                avg_cost = position.avg_cost  # 持仓股票的平均成本

                # 止盈逻辑
                if current_price >= avg_cost * (1 + profit_threshold):
                    log.info("{} 达到止盈线，执行平仓操作！".format(stock))
                    order_target_value(stock, 0)

                # 止损逻辑
                elif current_price <= avg_cost * (1 - stop_loss_threshold):
                    log.info("{} 达到止损线，执行平仓操作！".format(stock))
                    order_target_value(stock, 0)

                # 跟踪止损逻辑
                elif highest_price * (1 - trailing_stop_loss) >= current_price:
                    log.info("{} 触发跟踪止损，执行平仓操作！".format(stock))
                    order_target_value(stock, 0)

                else:
                    continue  # 如果没有触发止盈和止损条件，则继续持有股票
