class OrderObserver(bt.observers.Observer):
    lines = (
        'created',
        'expired'
    )

    plotinfo = dict(
        plot=True,
        subplot=True,
        plotlinelabels=True
    )

    plotlines = dict(
        created=dict(
            marker='*',
            markersize=8.0,
            color='lime',
            fillstyle='full'
        ),
        expired=dict(
            marker='s',
            markersize='8.0',
            color='red',
            fillstyle='full'
        )
    )

    def next(self):
        # 遍历 已创建的订单列表，broker已向策略发出相关通知事件
        for order in self._owner._orderspending:
            if order.data is not self.data:
                continue
            # 只处理买单
            if not order.isbuy():
                continue
            if order.status in [bt.Order.Accepted]:
                lg.error(order.created.price)
            if order.status in [bt.Order.Accepted, bt.Order.Submitted]:
                self.lines.created[0] = order.created.price
            elif order.status in [bt.Order.Expired]:
                self.lines.expired[0] = order.created.price
