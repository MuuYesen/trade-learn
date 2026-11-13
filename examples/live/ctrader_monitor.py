"""Frozen Sentinel preview for the unified TL candidate; not published 0.3.4.

Replace the personal account, cTrader application/OAuth and restricted sender
values below with your own. This file does not change IAM permissions.
"""
from tradelearn import engine as bt


class MyStrategy(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.ma = bt.talib.SMA(self.data.close, timeperiod=self.p.period)

    def next(self):
        ...


if __name__ == "__main__":
    store = bt.stores.NautilusStore(
        adapter="ctrader",
        account=12345678,  # Replace with your own cTrader numeric account ID.
        client_id="YOUR_CTRADER_APPLICATION_CLIENT_ID",
        client_secret="YOUR_CTRADER_APPLICATION_CLIENT_SECRET",
        access_token="YOUR_CTRADER_ACCESS_TOKEN",
        refresh_token="YOUR_CTRADER_REFRESH_TOKEN",
        practice=True,
        trader_id="DEMO-001",
        strategy_id="DEMO-001",
    )
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.setbroker(store.getbroker())
    cerebro.adddata(store.getdata(
        dataname="BTCUSD", timeframe=bt.TimeFrame.Seconds, compression=1, price="mid",
    ))
    cerebro.addstrategy(MyStrategy)
    cerebro.run(monitor={
        "endpoint": "https://sqs.ap-southeast-1.amazonaws.com/123456789012/your-registered-queue.fifo",
        "credentials": {
            "access_key_id": "YOUR_RESTRICTED_SENDER_ACCESS_KEY_ID",
            "secret_access_key": "YOUR_RESTRICTED_SENDER_SECRET_ACCESS_KEY",
        },
    })
