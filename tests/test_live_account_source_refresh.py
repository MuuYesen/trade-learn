"""Plain BT users keep bounded native account observations without an SDK base."""
from types import SimpleNamespace as N
import pytest
pytest.importorskip('nautilus_trader')
pytest.importorskip('tl_store.ctrader.account_values')


@pytest.mark.parametrize('logging,heartbeat,respond',[(False,10,True),(True,10,True),(True,1,True),(False,10,False)])
def test_actual_plain_live_host_refreshes_formal_cTrader_facts_without_monitor(tmp_path,logging,heartbeat,respond):
    import asyncio,time
    from tradelearn import engine as bt
    from tradelearn_runtime.execution import create_execution_host
    from tl_store.ctrader.account_values import CTraderAccountValues
    from tl_store.ctrader.execution import project_funds
    from nautilus_trader import model as m
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from nautilus_trader.common import Environment,LoggerConfig
    from nautilus_trader.config import LiveNodeConfig,DataClientConfig,ExecutionClientConfig,StrategyConfig,LiveExecutionEngineConfig
    from nautilus_trader.live import LiveNode
    from nautilus_trader.live.clients import MarketDataClient,DataClientFactory,ExecutionClient,ExecutionClientFactory
    original=TestInstrumentProvider.btcusdt_perp_binance();definition=original.to_dict();definition['id']='BTCUSDT-PERP.CTRADER';ins=type(original).from_dict(definition)
    currency=m.Currency.from_str('USDT');aid=m.AccountId('CTRADER-100001');symbol=str(ins.id)
    bars=m.BarType.from_str(symbol+'-1-SECOND-LAST-EXTERNAL');observations=[];queries=[];business=[]
    class Data(MarketDataClient):
        async def _connect(self):self._handle_instrument(ins)
        async def _disconnect(self):pass
        async def _subscribe_bars(self,command):
            async def deliver():
                for pause in ((.02,3.3) if respond else (.02,)):
                    await asyncio.sleep(pause);now=self.clock.timestamp_ns()
                    self._handle_data(m.Bar(bars,*[ins.make_price(1000)]*4,ins.make_qty(10),now,now))
            self._delivery=asyncio.create_task(deliver())
        async def _unsubscribe_bars(self,command):
            self._delivery.cancel()
    class DF(DataClientFactory):
        def create(self,**kwargs):return Data(**kwargs,venue=ins.id.venue)
    class Exec(ExecutionClient):
        def publish(self):
            now=time.clock_gettime_ns(time.CLOCK_MONOTONIC) if hasattr(time,'CLOCK_MONOTONIC') else time.monotonic_ns()
            revision=len(queries);equity=10010+revision*10
            source=dict(schema_version=1,account=dict(ctid_trader_account_id=42,trader_login=100001,deposit_asset_id=2),currency='USDT',epoch=1,revision=revision,observed_utc_ns=self.clock.timestamp_ns(),started_mono_ns=now-1000000,completed_mono_ns=now-500000,clock_scope='local_observation_window',raw=dict(balance=str(equity-10),net_unrealized_pnl='10',equity=str(equity),used_margin='100',free_margin=str(equity-100)))
            balance,audit=project_funds(source)
            observations.append((equity-100,equity,revision))
            self.generate_account_state([balance],[],True,source['observed_utc_ns'],{'ctrader_funds':audit})
        async def _connect(self):self.publish()
        async def _disconnect(self):pass
        async def _query_account(self,command):
            queries.append(self.clock.timestamp_ns())
            if respond:self.publish()
    class EF(ExecutionClientFactory):
        def create(self,**kwargs):return Exec(**kwargs,venue=ins.id.venue,account_id=aid,account_type=m.AccountType.MARGIN,oms_type=m.OmsType.NETTING,base_currency=currency)
    class Business(bt.Strategy):
        def next(self):business.append((self.broker.getcash(),self.broker.getvalue(),observations[-1]))
    async def run():
        node=LiveNode.build('ACCOUNT-REFRESH-OFFLINE',LiveNodeConfig(environment=Environment.SANDBOX,trader_id=m.TraderId('TEST-001'),logging=LoggerConfig(bypass_logging=True),exec_engine=LiveExecutionEngineConfig(reconciliation=False),timeout_connection_secs=2,timeout_disconnection_secs=1,delay_post_stop_secs=0,data_clients={'CTRADER':DataClientConfig()},exec_clients={'CTRADER':ExecutionClientConfig()}),data_factories={'CTRADER':DF()},exec_factories={'CTRADER':EF()})
        if logging:
            from leafquant_monitor.logger import MonitorSettings
            monitor=MonitorSettings(source='ctrader',strategy_id='S-001',account_id=str(aid),state_dir=str(tmp_path/'monitor'),heartbeat_seconds=heartbeat,account_seconds=heartbeat)
        else:monitor=None
        host=create_execution_host(StrategyConfig(strategy_id=m.StrategyId('S-001'),oms_type=m.OmsType.NETTING),monitor=monitor,business_cls=Business,feeds={symbol:'bar'},instruments={symbol:ins.id},subscriptions=[bars],account_id=str(aid),account_provider=CTraderAccountValues(N(failure=None),100001),intent_path=tmp_path/'intent.sqlite',intent_identity='plain-refresh-offline')
        node.add_strategy(host)
        if logging:host.prepare_monitor()
        handle=node.handle();cache=node.cache
        async def stop():await asyncio.sleep(3.7 if respond else 4.3);handle.stop()
        stopper=asyncio.create_task(stop())
        try:
            async with asyncio.timeout(6):await node.run_async()
            if respond:
                await host.drain()
                assert host._worker_error is None
                assert 3<=len(queries)<=(5 if logging else 4)
                assert len(business)==2
                assert all((cash,equity)==expected[:2] for cash,equity,expected in business)
                assert business[-1][2][2]>=3
            else:
                with pytest.raises(RuntimeError):await host.drain()
                assert isinstance(host._worker_error,RuntimeError)
                assert str(host._worker_error)=='native account query did not publish a fresh observation'
                assert len(queries)==1 and len(business)==1
                assert host.frontend.failed
            assert not cache.orders()
            assert not any(name.startswith('__tl_account') for name in host._callback_clock.timer_names())
            assert hasattr(host,'logger') is logging
        finally:
            await stopper;host.intent_journal.close();node.dispose()
    asyncio.run(run())
