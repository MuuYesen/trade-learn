"""TL entry clocks are raw observations, never synthetic execution facts."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from tradelearn.engine import Strategy, StrategyRuntime
from tradelearn.core import strategy_events as events
from leafquant_monitor.tradelearn import TradeLogStrategy, TradeLogConfig
from leafquant_monitor.logger import TradeLogger, MonitorSettings
from tradelearn_runtime.broker import NautilusBroker

class Recorder:
    def __init__(self): self.rows=[]; self.fail=False
    def append(self, raw):
        if self.fail: raise OSError('synthetic recorder failure')
        self.rows.append(json.loads(raw)); return True
    def status(self): return json.dumps(dict(run_id='tl-clock-test',accepted=len(self.rows),durable=len(self.rows),error=None))
    flush=close=status

class ClockTests(unittest.TestCase):
    def test_callback_clock_precedes_snapshot_and_user_callback(self):
        times={}
        class Item:
            def __strategy_snapshot__(self):
                times['snapshot']=time.monotonic_ns();return {'status':2}
        class S(Strategy):
            def notify_order(self, order): times['callback']=time.monotonic_ns()
        s=S();seen=[];events.add_observer(s,seen.append,events={'business.notify_order'},phases=('before',))
        events.invoke_callback(s,'notify_order',Item());e=seen[0]
        self.assertIsInstance(getattr(e,'observed_mono_ns',None),int)
        self.assertLessEqual(e.observed_mono_ns,times['snapshot']);self.assertLessEqual(e.observed_mono_ns,times['callback'])
        self.assertIsInstance(e.observed_ts_ns,int)
    def test_no_observers_do_not_read_observation_clocks(self):
        s=Strategy()
        with patch('time.monotonic_ns',side_effect=AssertionError('unexpected clock')),patch('time.time_ns',side_effect=AssertionError('unexpected clock')):
            events.invoke_callback(s,'notify_order',None)

class TimingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.rec=Recorder()
        self.logger=TradeLogger(MonitorSettings('BINANCE','tl-test','BINANCE-TEST',self.directory.name),recorder=self.rec)
        self.config=TradeLogConfig(context=dict(environment='paper',run_id='tl-context',source='BINANCE',strategy_id='tl-test',account_id='BINANCE-TEST',state_dir=self.directory.name),logger=self.logger)
    def tearDown(self): self.logger.close();self.directory.cleanup()
    def runtime(self, body=None, notify=None):
        config=self.config
        class S(TradeLogStrategy):
            monitor_config=config
            def next(self):
                if body: body(self)
                else: self.buy(size=.01,tif='ioc')
            def notify_order(self,order):
                if notify:notify(self,order)
        gateway=NS(lookup_order=lambda cid:None,submit=lambda request,**kw:None)
        broker=NautilusBroker(gateway);r=StrategyRuntime(S,feeds={'BTC':'tick'},broker=broker);r.start();return r,broker
    def timing(self,stage=None):
        rows=[r for r in self.rec.rows if r['fields'].get('context_type')=='tradelearn_timing']
        return rows if stage is None else [r for r in rows if r['fields']['stage']==stage]
    def test_automatic_cid_market_submit_and_synchronous_submitted_notification(self):
        r,b=self.runtime();r.step_market('BTC',NS(ts_event=100),kind='tick')
        submits=self.timing('tl_submit_entry');self.assertEqual(len(submits),1)
        s=submits[0];cid=next(iter(b._orders));self.assertEqual(s['fields']['client_order_id'],cid)
        market=s['fields']['market_callback_entry'];self.assertEqual(market['callback'],'runtime.step_market');self.assertEqual(market['symbol'],'BTC')
        self.assertLessEqual(market['observed_mono_ns'],s['observed_mono_ns'])
        notifications=self.timing('tl_order_notification_entry');self.assertEqual(len(notifications),1)
        self.assertEqual(notifications[0]['fields']['client_order_id'],cid)
        self.assertLessEqual(s['observed_mono_ns'],notifications[0]['observed_mono_ns'])
        for status,qty,price in [('ACCEPTED',0,None),('PARTIALLY_FILLED',.005,10),('FILLED',.01,10)]:
            b.apply(cid,status=status,filled_qty=qty,avg_price=price,event_id=status,commission=0)
        self.assertEqual(len(self.timing('tl_order_notification_entry')),4)
        self.assertTrue(all(x['kind']=='order_context' for x in self.timing()))
        self.assertFalse(any(x['kind'] in ('order_accepted','order_filled') for x in self.rec.rows));r.stop()
    def test_market_origin_precedes_feed_and_indicator_preparation(self):
        r,b=self.runtime();seen=[];advance=r._advance_market
        def observed(*args):seen.append(time.monotonic_ns());return advance(*args)
        r._advance_market=observed;r.step_market('BTC',NS(ts_event=100),kind='tick')
        market=self.timing('tl_submit_entry')[0]['fields']['market_callback_entry']
        self.assertLessEqual(market['observed_mono_ns'],seen[0]);r.stop()

    def test_direct_next_outside_market_step_cannot_reuse_old_ingress(self):
        r,b=self.runtime();r.step_market('BTC',NS(ts_event=100),kind='tick');r.invoke_business('next')
        self.assertIsNone(self.timing('tl_submit_entry')[-1]['fields']['market_callback_entry']);r.stop()

    def test_submit_clock_is_before_data_resolution(self):
        r,b=self.runtime();s=r.strategy;times=[];resolve=s._resolve_data
        def slow(*args,**kwargs):times.append(time.monotonic_ns());time.sleep(.002);return resolve(*args,**kwargs)
        s._resolve_data=slow;r.step_market('BTC',NS(ts_event=100),kind='tick')
        self.assertEqual(len(self.timing('tl_submit_entry')),1)
        self.assertLessEqual(self.timing('tl_submit_entry')[0]['observed_mono_ns'],times[0]);r.stop()
    def test_submit_from_notification_has_no_stale_market_cause(self):
        def notify(s,o):
            if o.info.get('submission_state')=='ACCEPTED':s.buy(size=.01,tif='ioc')
        r,b=self.runtime(notify=notify);r.step_market('BTC',NS(ts_event=100),kind='tick');cid=next(iter(b._orders));b.apply(cid,status='ACCEPTED',filled_qty=0,event_id='ack')
        self.assertEqual(len(self.timing('tl_submit_entry')),2)
        self.assertIsNone(self.timing('tl_submit_entry')[-1]['fields']['market_callback_entry']);r.stop()
    def test_market_without_orders_is_not_written_per_tick(self):
        r,b=self.runtime(body=lambda s:None)
        for n in range(50):r.step_market('BTC',NS(ts_event=n),kind='tick')
        self.assertEqual(self.timing(),[]);r.stop()
    def test_copied_callback_context_expires_before_later_submit(self):
        from contextvars import copy_context
        saved=[]
        r,b=self.runtime(body=lambda strategy:saved.append(copy_context()))
        r.step_market('BTC',NS(ts_event=100),kind='tick')
        saved[0].run(r.strategy.buy,size=.01,tif='ioc')
        self.assertIsNone(self.timing('tl_submit_entry')[0]['fields']['market_callback_entry']);r.stop()
    def test_bridge_passes_raw_notification_identity_without_new_business_events(self):
        import inspect
        self.assertIn('native_evidence',inspect.signature(NautilusBroker.apply).parameters)
        r,b=self.runtime();r.step_market('BTC',NS(ts_event=100),kind='tick');cid=next(iter(b._orders))
        evidence=dict(instrument_id='BTC',account_id='BINANCE-TEST',strategy_id='tl-test',trade_ids=['trade'],cumulative_filled_qty='0.01',status='FILLED')
        b.apply(cid,status='FILLED',filled_qty=.01,avg_price=10,event_id='event',native_evidence=evidence)
        self.assertEqual(self.timing('tl_order_notification_entry')[-1]['fields']['native_evidence'],evidence)
        before=len(self.timing());self.assertFalse(b.apply(cid,status='FILLED',filled_qty=.01,avg_price=10,event_id='event',native_evidence=evidence));self.assertEqual(len(self.timing()),before);r.stop()

    def test_logging_failure_does_not_change_order_submission(self):
        r,b=self.runtime();self.rec.fail=True
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');r.step_market('BTC',NS(ts_event=100),kind='tick')
        self.assertEqual(len(b._orders),1);self.rec.fail=False;r.stop()

if __name__=='__main__':unittest.main()
