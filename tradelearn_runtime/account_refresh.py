"""Bounded native account reporting for the existing execution host.

Only timestamp-qualified providers activate this lifecycle. Queries use the
existing official NA account command and clock, never a second account ledger
or an asyncio polling loop. SDK requests share this dispatch when active.
"""
import asyncio
from datetime import timedelta


class NativeAccountRefresh:
    timer_name='__tl_account_source'
    deadline_name='__tl_account_query_deadline'

    def __init__(self,host):
        self.host=host
        self.handle=None
        self.pending=None
        self.active=False
        self.last_dispatch_ns=None
        self.deadline_due_ns=None

    def start(self):
        host=self.host
        if host._offline_test or host._account_provider is None:
            return
        try:self.loop=asyncio.get_running_loop()
        except RuntimeError:return  # Synchronous historical engines have no live source query.
        self.clock=host._callback_clock
        self.cache=host._callback_cache
        self.handle=self.loop.call_soon(self._prepare)

    def _prepare(self):
        self.handle=None
        host=self.host
        if host is None or host._stopping:return
        try:
            values=host.account()
            observed=values.get('source_observed_utc_ns') if isinstance(values,dict) else None
            if observed is None:return  # Static/custom provider has not declared a timed source.
            if type(observed) is not int or observed<=0:
                raise ValueError('qualified native account source observation clock required')
            self.active=True
            self.clock.set_timer(self.timer_name,timedelta(seconds=1),callback=self._on_timer)
        except BaseException as error:self._fail(error)

    def request(self):
        if self.host is None or not self.active or self.host._stopping or self.handle is not None:return False
        self.handle=self.loop.call_soon(self._run)
        return True

    def _on_timer(self,event):
        # rc7 holds Strategy/cache borrows during callbacks. Only schedule a
        # public dispatch after that callback returns; coalesce same-turn work.
        self.request()

    def _on_deadline(self,event):
        if self.pending is None:return
        # This exact owned native alert proves its scheduled deadline was
        # delivered; do not wait a whole extra heartbeat for wall-clock jitter.
        self.deadline_due_ns=self.pending[1]+3_000_000_000
        self.request()

    def _run(self):
        self.handle=None
        host=self.host
        if host is None or host._stopping:return
        try:
            from nautilus_trader.model import AccountId
            account=self.cache.account(AccountId(host._account_id))
            if account is None or str(account.id)!=host._account_id:
                raise ValueError('native account query lost bound account')
            identity=str(account.last_event.event_id)
            now=max(int(self.clock.timestamp_ns()),self.deadline_due_ns or 0)
            if self.last_dispatch_ns is not None and now-self.last_dispatch_ns<500_000_000:
                return  # A nearby SDK tick must not double the same account query.
            if self.pending is not None:
                previous,started=self.pending
                if identity==previous:
                    if now-started>=3_000_000_000:
                        raise RuntimeError('native account query did not publish a fresh observation')
                    return  # One outstanding query, not a backlog of timed commands.
                self.pending=None
                self.deadline_due_ns=None
                if self.deadline_name in self.clock.timer_names():self.clock.cancel_timer(self.deadline_name)
            host.account()  # Existing provider checks freshness, identity and raw reconciliation.
            self.pending=(identity,now)
            self.last_dispatch_ns=now
            self.clock.set_time_alert_ns(self.deadline_name,now+3_000_000_000,callback=self._on_deadline)
            host.query_account(AccountId(host._account_id))
        except BaseException as error:self._fail(error)

    def _fail(self,error):
        host=self.host
        if host is None:return
        host._worker_error=error
        host._stopping=True
        self.stop()
        frontend=getattr(host,'frontend',None)
        if frontend is not None:
            frontend._failed=True
            try:frontend.cleanup(error)
            except BaseException as secondary:
                from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                _safe_cleanup_diagnostic(error,secondary)
        callback=getattr(host,'_failure_callback',None)
        if callback is not None:
            try:callback()
            except BaseException as secondary:
                from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
                _safe_cleanup_diagnostic(error,secondary)

    def stop(self):
        if self.handle is not None:self.handle.cancel()
        self.handle=None
        if self.active:
            for name in (self.timer_name,self.deadline_name):
                if name in self.clock.timer_names():self.clock.cancel_timer(name)
        self.active=False
        self.pending=None
        self.host=None
