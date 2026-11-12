"""Cerebro component lifecycle on the one Runtime; no scheduler or execution state."""
from types import SimpleNamespace


class _ObservationWindow(list):
    """Retain the existing list view while bounding framework sampling appends."""
    def __init__(self, capacity, values):
        self._capacity = capacity
        super().__init__(values[-capacity:])

    def append(self, value):
        if len(self) >= self._capacity:
            del self[:len(self) - self._capacity + 1]
        super().append(value)


def strategy_spec(engine):
    """Normalize existing SignalStrategy options without changing signal execution."""
    specs=list(engine.strats)
    if not engine.signals:
        if len(specs)!=1:raise ValueError('native Store requires one strategy')
        return specs[0]
    from tradelearn.engine.signal import SignalStrategy
    selected=engine._signal_strat
    if selected is not None:
        if specs:raise ValueError('native Store requires one signal strategy')
        cls,args,parameters=selected
    elif len(specs)==1 and issubclass(specs[0][0],SignalStrategy):
        cls,args,parameters=specs[0]
    elif not specs:
        cls,args,parameters=SignalStrategy,(),{}
    else:
        raise ValueError('native Store requires one signal strategy')
    if not issubclass(cls,SignalStrategy):
        raise TypeError('signal_strategy must use the existing SignalStrategy')
    return cls,args,dict(parameters,signals=list(engine.signals),
        _accumulate=engine._signal_accumulate,_concurrent=engine._signal_concurrent)


def native_sizer(engine):
    """Bind the existing BT Sizer, without replacing it with a quantity-only type."""
    from tradelearn.engine.sizers import FixedSize, Sizer

    cls, options = engine._sizer_spec
    if not isinstance(cls, type) or not issubclass(cls, Sizer):
        raise TypeError('native Store requires a Backtrader-compatible TL Sizer')
    configured = cls(**options)
    if cls is FixedSize:
        # FixedSize does not consume cash or a commission estimate. Keep its
        # public type/params/setsizing, without requesting unknown account money.
        configured.getsizing = lambda data, isbuy: configured._getsizing(None, None, data, isbuy)
    return configured


def validate_components(engine,*,account_provider):
    from tradelearn.engine.observers import Value,DrawDown,TimeReturn
    native_sizer(engine)
    for spec in engine.analyzers.values():
        cls=spec[0]
        if getattr(cls,'_metric_key','') and not getattr(cls,'_is_streaming',False):
            raise NotImplementedError('post analyzer requires backtest Stats; use a streaming analyzer for native runs')
    if account_provider is None and any(getattr(spec[0], '_requires_account_value', False)
                                       for spec in engine.analyzers.values()):
        raise ValueError('value analyzer requires an authoritative account provider')
    if account_provider is None and any(issubclass(spec[0],(Value,DrawDown,TimeReturn))
                                       for spec in engine.observers.values()):
        raise ValueError('value observer requires an authoritative account provider')


class LiveComponents:
    def __init__(self,engine,runtime,*,account_provider=None):
        validate_components(engine,account_provider=account_provider)
        self.engine,self.runtime,self.strategy=engine,runtime,runtime.strategy
        self.bars=self.broker_events=self.fill_events=self.closed_trades=0
        self._finished=False
        self._started=[]
        self._start_done=False
        engine._attach_analyzers(self.strategy)
        # Default backtest statistics require a real value source in live mode.
        view=SimpleNamespace(observers=engine.observers,
            stdstats=engine.stdstats and account_provider is not None)
        type(engine)._attach_observers(view,self.strategy)
        capacity = max((feed._buffer.capacity for feed in runtime.feeds.values()), default=512)
        from tradelearn.engine.observers import BuySell,Trades
        for observer in self.strategy.observers.values():
            if type(observer.values) is list:
                observer.values = _ObservationWindow(capacity, observer.values)
            # Counts are projections of deduplicated native events in this run,
            # not fabricated _fills/_trades collections or another order ledger.
            if type(observer)._observe is BuySell._observe:
                observer._observe=lambda:self.fill_events
            elif type(observer)._observe is Trades._observe:
                observer._observe=lambda:self.closed_trades
        from tradelearn.backtest.engine import _build_analyzer_bar_callbacks,_build_observer_nexts
        self._bar_callbacks=_build_analyzer_bar_callbacks(self.strategy)
        self._observer_callbacks=_build_observer_nexts(self.strategy)
        runtime._component_start_callback=self.start

    def start(self):
        if self._start_done:return
        self._start_done=True
        for analyzer in self.strategy.analyzers.values():
            self._started.append(analyzer)
            for name in ('on_start','start'):
                callback=getattr(analyzer,name,None)
                if callable(callback):self.runtime.invoke_component(callback)
        for observer in self.strategy.observers.values():
            self._started.append(observer)
            self.runtime.invoke_component(observer.start)

    def after_market(self):
        self.bars+=1
        if self.runtime._warming_up():return
        from tradelearn.backtest.engine import _analyzer_bar_step,_observer_step
        _analyzer_bar_step(self.strategy,self._bar_callbacks)
        _observer_step(self.runtime,self._observer_callbacks)

    def order(self,order):
        for analyzer in self.strategy.analyzers.values():
            callback=getattr(analyzer,'notify_order',None)
            if callable(callback):self.runtime.invoke_component(callback,order)
            callback=getattr(analyzer,'on_order',None)
            if callable(callback):self.runtime.invoke_component(callback,order)

    def native_order(self,event,order):
        from tradelearn.core.broker_events import BrokerEvent
        name=type(event).__name__
        kind='status'
        if name=='OrderFilled':
            self.fill_events+=1
            kind='partial' if order.status==order.Partial else 'fill'
        elif name=='OrderCanceled':kind='cancel'
        elif name in ('OrderRejected','OrderDenied'):kind='reject'
        self.broker_events+=1
        normalized=BrokerEvent(kind,str(event.client_order_id),payload=order,
            status=order.info.get('native_status'))
        for analyzer in self.strategy.analyzers.values():
            callback=getattr(analyzer,'on_broker_event',None)
            if callable(callback):self.runtime.invoke_component(callback,normalized)

    def trade(self,trade):
        if trade.isclosed:self.closed_trades+=1
        for analyzer in self.strategy.analyzers.values():
            callback=getattr(analyzer,'notify_trade',None)
            if callable(callback):self.runtime.invoke_component(callback,trade)
            callback=getattr(analyzer,'on_trade',None)
            if callable(callback):self.runtime.invoke_component(callback,trade)

    def finish(self):
        if self._finished:return
        self._finished=True
        errors=[];results={}
        def invoke(callback,*args):
            try:return self.runtime.invoke_component(callback,*args)
            except BaseException as error:errors.append(error)
        stats=dict(bars=self.bars,broker_events=self.broker_events,mode='live')
        for name,analyzer in self.strategy.analyzers.items():
            if analyzer not in self._started:continue
            callback=getattr(analyzer,'on_end',None)
            if callable(callback):invoke(callback,stats)
            callback=getattr(analyzer,'stop',None)
            if callable(callback):invoke(callback)
            analyzer._stopped=True
            results[name]=invoke(analyzer.get_analysis)
        for component in self._started:
            callback=getattr(component,'stop',None) or getattr(component,'_stop',None)
            if callable(callback) and not getattr(component,'_stopped',False):invoke(callback)
        self.strategy.analyzer_results=self.engine.analyzer_results=results
        if errors:
            from tradelearn.backtest.event_runtime import _safe_cleanup_diagnostic
            primary=errors.pop(0)
            for error in errors:_safe_cleanup_diagnostic(primary,error)
            raise primary
