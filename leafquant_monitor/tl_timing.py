"""Raw TL callback boundaries; no durations, execution facts or retained market objects."""
from collections.abc import Mapping
from tradelearn.core.observation_clock import CLOCK_DOMAIN

ORIGIN = 'TL_CALLBACK_ENTRY_V1'


def _point(event):
    if any(type(v) is not int or v <= 0 for v in (event.observed_mono_ns, event.observed_ts_ns)):
        return None
    return dict(observed_mono_ns=event.observed_mono_ns, observed_ts_ns=event.observed_ts_ns)


def _order(event):
    value = event.details.get('order')
    if value is None:
        args = event.details.get('args', ())
        value = args[0] if args else None
    return value if isinstance(value, Mapping) else {}


class TLBoundaryTiming:
    def __init__(self, logger):
        self.logger = logger
        self.clock_domain = CLOCK_DOMAIN

    def observe(self, event):
        if event.name == 'intent.submit_order' and event.phase == 'after':
            # CID is available after submission; the clock is the original entry,
            # carried by the neutral domain event, not recaptured here.
            clock = event.details.get('entry_clock')
            point = (dict(observed_mono_ns=clock[0], observed_ts_ns=clock[1])
                     if isinstance(clock, (tuple,list)) and len(clock)==2
                     and all(type(n) is int and n>0 for n in clock) else None)
            parent = event.callback_context or {}
            market = None
            ingress = parent.get('market_entry_clock')
            if (parent.get('callback') in {'business.next','business.prenext'} and
                    isinstance(ingress, (tuple,list)) and len(ingress)==2):
                market = dict(callback=parent.get('market_callback'),
                    observed_mono_ns=ingress[0], observed_ts_ns=ingress[1],
                    symbol=parent.get('symbol'), sequence=parent.get('sequence'),
                    ts_event_ns=parent.get('ts_event_ns'), business_callback=parent['callback'],
                    business_callback_mono_ns=parent.get('observed_mono_ns'),
                    business_callback_ts_ns=parent.get('observed_ts_ns'))
            self._record(event, 'tl_submit_entry', point, market)
        elif event.name == 'business.notify_order' and event.phase == 'before':
            self._record(event, 'tl_order_notification_entry', _point(event), None)
        # Successful market callbacks establish scope but write no per-tick log.
        # No per-CID cache or pending submission stack is retained by this observer.

    def _record(self, event, stage, point, market):
        if point is None:
            return
        order = _order(event)
        info = order.get('info') or {}
        cid = info.get('client_oid') or order.get('client_oid')
        if not isinstance(cid, str) or not cid:
            return  # Never guess a CID from a local ref or a similar order.
        fields = dict(context_type='tradelearn_timing', timing_origin=ORIGIN,
            clock_domain=self.clock_domain, stage=stage, client_order_id=cid,
            instrument_id=info.get('symbol'), tl_status=order.get('status'),
            market_callback_entry=market, native_evidence=(dict(info['native_evidence'], trade_ids=list(info['native_evidence'].get('trade_ids', ())))
                if isinstance(info.get('native_evidence'), Mapping) else None),
            tl_cumulative_filled_qty=(order.get('executed') or {}).get('size'))
        self.logger.record('order_context', fields, **point)
