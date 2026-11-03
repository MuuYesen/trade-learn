"""Public owned facts. Logging never drives inventory or execution accounting."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import hashlib
import logging
import math
from pathlib import Path
import re
import time
import uuid

from .snapshots import account_evidence, framework_snapshot, public_fields, public_value, native_info_value


@dataclass(frozen=True)
class MonitorSettings:
    source: str
    strategy_id: str
    account_id: str
    state_dir: str
    account_source: str = 'framework'
    heartbeat_seconds: int = 10
    account_seconds: int = 30
    capacity: int = 4096
    segment_bytes: int = 8 * 1024 * 1024
    max_disk_bytes: int = 1024 * 1024 * 1024
    max_event_history: int = 100_000
    archive_window_seconds: int = 300
    # Sender gets a separate private config. A URL alone grants no permission.
    sender_config: str | None = None
    receipt_origin: str | None = None
    archive_binding_id: str | None = None
    framework_strategy_id: str | None = None
    framework_account_id: str | None = None

    def __post_init__(self):
        identities = [self.source, self.strategy_id, self.account_id]
        identities.extend(v for v in (self.framework_strategy_id, self.framework_account_id) if v is not None)
        for value in identities:
            if not isinstance(value, str) or not value or value.strip() != value or re.search(r'[\x00-\x1f\x7f]', value):
                raise ValueError('invalid monitor identity')
        if self.account_source not in ('framework', 'external', 'framework_and_external'):
            raise ValueError('invalid account observation source')
        if any(type(v) is not int or v <= 0 for v in (self.heartbeat_seconds, self.account_seconds,
                self.capacity, self.segment_bytes, self.max_disk_bytes, self.max_event_history, self.archive_window_seconds)):
            raise ValueError('positive monitor budgets required')
        if not isinstance(self.state_dir, str) or not self.state_dir:
            raise ValueError('durable monitor state directory required')
        if self.sender_config is not None:
            from urllib.parse import urlsplit
            if not isinstance(self.receipt_origin,str):raise ValueError('trusted HTTPS receipt origin required')
            origin=urlsplit(self.receipt_origin)
            if (origin.scheme!='https' or not origin.hostname or origin.username is not None or origin.password is not None
                    or origin.path not in ('','/') or origin.query or origin.fragment
                    or not isinstance(self.archive_binding_id,str) or re.fullmatch(r'[A-Za-z0-9_-]{1,128}',self.archive_binding_id) is None):
                raise ValueError('trusted receipt origin and archive binding required')


    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text()))


EVENT_FIELDS = ('client_order_id', 'venue_order_id', 'trade_id', 'instrument_id', 'account_id', 'strategy_id',
                'position_id', 'side', 'avg_px_open', 'avg_px_close', 'realized_pnl', 'quote_currency',
                'settlement_currency', 'order_side', 'order_type', 'time_in_force', 'last_qty', 'last_px',
                'quantity', 'price', 'currency', 'commission', 'liquidity_side', 'reason', 'reconciliation',
                'ts_init', 'ts_event', 'filled_qty', 'leaves_qty', 'trigger_price', 'due_post_only',
                'voided_qty', 'commission_voided', 'correction_id', 'is_reopened', 'info')


class TradeLogger:
    def __init__(self, settings, *, recorder=None, sender_factory=None):
        if sender_factory is not None and (not callable(sender_factory) or settings.sender_config is None):
            raise ValueError('trusted sender factory requires explicit sender configuration')
        self.settings = settings
        if recorder is None:
            from tradelearn._rust import Recorder
            recorder = Recorder(str(Path(settings.state_dir) / 'segments'), settings.capacity,
                                settings.segment_bytes, settings.max_disk_bytes,
                                archive_window_seconds=settings.archive_window_seconds)
        self.recorder = recorder
        self.run_id = json.loads(recorder.status())['run_id']
        self._sequence = 0
        self._monotonic_ns = time.monotonic_ns
        self._error = None
        self._order_traces = {}
        self._order_references = {}
        self._active_calculation = None
        self._pending_submits = {}
        self._fill_events = OrderedDict()
        self._fills = OrderedDict()
        self._account_events = OrderedDict()
        self._account_info_carriers = {}
        self._account_last_event = None
        self._account_gap_reason = None
        self._account_recovery_ambiguous = False
        self._restore_fill_revisions()
        self.sender = None
        if settings.sender_config is not None:
            from .sender import BackgroundSender
            self.sender = (sender_factory(settings, flush=self.flush) if sender_factory is not None
                           else BackgroundSender(settings, flush=self.flush))
            self.sender.start()

    @staticmethod
    def _execution_key(fields):
        values = tuple(fields.get(name) for name in ('instrument_id', 'client_order_id', 'trade_id'))
        if any(not isinstance(v, str) or not v for v in values):
            raise ValueError('execution requires instrument, client order and native trade identity')
        return values

    def _restore_fill_revisions(self):
        # Recovery happens before strategy startup, never on a trading callback.
        # Keep a bounded recent index; old raw segments remain authoritative.
        from .recovery import RecoveryIndex
        def durable_records():
            yield from RecoveryIndex(self.settings).records()
            for path in sorted((Path(self.settings.state_dir) / 'segments').glob('*.log')):
                with path.open('rb') as stream:
                    yield from stream
        account_cursor = None
        for raw in durable_records():
            if not raw.endswith(b'\n'):
                continue  # Interrupted tail is archive-only, never a recovered business fact.
            if len(raw) > 96 * 1024:
                raise ValueError('corrupt durable source segment')
            fact = json.loads(raw)
            from leafquant_monitor.codec import decode_fact
            fact = decode_fact(fact)
            if any(fact.get(k) != getattr(self.settings, k)
                   for k in ('source', 'strategy_id', 'account_id')):
                raise ValueError('source segment identity differs from connector binding')
            fields = fact['fields']
            expected_native = {
                'strategy_id': self.settings.framework_strategy_id or self.settings.strategy_id,
                'account_id': self.settings.framework_account_id or self.settings.account_id,
            }
            for name, expected in expected_native.items():
                mapped = fields.get('framework_' + name)
                original = fields.get(name) if fact.get('kind', '').startswith(('order_', 'position_')) else None
                if any(value is not None and value != expected for value in (mapped, original)):
                    raise ValueError('source segment native identity differs from connector binding')
            if fact.get('kind') == 'order_context' and fields.get('context_type') == 'submit_reference':
                reference=self._validate_reference(fields['reference'])
                cid=fields['client_order_id']
                if (fields['origin_run_id']!=fact['run_id'] or fields['reference_revision']!=0
                        or fields['reference_id']!=self._reference_id(cid, reference, fields['origin_run_id'])):
                    raise ValueError('durable order reference digest or origin conflict')
                bound=dict(reference,reference_id=fields['reference_id'],reference_revision=fields['reference_revision'],
                           origin_run_id=fields['origin_run_id'])
                if cid in self._order_references and self._order_references[cid]!=bound:
                    raise ValueError('durable order reference identity conflict')
                self._order_references[cid]=bound
                if len(self._order_references)>self.settings.max_event_history:
                    raise ValueError('durable order reference history exceeds configured budget')
                self._order_traces[cid]=fields.get('trace_id')
            if fact.get('kind') == 'account_observation' and fields.get('scope') == 'framework_account_event':
                self._remember_account_event(fields)
                try:
                    seq=int(fact['record_id'].rsplit(':',1)[1])
                    order=(fact['observed_ts_ns'],seq)
                except (KeyError,ValueError,TypeError,IndexError):
                    raise ValueError('invalid durable account event cursor')
                if account_cursor is not None and order[0]==account_cursor[0][0] and fact['run_id']!=account_cursor[1]:
                    self._account_recovery_ambiguous=True
                if account_cursor is None or order>account_cursor[0]:
                    account_cursor=(order,fact['run_id'],fields['native_event_id'])
            if fact.get('kind') not in ('order_filled', 'order_fill_voided'):
                continue
            trade = fields.get('trade_id')
            if not trade or not fields.get('event_id'):
                continue
            self._fill_events[fields['event_id']] = dict(fields)
            if len(self._fill_events) > self.settings.max_event_history:
                raise ValueError('durable native event history exceeds configured identity budget')
            execution = self._execution_key(fields)
            old = self._fills.get(execution)
            revision = fields.get('revision', 0)
            if old is None or revision > old['revision']:
                self._fills[execution] = dict(event_id=fields['event_id'], revision=revision, fields=fields)
                self._fills.move_to_end(execution)
            while len(self._fills) > self.settings.capacity:
                self._fills.popitem(last=False)

        if account_cursor is not None:
            self._account_last_event=account_cursor[2]

    def record(self, kind, fields, *, source_ts_ns=None, observed_ts_ns=None, observed_mono_ns=None):
        self._sequence += 1
        observed_ts_ns = time.time_ns() if observed_ts_ns is None else observed_ts_ns
        observed_mono_ns = self._monotonic_ns() if observed_mono_ns is None else observed_mono_ns
        if any(type(v) is not int or v < 0 for v in (observed_ts_ns, observed_mono_ns)):
            raise ValueError('invalid observation clock')
        if source_ts_ns is not None and (type(source_ts_ns) is not int or source_ts_ns < 0):
            raise ValueError('invalid source clock')
        if kind in {'risk_check','exception'}:
            field='risk_check_id' if kind=='risk_check' else 'exception_id'
            fields=dict(fields)
            fields.setdefault(field,f'{self.run_id}:{kind}:{uuid.uuid4().hex}')
        source_fields = dict(fields)
        info_present = (kind == 'account_observation' and source_fields.get('scope') in ('framework_account_event','framework_cache') and 'native_info' in source_fields)
        native_info = source_fields.pop('native_info') if info_present else None
        encoded_fields = public_value(source_fields)
        if info_present:encoded_fields['native_info'] = native_info_value(native_info)
        fact = dict(schema_version=1, source=self.settings.source, strategy_id=self.settings.strategy_id,
                    account_id=self.settings.account_id, run_id=self.run_id,
                    record_id=f'{self.run_id}:{self._sequence}', kind=kind, source_ts_ns=source_ts_ns,
                    observed_ts_ns=observed_ts_ns, observed_mono_ns=observed_mono_ns, fields=encoded_fields)
        from leafquant_monitor.codec import encode_fact
        fact = encode_fact(fact)
        raw = json.dumps(fact, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if len(raw.encode()) + 1 > 96 * 1024:
            self._error = 'RecordExceedsTransportLineLimit'
            raise ValueError('fact exceeds shared complete log-line transport limit')
        try:
            if not self.recorder.append(raw):
                self._report_recorder_rejection()
                raise RuntimeError('monitor record was not accepted')
        except Exception as exc:
            self._error = type(exc).__name__
            raise
        return fact

    def record_event(self, kind, event, *, observed_ts_ns=None, observed_mono_ns=None, extra=None):
        if self.settings.framework_strategy_id is not None and getattr(event, 'strategy_id', None) is None:
            raise ValueError('event identity missing explicit native actor binding')
        bindings = {'strategy_id': self.settings.framework_strategy_id or self.settings.strategy_id,
                    'account_id': self.settings.framework_account_id or self.settings.account_id}
        for field, expected in bindings.items():
            value = getattr(event, field, None)
            if value is not None and str(value) != expected:
                raise ValueError('event identity does not match connector binding')
        fields = public_fields(event, EVENT_FIELDS)
        if self.settings.framework_strategy_id is not None:
            fields['framework_strategy_id'] = fields['strategy_id']
        if self.settings.framework_account_id is not None:
            fields['framework_account_id'] = fields['account_id']
        fields['order_reference'] = self._order_references.get(fields['client_order_id'])
        fields['event_id'] = public_value(getattr(event, 'id', None) or getattr(event, 'event_id', None))
        fields['side'] = fields['order_side'] if fields['order_side'] is not None else fields['side']
        fields['quantity'] = fields['last_qty'] if fields['last_qty'] is not None else fields['quantity']
        fields['price'] = fields['last_px'] if fields['last_px'] is not None else fields['price']
        commission = getattr(event, 'commission', None)
        fields['commission_currency'] = public_value(getattr(commission, 'currency', None))
        fields['realized_pnl_currency'] = public_value(getattr(getattr(event, 'realized_pnl', None), 'currency', None))
        cid = fields['client_order_id']
        fields['order_reference'] = self._order_references.get(cid)
        fields['trace_id'] = self._order_traces.get(cid)
        fields['phase'] = 'strategy_callback'
        if extra:
            # Extensions may attach metadata, never replace validated native
            # facts or the logger-owned causal/revision identity.
            protected = set(EVENT_FIELDS) | {
                'framework_strategy_id', 'framework_account_id', 'order_reference',
                'event_id', 'trace_id', 'phase', 'commission_currency', 'realized_pnl_currency',
                'revision', 'correction_of', 'correction_kind', 'execution_source_ts_ns'}
            for name in protected & extra.keys():
                if public_value(extra[name]) != fields.get(name):
                    raise ValueError('observation extra cannot rewrite validated event identity')
        trade = fields.get('trade_id')
        if kind in ('order_filled', 'order_fill_voided') and trade:
            fields['execution_source_ts_ns'] = getattr(event, 'ts_event', None)
            execution = self._execution_key(fields)
            previous = self._fills.get(execution)
            event_id = fields['event_id']
            if previous:
                fields['execution_source_ts_ns'] = previous['fields'].get('execution_source_ts_ns')
            replay = self._fill_events.get(event_id)
            if event_id and replay is None and len(self._fill_events) >= self.settings.max_event_history:
                self._error = 'EventHistoryCapacityExhausted'
                raise RuntimeError('native event identity budget exhausted; retain source and increase budget before continuing')
            if replay is not None:
                for name in ('quantity', 'price', 'commission', 'commission_currency', 'client_order_id',
                             'venue_order_id', 'instrument_id', 'side', 'currency', 'voided_qty'):
                    if fields.get(name) is not None and replay.get(name) != fields[name]:
                        raise ValueError('native fill event identity changed its exact facts')
                for name in ('revision', 'correction_of', 'correction_kind', 'execution_source_ts_ns'):
                    if name in replay:
                        fields[name] = replay[name]
                if kind == 'order_fill_voided':
                    for name in ('quantity', 'price', 'commission', 'commission_currency', 'client_order_id',
                                 'venue_order_id', 'instrument_id', 'side', 'currency'):
                        if fields.get(name) is None:
                            fields[name] = replay.get(name)
            changed = (previous is not None and any(fields.get(name) != previous['fields'].get(name)
                       for name in ('quantity', 'price', 'commission', 'commission_currency', 'side', 'currency')))
            if replay is None and previous and previous['event_id'] != event_id and (changed or kind == 'order_fill_voided'):
                fields['revision'] = previous['revision'] + 1
                fields['correction_of'] = previous['event_id']
                if kind == 'order_fill_voided':
                    fields['correction_kind'] = 'fill_voided'
                    # The void event references a prior execution, rather than a
                    # second fill. Carry its immutable public execution fields.
                    for name in ('quantity', 'price', 'side', 'currency', 'client_order_id',
                                 'venue_order_id', 'instrument_id'):
                        original = previous['fields'][name]
                        if fields.get(name) is not None and fields[name] != original:
                            raise ValueError('fill void identity conflicts with original execution')
                        fields[name] = original
                else:
                    immutable = ('quantity', 'side', 'currency', 'client_order_id', 'instrument_id')
                    if any(fields.get(name) != previous['fields'].get(name) for name in immutable):
                        raise ValueError('execution correction changed immutable identity')
                    old_venue, new_venue = previous['fields'].get('venue_order_id'), fields.get('venue_order_id')
                    if old_venue is not None and new_venue is not None and old_venue != new_venue:
                        raise ValueError('execution venue order identity changed')
                    fields['correction_kind'] = ('price_restatement' if fields['price'] != previous['fields']['price']
                                                 else 'fee_revision')
            elif replay is None and previous:
                for name in ('revision', 'correction_of', 'correction_kind'):
                    if name in previous['fields']:
                        fields[name] = previous['fields'][name]
            elif replay is None:
                fields['revision'] = 0
                if kind == 'order_fill_voided':
                    # Preserve the public reference; server will reject a void
                    # until its original execution has arrived.
                    fields['correction_kind'] = 'fill_voided'
        if extra:
            fields.update(public_value(extra))
        fact = self.record(kind, fields, source_ts_ns=getattr(event, 'ts_event', None),
                           observed_ts_ns=observed_ts_ns, observed_mono_ns=observed_mono_ns)
        # Advance only after queue admission. A rejected append must be safe to
        # retry against the last accepted correction parent, including voids.
        if kind in ('order_filled', 'order_fill_voided') and trade and event_id:
            self._fill_events[event_id] = dict(fields)
            self._fill_events.move_to_end(event_id)
            if previous is None or fields.get('revision', 0) >= previous['revision']:
                self._fills[execution] = dict(event_id=event_id, revision=fields.get('revision', 0), fields=dict(fields))
                self._fills.move_to_end(execution)
            while len(self._fills) > self.settings.capacity:
                self._fills.popitem(last=False)
        return fact

    def signal(self, fields):
        trace = f'{self.run_id}:trace:{uuid.uuid4().hex}'
        fields=dict(fields,trace_id=trace)
        if self._active_calculation is not None:
            fields['calculation_id']=self._active_calculation['calculation_id']
        self.record('signal',fields)
        if self._active_calculation is not None:
            self._active_calculation['trace_ids'].append(trace)
        return trace

    def risk_check(self, *, trace_id, passed, reason, fields=None):
        return self.record('risk_check', dict(fields or {}, trace_id=trace_id, passed=bool(passed), reason=reason))

    @staticmethod
    def _validate_reference(reference):
        from decimal import Decimal
        required={'instrument_id','side','price','price_currency','quantity_unit','source_ts_ns','source_reference'}
        if not isinstance(reference,dict) or set(reference)!=required:
            raise ValueError('complete explicit order reference required')
        if (reference['side'] not in {'BUY','SELL'} or type(reference['source_ts_ns']) is not int
                or reference['source_ts_ns']<=0 or any(not isinstance(reference[k],str) or not reference[k]
                    for k in required-{'source_ts_ns'})):
            raise ValueError('invalid order reference identity/clock')
        price=reference['price']
        if not re.fullmatch(r'\d+(?:\.\d+)?',price) or not Decimal(price).is_finite() or Decimal(price)<=0:
            raise ValueError('exact positive order reference price required')
        return dict(reference)

    def _reference_id(self, cid, reference, origin_run_id):
        import hashlib
        identity=[self.settings.source,self.settings.strategy_id,self.settings.account_id,origin_run_id,cid,reference]
        return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def bind_order(self, client_order_id, trace_id, *, reference=None):
        cid = str(client_order_id)
        if cid in self._order_traces and self._order_traces[cid] != trace_id:
            raise ValueError('order trace cannot change')
        prepared=None
        if reference is not None:
            reference=self._validate_reference(reference)
            old=self._order_references.get(cid)
            if old is not None:
                if old['origin_run_id']!=self.run_id:
                    raise ValueError('client order ID reuse cannot rebind a recovered reference')
                if {k:old[k] for k in reference}!=reference:
                    raise ValueError('immutable order reference cannot change')
            else:
                if len(self._order_references)>=self.settings.max_event_history:
                    self._error='OrderReferenceCapacityExhausted'
                    raise RuntimeError('exact order reference history budget exhausted; retain source and increase budget')
                reference_id=self._reference_id(cid,reference,self.run_id)
                prepared=dict(reference,reference_id=reference_id,reference_revision=0,origin_run_id=self.run_id)
                # Persistence comes before exposure; logging failure cannot pretend
                # a reference was captured for subsequent executions.
                self.record('order_context',dict(context_type='submit_reference',client_order_id=cid,trace_id=trace_id,
                    strategy_id=self.settings.framework_strategy_id or self.settings.strategy_id,
                    account_id=self.settings.framework_account_id or self.settings.account_id,
                    reference_id=reference_id,reference_revision=0,origin_run_id=self.run_id,reference=reference),
                    source_ts_ns=reference['source_ts_ns'])
        self._order_traces[cid] = trace_id
        if prepared is not None:
            self._order_references[cid]=prepared

    @staticmethod
    def _logical_account_info(info):
        from leafquant_monitor.codec import decoded_info
        return decoded_info(info)

    def _remember_account_event(self, fields, digest=None):
        event_id=fields['native_event_id']
        if digest is None: digest=hashlib.sha256(json.dumps([fields['framework_account_id'],fields['source_ts_ns'],
            fields['source_init_ts_ns'],fields['is_reported'],self._logical_account_info(fields['native_info'])],sort_keys=True,
            separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        old=self._account_events.get(event_id)
        if old is not None and old!=digest:raise ValueError('native account event identity conflict')
        if old is None:
            if len(self._account_events)>=self.settings.max_event_history:raise BufferError('account event identity budget exhausted')
            self._account_events[event_id]=digest
        self._account_last_event=event_id
        return digest

    def _account_event_gap(self, reason, observed_ts_ns):
        if reason!=self._account_gap_reason:
            self.record('account_observation',dict(scope='framework_account_event_gap',account_id=self.settings.account_id,
                framework_account_id=self.settings.framework_account_id or self.settings.account_id,
                reason=reason,coverage='retained_framework_cache_only'),observed_ts_ns=observed_ts_ns)
            self._account_gap_reason=reason

    def _sample_account_events(self, cache, observed_ts_ns):
        from nautilus_trader.model import AccountId
        native_id=AccountId(self.settings.framework_account_id or self.settings.account_id)
        try:
            account=cache.account(native_id)
            if account is None:raise LookupError()
            events=getattr(account,'events',None)
            if not isinstance(events,list):raise LookupError()
        except Exception:
            self._account_event_gap('account_events_unavailable',observed_ts_ns);return
        if not events:
            self._account_event_gap('account_events_empty',observed_ts_ns);return
        if len(events)>self.settings.max_event_history:
            self._account_event_gap('account_events_budget_exceeded',observed_ts_ns);return
        ids=[public_value(getattr(event,'event_id',None)) for event in events]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            self._account_event_gap('account_event_identity_invalid',observed_ts_ns);return
        if self._account_recovery_ambiguous:
            self._account_event_gap('account_event_recovery_order_ambiguous',observed_ts_ns)
            self._account_recovery_ambiguous=False
        elif self._account_last_event is not None and self._account_last_event not in ids:
            self._account_event_gap('previous_account_event_not_retained',observed_ts_ns)
        else:self._account_gap_reason=None
        retained=set(ids)
        self._account_events=OrderedDict((k,v) for k,v in self._account_events.items() if k in retained)
        self._account_info_carriers={k:v for k,v in self._account_info_carriers.items() if k in retained}
        for event,event_id in zip(events,ids):
            account_id=public_value(getattr(event,'account_id',None))
            timestamp=getattr(event,'ts_event',None);initial=getattr(event,'ts_init',None)
            reported=getattr(event,'is_reported',None)
            if (account_id!=(self.settings.framework_account_id or self.settings.account_id)
                or type(timestamp) is not int or timestamp<=0 or type(initial) is not int or initial<=0 or type(reported) is not bool):
                self._account_event_gap('account_event_identity_or_clock_invalid',observed_ts_ns);return
            fields=dict(scope='framework_account_event',account_id=self.settings.account_id,
                framework_account_id=account_id,framework_strategy_id=self.settings.framework_strategy_id or self.settings.strategy_id,
                native_event_id=event_id,source_ts_ns=timestamp,source_init_ts_ns=initial,is_reported=reported,
                native_info=native_info_value(getattr(event,'info',None)),coverage='retained_framework_cache_only')
            carrier=hashlib.sha256(json.dumps([account_id,timestamp,initial,reported,fields['native_info']],sort_keys=True,
                separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            if event_id in self._account_events and self._account_info_carriers.get(event_id)==carrier:
                continue
            digest=hashlib.sha256(json.dumps([account_id,timestamp,initial,reported,self._logical_account_info(fields['native_info'])],sort_keys=True,
                separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            if event_id in self._account_events:
                if self._account_events[event_id]!=digest:raise ValueError('native account event identity conflict')
                self._account_info_carriers[event_id]=carrier
                continue
            if len(self._account_events)>=self.settings.max_event_history:
                self._account_event_gap('account_event_identity_budget_exceeded',observed_ts_ns);return
            try:
                self.record('account_observation',fields,source_ts_ns=timestamp,observed_ts_ns=observed_ts_ns)
            except ValueError:
                self._account_event_gap('account_event_record_rejected',observed_ts_ns);return
            self._remember_account_event(fields,digest)
            self._account_info_carriers[event_id]=carrier

    def sample_account(self, cache, *, observed_ts_ns):
        if self.settings.account_source == 'external':
            return None
        self._sample_account_events(cache, observed_ts_ns)
        fields = framework_snapshot(cache, self.settings.account_id, self.settings.strategy_id,
                                    observed_ts_ns=observed_ts_ns,
                                    framework_strategy_id=self.settings.framework_strategy_id,
                                    framework_account_id=self.settings.framework_account_id)
        return self.record('account_observation', fields, source_ts_ns=fields['source_ts_ns'],
                           observed_ts_ns=observed_ts_ns)

    def record_account_evidence(self, value, *, account_identity, observed_ts_ns):
        """Push existing account-evidence.v1; never query or calculate an account.

        Call from the owning application's evidence ingestion task. The native
        identity must come from its trusted configuration, not from the payload.
        Large originals use the existing collector/source-file v2 path; this
        single-record API rejects the shared 96KiB line limit without truncation.
        Acceptance is not durability, financial qualification or delivery proof.
        """
        import hashlib
        if self.settings.account_source not in ('external', 'framework_and_external'):
            raise ValueError('explicit external account observation mode required')
        native = account_evidence(value, source=self.settings.source,
                                  account_identity=account_identity, observed_ts_ns=observed_ts_ns)
        coverage = native.get('coverage', {})
        identity = [self.settings.source, self.settings.strategy_id, self.settings.account_id,
                    account_identity, native['kind'], native['observed_at_ms'],
                    coverage.get('from_timestamp_ms'), coverage.get('to_timestamp_ms')]
        event_id = 'account-evidence:' + hashlib.sha256(
            json.dumps(identity, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        return self.record('account_observation', dict(scope='external', account_evidence=native,
                           event_id=event_id, revision=0),
                           source_ts_ns=native['observed_at_ms'] * 1_000_000,
                           observed_ts_ns=observed_ts_ns)

    @staticmethod
    def _safe_error_class(value):
        allowed={'RuntimeError','ValueError','TypeError','OSError','JSONDecodeError',
                 'ClientError','TimeoutError','ConnectionError','EndpointConnectionError',
                 'ConnectTimeoutError','ReadTimeoutError','BufferError','PermissionError',
                 'RecordExceedsTransportLineLimit'}
        return value if value in allowed else ('UNCLASSIFIED' if value is not None else None)

    @staticmethod
    def _recorder_code(error,closed):
        fixed={'recorder queue capacity exhausted':'QUEUE_CAPACITY_EXHAUSTED',
               'recorder worker disconnected':'WORKER_DISCONNECTED',
               'native recorder requires native payload':'RECORDER_MODE_MISMATCH',
               'invalid object JSON or record exceeds segment_bytes':'INVALID_RECORD_OR_SIZE',
               'explicit native recorder and valid producer clock required':'INVALID_NATIVE_CLOCK_OR_MODE',
               'recorder closed':'CLOSED'}
        if error in fixed:return fixed[error]
        if isinstance(error,str) and error.startswith('storage:'):
            if error=='storage: recorder disk limit exceeded':return 'RECORDER_DISK_BUDGET_EXHAUSTED'
            if error=='storage: storage already failed':return 'STORAGE_ALREADY_FAILED_ORIGINAL_UNAVAILABLE'
            if 'native formatter failed' in error:return 'NATIVE_FORMATTER_FAILED'
            if error=='storage: invalid native producer clock':return 'STORAGE_INVALID_NATIVE_CLOCK'
            errno=re.search(r'\(os error ([0-9]+)\)',error)
            if errno:
                code={2:'STORAGE_ENOENT',5:'STORAGE_EIO',13:'STORAGE_EACCES',16:'STORAGE_EBUSY',17:'STORAGE_EEXIST',24:'STORAGE_EMFILE',28:'STORAGE_ENOSPC',30:'STORAGE_EROFS',122:'STORAGE_EDQUOT'}.get(int(errno.group(1)))
                if code:return code
            return 'STORAGE_ERROR_UNCLASSIFIED'
        if error is not None:return 'RECORDER_ERROR_UNCLASSIFIED'
        return 'CLOSED' if closed is True else 'OK'

    def health_summary(self):
        """Same-owner read-only diagnostics. Never return native text, identities or paths."""
        result=dict(schema_version=1,checked_at_ns=time.time_ns(),queue_capacity=self.settings.capacity,
                    queue_outstanding=None,queue_outstanding_available=False,
                    logger_error_class=self._safe_error_class(self._error))
        try:
            state=json.loads(self.recorder.status())
            accepted=state.get('accepted');durable=state.get('durable');closed=state.get('closed')
            accepted=accepted if type(accepted) is int and accepted>=0 else None
            durable=durable if type(durable) is int and durable>=0 else None
            closed=closed if type(closed) is bool else None
            result.update(recorder_code=self._recorder_code(state.get('error'),closed),recorder_closed=closed,
                          accepted=accepted,durable=durable,
                          accepted_minus_durable=accepted-durable if accepted is not None and durable is not None and accepted>=durable else None)
        except Exception as exc:
            result.update(recorder_code='HEALTH_READ_FAILED',recorder_closed=None,accepted=None,durable=None,
                          accepted_minus_durable=None,status_read_error_class=self._safe_error_class(type(exc).__name__))
        if self.sender is not None:
            try:
                sender=self.sender.health();sent=sender.get('sent');checked=sender.get('checked_at');running=sender.get('running')
                result['sender']=dict(running=running if type(running) is bool else None,
                    error_class=self._safe_error_class(sender.get('error')),
                    sent=sent if type(sent) is int and sent>=0 else None,
                    checked_at=checked if type(checked) in (int,float) and math.isfinite(checked) else None)
            except Exception as exc:result['sender']=dict(error_class='HEALTH_READ_FAILED',status_read_error_class=self._safe_error_class(type(exc).__name__))
        return result

    def _report_recorder_rejection(self):
        # Existing bounded application/Docker logs are the persistence destination.
        # Diagnostic failures cannot replace the original append rejection or drive trading.
        try:
            summary=self.health_summary()
            if summary['recorder_code']=='OK':summary['recorder_code']='APPEND_REJECTED_WITHOUT_NATIVE_ERROR'
            logging.getLogger(__name__).error('monitor_recorder_rejected %s',json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False))
        except Exception:
            pass

    def health(self):
        result = json.loads(self.recorder.status())
        if self._error is not None:
            result['error'] = self._error
        if self.sender is not None:
            result['sender'] = self.sender.health()
        return result

    def flush(self):
        result = json.loads(self.recorder.flush())
        if result.get('error'):
            self._error = result['error']
            raise RuntimeError('native monitor persistence failed')
        return result

    def close(self):
        # Seal local raw data. Network sending stops asynchronously; no network
        # wait is added to the framework lifecycle callback.
        result = json.loads(self.recorder.close())
        if self.sender is not None:
            self.sender.stop.set()
        return result

    def finish_delivery(self, timeout=30):
        """Application shutdown may join sender outside strategy callbacks.

        This is not an archive receipt and never permits deleting raw segments.
        """
        result = self.close()
        if self.sender is not None:
            self.sender.close(timeout)
        return result
