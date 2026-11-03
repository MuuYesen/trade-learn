"""Read public framework cache on its owning thread; never query a broker."""
from decimal import Decimal
import json
import math


def public_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('non-finite public numeric value')
        return format(value, 'f')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('non-finite public numeric value')
        # Do not introduce additional binary-float conversions.
        return str(value)
    if isinstance(value, dict):
        return {str(k): public_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(v) for v in value]
    if hasattr(value, 'as_decimal'):
        return public_value(value.as_decimal())
    if hasattr(value, 'code'):
        return str(value.code)
    if hasattr(value, 'name'):
        return str(value.name)
    return str(value)


def native_info_value(value):
    """Copy Params JSON without converting native floats or object keys."""
    def check(item):
        if isinstance(item,dict):
            if any(not isinstance(k,str) for k in item):raise ValueError('native info JSON object key must be a string')
            for child in item.values():check(child)
        elif isinstance(item,list):
            for child in item:check(child)
        elif item is not None and type(item) not in (str,int,float,bool):
            raise ValueError('native info must contain JSON values')
    check(value)
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))

def public_fields(value, names):
    return {name: public_value(getattr(value, name, None)) for name in names}


def account_evidence(value, *, source, account_identity, observed_ts_ns):
    """Validate/copy existing native evidence, without valuation or broker I/O.

    Partial coverage is retained as partial evidence. Financial qualification and
    same-observation conflicts belong to the existing manager source journal.
    """
    if (not isinstance(account_identity, str) or not account_identity
            or account_identity.strip() != account_identity
            or any(ord(c) < 32 or ord(c) == 127 for c in account_identity)
            or type(observed_ts_ns) is not int or observed_ts_ns <= 0):
        raise ValueError('explicit native account binding and observation clock required')
    # Native envelopes are JSON, not arbitrary objects coerced to repr strings.
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

    def validate(native):
        if (not isinstance(native, dict)
                or native.get('schema_version') != 'leafquant.account-evidence.v1'
                or native.get('source') != source
                or native.get('account_identity') != account_identity
                or native.get('kind') not in ('snapshot', 'history')
                or type(native.get('observed_at_ms')) is not int
                or native['observed_at_ms'] <= 0
                or native['observed_at_ms'] * 1_000_000 > observed_ts_ns + 1_000_000_000
                or not isinstance(native.get('payload'), dict)):
            raise ValueError('native account evidence identity/schema/clock mismatch')
        if native['kind'] == 'history':
            coverage = native.get('coverage')
            if (not isinstance(coverage, dict)
                    or any(type(coverage.get(k)) is not int or coverage[k] < 0
                           for k in ('from_timestamp_ms', 'to_timestamp_ms'))
                    or not coverage['from_timestamp_ms'] <= coverage['to_timestamp_ms'] <= native['observed_at_ms']
                    or any(k in coverage and type(coverage[k]) is not bool
                           for k in ('complete', 'inclusive'))):
                raise ValueError('native history coverage mismatch')
            nested = native['payload'].get('snapshot')
            if nested is not None:
                if not isinstance(nested, dict) or nested.get('kind') != 'snapshot':
                    raise ValueError('native history requires snapshot evidence')
                validate(nested)
                if nested['observed_at_ms'] > coverage['to_timestamp_ms']:
                    raise ValueError('native snapshot exceeds history coverage')
    validate(value)
    return value


def framework_snapshot(cache, account_id, strategy_id, *, observed_ts_ns, framework_strategy_id=None, framework_account_id=None):
    actor_id = framework_strategy_id or strategy_id
    native_account_id = framework_account_id or account_id
    actor_fields = {'framework_strategy_id': actor_id} if framework_strategy_id is not None else {}
    if framework_account_id is not None:
        actor_fields['framework_account_id'] = framework_account_id
    # Official v2 cache lookup accepts typed identifiers, not account strings.
    from nautilus_trader.model import AccountId, StrategyId
    native_account, native_strategy = AccountId(native_account_id), StrategyId(actor_id)
    try:
        account = cache.account(native_account)
        account_missing = 'not_in_framework_cache'
    except Exception:
        account = None
        account_missing = 'cache_query_failed'
    collection_missing = {}
    def collection(method_name, label):
        method = getattr(cache, method_name, None)
        if not callable(method):
            collection_missing[label] = 'cache_query_unavailable'
            return None
        try:
            result = method(strategy_id=native_strategy, account_id=native_account)
            if result is None:
                collection_missing[label] = 'cache_query_unavailable'
                return None
            return list(result)
        except Exception:
            collection_missing[label] = 'cache_query_failed'
            return None
    positions = collection('positions_open', 'positions')
    orders = collection('orders_open', 'open_orders')
    inflight = collection('orders_inflight', 'open_orders')
    position_rows = None if positions is None else [dict(public_fields(p, ('id', 'instrument_id', 'account_id', 'strategy_id', 'side',
        'quantity', 'avg_px_open', 'realized_pnl', 'quote_currency', 'ts_last')),
        realized_pnl_currency=public_value(getattr(getattr(p, 'realized_pnl', None), 'currency', None)))
        for p in positions]
    by_id = None if orders is None or inflight is None else {str(order.client_order_id): order for order in [*orders, *inflight]}
    order_rows = None if by_id is None else [public_fields(o, ('client_order_id', 'venue_order_id', 'instrument_id', 'account_id',
        'strategy_id', 'side', 'quantity', 'price', 'filled_qty', 'leaves_qty', 'status', 'ts_last'))
        for o in by_id.values()]
    if account is None:
        return dict(**actor_fields, scope='framework_cache', account_id=account_id, observed_ts_ns=observed_ts_ns,
                    source_ts_ns=None, balances=[], margins=[], margins_status='unknown',
                    margins_scope='reported_account_state', positions=position_rows, open_orders=order_rows,
                    positions_scope='strategy_cache', open_orders_scope='strategy_cache', equity=None, gross_exposure=None,
                    missing={'account': account_missing, 'equity': 'not_exposed_as_standard_field',
                             'gross_exposure': 'valuation_not_provided', 'account_history': 'cache_is_not_complete_history',
                             'margins': 'not_reported_in_account_state', **collection_missing}, complete=False)
    event = account.last_event
    balances = [public_fields(balance, ('currency', 'total', 'free', 'locked'))
                for balance in account.balances().values()]
    # MarginBalance is an explicit public AccountState observation; wallet
    # locked/free balances do not establish initial or maintenance margin.
    margins = [public_fields(row, ('currency', 'instrument_id', 'initial', 'maintenance'))
               for row in (getattr(event, 'margins', None) or [])]
    account_type = public_value(account.account_type)
    margins_status = 'reported' if margins else 'not_applicable' if account_type == 'CASH' else 'unknown'
    margin_missing = {'margins': 'not_reported_in_account_state'} if margins_status == 'unknown' else {}
    return dict(**actor_fields, scope='framework_cache', account_id=account_id, account_type=public_value(account.account_type),
                observed_ts_ns=observed_ts_ns, source_ts_ns=getattr(event, 'ts_event', None),
                source_init_ts_ns=getattr(event, 'ts_init', None), is_reported=getattr(event, 'is_reported', None),
                native_info=native_info_value(getattr(event, 'info', None)), balances=balances,
                margins=margins, margins_status=margins_status, margins_scope='reported_account_state',
                positions=position_rows, open_orders=order_rows,
                positions_scope='strategy_cache', open_orders_scope='strategy_cache',
                equity=None, gross_exposure=None, complete=False,
                missing={'equity': 'not_exposed_as_standard_field', 'gross_exposure': 'valuation_not_provided',
                         'account_history': 'cache_is_not_complete_history', **margin_missing, **collection_missing})
