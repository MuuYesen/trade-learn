"""Typed native factory controls carried by existing BT broker kwargs."""
from decimal import Decimal
import math
from numbers import Real

CONTROL_NAMES=frozenset({'reduce_only','position_id','post_only','display_qty','trigger_type',
                         'emulation_trigger','exec_algorithm_id','exec_algorithm_params'})
_CONDITIONAL={'stop','stop_limit','stop_trail','stop_trail_limit'}
_LIMIT={'limit','stop_limit','stop_trail_limit'}


def _trigger(value,name):
    from nautilus_trader.model import TriggerType
    if isinstance(value,TriggerType):
        return value
    candidate=getattr(TriggerType,value.upper(),None) if isinstance(value,str) else None
    if not isinstance(candidate,TriggerType):
        raise ValueError(f'{name} requires a fixed native TriggerType name')
    return candidate


def _params(value):
    if value is not None and (not isinstance(value,dict) or
            any(not isinstance(k,str) or not isinstance(v,str) for k,v in value.items())):
        raise TypeError('exec_algorithm_params must be a mapping of strings to strings')
    return None if value is None else dict(value)


def factory_control_kwargs(request,controls,instrument):
    """Consume only declared controls; keep position routing outside factory fields."""
    from nautilus_trader.model import ExecAlgorithmId,TriggerType
    result={}
    post_only=controls.pop('post_only',False)
    reduce_only=controls.pop('reduce_only',False)
    if type(post_only) is not bool:
        raise TypeError('post_only must be boolean')
    if type(reduce_only) is not bool:
        raise TypeError('reduce_only must be boolean')
    if post_only and request.order_type not in _LIMIT:
        raise ValueError('post-only requires a limit execution order')
    result['reduce_only']=reduce_only
    if request.order_type in _LIMIT:result['post_only']=post_only
    display=controls.pop('display_qty',None)
    if display is not None:
        if isinstance(display,bool) or not isinstance(display,(Real,Decimal)):
            raise TypeError('display_qty must be a finite numeric quantity')
        if (not math.isfinite(float(display)) or display<=0 or
                Decimal(str(display))>Decimal(str(request.qty))):
            raise ValueError('positive display_qty cannot exceed total quantity')
        if request.order_type=='market':
            raise ValueError('display_qty requires a native non-Market factory')
        quantity=instrument.make_qty(display)
        if Decimal(str(quantity))!=Decimal(str(display)):
            raise ValueError('display quantity would be rounded by venue precision')
        result['display_qty']=quantity
    trigger=controls.pop('trigger_type',None)
    if trigger is not None:
        if request.order_type not in _CONDITIONAL:
            raise ValueError('trigger_type requires a conditional order')
        result['trigger_type']=_trigger(trigger,'trigger_type')
    emulation=controls.pop('emulation_trigger',None)
    if emulation is not None:
        emulation=_trigger(emulation,'emulation_trigger')
        if request.order_type=='market':
            if emulation!=TriggerType.NO_TRIGGER:
                raise ValueError('emulation_trigger requires a native conditional/limit factory')
        else:result['emulation_trigger']=emulation
    algorithm=controls.pop('exec_algorithm_id',None)
    params=_params(controls.pop('exec_algorithm_params',None))
    if algorithm is None:
        if params is not None:
            raise ValueError('exec_algorithm_params require exec_algorithm_id')
    else:
        if isinstance(algorithm,str):algorithm=ExecAlgorithmId(algorithm)
        if not isinstance(algorithm,ExecAlgorithmId):
            raise TypeError('exec_algorithm_id must be an algorithm ID string')
        result.update(exec_algorithm_id=algorithm,exec_algorithm_params=params)
    return result


def _field(native,name):
    marker=object()
    value=getattr(native,name,marker)
    if value is marker:
        value=getattr(getattr(native,'init_event',None),name,None)
    return value


def control_facts(native):
    """Return JSON-safe immutable controls, not mutable execution status."""
    native=emulation_admission(native) or native
    result={}
    for name in ('display_qty','trigger_type','emulation_trigger','exec_algorithm_id'):
        value=_field(native,name)
        result[name]=None if value is None else str(value)
    if result['emulation_trigger']=='NO_TRIGGER':result['emulation_trigger']=None
    params=_field(native,'exec_algorithm_params')
    result['exec_algorithm_params']=None if params is None else dict(params)
    return result


def native_control_proof(native,controls):
    """Preserve the legacy minimal proof, adding only non-default new controls."""
    admission=emulation_admission(native)
    proof=dict(reduce_only=bool(admission.reduce_only if admission is not None else native.is_reduce_only),
               position_id=str(controls['position_id']) if controls.get('position_id') is not None else None,
               post_only=bool(admission.post_only if admission is not None else getattr(native,'is_post_only',False)))
    for name,value in control_facts(native).items():
        if value is not None and not (name=='trigger_type' and value=='DEFAULT'):
            proof[name]=value
    return proof


def validate_extra_control_proof(proof,request,native):
    """Refuse missing/mutated admission fields while reading legacy defaults."""
    actual=control_facts(native)
    defaults=dict(display_qty=None,trigger_type='DEFAULT' if request.order_type in _CONDITIONAL else None,
                  emulation_trigger=None,exec_algorithm_id=None,exec_algorithm_params=None)
    for name in defaults:
        expected=proof.get(name,defaults[name])
        observed=actual[name]
        if name=='display_qty' and expected is not None:
            if not isinstance(expected,str):raise ValueError('invalid durable native display quantity proof')
            try:
                if observed is None or Decimal(expected)!=Decimal(observed):
                    raise ValueError('native recovery display_qty differs from admitted route')
            except ArithmeticError as error:
                raise ValueError('invalid native recovery display quantity') from error
            continue
        if name in ('trigger_type','emulation_trigger') and expected is not None:
            try:expected=str(_trigger(expected,name))
            except (ValueError,TypeError) as error:raise ValueError('invalid native recovery trigger proof') from error
            if name=='emulation_trigger' and expected=='NO_TRIGGER':expected=None
        if name=='exec_algorithm_params':
            try:expected=_params(expected)
            except (ValueError,TypeError) as error:raise ValueError('invalid native recovery algorithm params') from error
        if expected!=observed:
            raise ValueError(f'native recovery {name} differs from admitted route')


def emulation_admission(native):
    """Prove official emulator transformation before using original admission.

    rc7 preserves its original initialization in init_event while replacing a
    released conditional/limit object with a Market/Limit execution object.
    A changed type alone never authorizes recovery against stale admission.
    """
    initial=getattr(native,'init_event',None)
    trigger=getattr(initial,'emulation_trigger',None)
    if trigger is None or str(trigger)=='NO_TRIGGER':return None
    from nautilus_trader.model import OrderInitialized,OrderEmulated,OrderReleased
    events=list(native.events())
    released=[i for i,e in enumerate(events) if isinstance(e,OrderReleased)]
    if not released:return None
    if len(events)>4096 or len(released)!=1 or not isinstance(initial,OrderInitialized):
        raise ValueError('invalid native emulation release history')
    index=released[0]
    prior=events[:index]
    initialized=[e for e in prior if isinstance(e,OrderInitialized)]
    emulated=[i for i,e in enumerate(prior) if isinstance(e,OrderEmulated)]
    if (len(initialized)!=2 or len(emulated)!=1 or not events
            or initialized[0].to_dict()!=initial.to_dict()
            or not emulated[0] < next(i for i,e in enumerate(prior) if e is initialized[1])):
        raise ValueError('native emulation release lacks ordered initialization evidence')
    transformed=initialized[1]
    permitted={'LIMIT':'MARKET','STOP_MARKET':'MARKET','STOP_LIMIT':'LIMIT',
               'TRAILING_STOP_MARKET':'MARKET','TRAILING_STOP_LIMIT':'LIMIT'}
    if (permitted.get(initial.order_type.name)!=transformed.order_type.name
            or native.order_type!=transformed.order_type):
        raise ValueError('invalid native emulation release type conversion')
    for event in (initialized[0],*[events[i] for i in emulated],transformed,events[index]):
        if any(getattr(event,name)!=getattr(initial,name) for name in
               ('trader_id','strategy_id','instrument_id','client_order_id')):
            raise ValueError('native emulation release changes admitted scope')
    if any(getattr(native,name)!=getattr(initial,name) for name in
           ('trader_id','strategy_id','instrument_id','client_order_id')):
        raise ValueError('native emulation execution changes admitted scope')
    if (initial.order_side!=transformed.order_side or native.side!=initial.order_side
            or initial.quantity!=transformed.quantity
            or initial.time_in_force!=transformed.time_in_force
            or native.time_in_force!=initial.time_in_force
            or initial.reduce_only!=transformed.reduce_only
            or native.is_reduce_only!=transformed.reduce_only
            or getattr(native,'is_post_only',False)!=transformed.post_only
            or (transformed.emulation_trigger is not None and str(transformed.emulation_trigger)!='NO_TRIGGER')
            or initial.exec_algorithm_id!=transformed.exec_algorithm_id
            or initial.exec_algorithm_params!=transformed.exec_algorithm_params
            or initial.exec_spawn_id!=transformed.exec_spawn_id):
        raise ValueError('native emulation release changes admitted routing or quantity')
    for name in ('display_qty','trigger_type','emulation_trigger','exec_algorithm_id',
                 'exec_algorithm_params','exec_spawn_id'):
        if getattr(native,name,None)!=getattr(transformed,name,None):
            raise ValueError('native emulation execution changes released factory controls')
    if native.quantity!=transformed.quantity and initial.exec_algorithm_id is None:
        raise ValueError('native emulation released quantity differs from execution')
    if native.order_type.name=='LIMIT' and native.price!=transformed.price:
        raise ValueError('native emulation released price differs from execution')
    if not (initial.ts_event<=events[emulated[0]].ts_event<=transformed.ts_event<=events[index].ts_event):
        raise ValueError('native emulation release has inconsistent event times')
    return initial
