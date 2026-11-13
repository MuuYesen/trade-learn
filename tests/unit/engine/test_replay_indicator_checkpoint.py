"""Replay replaces one logical observation without advancing native state twice."""
import math
import pytest
from tradelearn.indicators.base import StreamBinding


@pytest.fixture(params=['talib', 'tv', 'tdx'])
def indicator(request):
    # The pinned PyneCore distribution requires CPython >= 3.11.
    # Keep the other backends running on CP310; supported profiles must import.
    import sys
    import importlib
    if request.param == 'tv' and sys.version_info < (3, 11):
        pytest.skip('pynesys-pynecore==6.13.3 requires Python >= 3.11')
    backend = importlib.import_module('tradelearn.indicators.' + request.param)
    if request.param == 'talib':
        return backend.EMA.configured(timeperiod=2)
    if request.param == 'tv':
        return backend.ema.configured(length=2)
    return backend.EMA.configured(N=2)


@pytest.mark.parametrize('prefix',[[],[1.],[1.,2.,4.,8.]])
def test_replay_checkpoint_restores_finite_backend_without_replaying_committed_prefix(indicator,prefix):
    binding=StreamBinding(indicator,())
    try:
        for value in prefix:binding.update(value)
        checkpoint=binding._checkpoint()
        for revision in (10.,20.,30.):
            binding._restore_checkpoint(checkpoint)
            actual=binding.update(revision)
            reference=StreamBinding(indicator,())
            try:
                for value in [*prefix,revision]:expected=reference.update(value)
            finally:reference.reset()
            if math.isnan(expected):assert math.isnan(actual)
            else:assert actual==pytest.approx(expected)
    finally:binding.reset()


def test_checkpoint_rejects_foreign_binding_and_expired_generation(indicator):
    a,b=StreamBinding(indicator,()),StreamBinding(indicator,())
    try:
        a.update(1.);checkpoint=a._checkpoint()
        with pytest.raises(ValueError,match='foreign or expired'):b._restore_checkpoint(checkpoint)
        a.reset()
        with pytest.raises(ValueError,match='foreign or expired'):a._restore_checkpoint(checkpoint)
    finally:a.reset();b.reset()


def test_repeated_revisions_advance_only_current_bar_baseline_over_long_prefix(indicator):
    actual,reference=StreamBinding(indicator,()),StreamBinding(indicator,())
    try:
        for index in range(400):
            base=10.+index*.125
            checkpoint=actual._checkpoint()
            for candidate in (base+2.,base-1.,base):
                actual._restore_checkpoint(checkpoint);value=actual.update(candidate)
            expected=reference.update(base)
            if math.isnan(expected):assert math.isnan(value)
            else:assert value==pytest.approx(expected,rel=1e-12,abs=1e-12)
        # The public visible state is finite; no committed prefix is buffered.
        assert all(len(v)<100 for v in actual.state.buffers.values())
    finally:actual.reset();reference.reset()
