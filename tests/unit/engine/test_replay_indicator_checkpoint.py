"""Replay replaces one logical observation without advancing native state twice."""
import math
import pytest
from tradelearn.indicators.base import StreamBinding


def indicators():
    from tradelearn.indicators import talib,tv,tdx
    return [('talib',talib.EMA.configured(timeperiod=2)),('tv',tv.ema.configured(length=2)),('tdx',tdx.EMA.configured(N=2))]


@pytest.mark.parametrize('name,indicator',indicators())
@pytest.mark.parametrize('prefix',[[],[1.],[1.,2.,4.,8.]])
def test_replay_checkpoint_restores_finite_backend_without_replaying_committed_prefix(name,indicator,prefix):
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


@pytest.mark.parametrize('name,indicator',indicators())
def test_checkpoint_rejects_foreign_binding_and_expired_generation(name,indicator):
    a,b=StreamBinding(indicator,()),StreamBinding(indicator,())
    try:
        a.update(1.);checkpoint=a._checkpoint()
        with pytest.raises(ValueError,match='foreign or expired'):b._restore_checkpoint(checkpoint)
        a.reset()
        with pytest.raises(ValueError,match='foreign or expired'):a._restore_checkpoint(checkpoint)
    finally:a.reset();b.reset()


@pytest.mark.parametrize('name,indicator',indicators())
def test_repeated_revisions_advance_only_current_bar_baseline_over_long_prefix(name,indicator):
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
