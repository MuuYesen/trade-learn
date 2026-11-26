"""Cold event materialization preserves replay evidence and eviction order."""
from leafquant_monitor.logger import _PackedHistory


def test_revisions_and_nested_references_round_trip_independently():
    history = _PackedHistory()
    original = {'source_ts_ns': 1795576835123456789, 'fee': None,
                'reference': {'price': '103.10000'}, 'revision': 0}
    history['fill'] = original
    original['reference']['price'] = '107'
    history['correction'] = dict(original, revision=1)
    assert history['fill']['reference'] == {'price': '103.10000'}
    assert history['fill']['source_ts_ns'] == 1795576835123456789
    assert history['fill']['fee'] is None
    assert history['correction']['revision'] == 1
    history.move_to_end('fill')
    assert history.popitem(last=False) == ('correction', dict(original, revision=1))
    assert list(history) == ['fill'] and history.get('missing') is None
