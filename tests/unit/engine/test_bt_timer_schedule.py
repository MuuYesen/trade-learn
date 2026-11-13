"""Finite P04 bar-driven timer checks compare fixed BT scheduling, without threads."""
from datetime import datetime,time,timedelta
from types import SimpleNamespace
import backtrader as bt
import pytest
from tradelearn import engine as tl


def schedule(api,dates,**options):
    timer=api.Timer(when=time(9),**options)
    timer.start(SimpleNamespace(p=SimpleNamespace(sessionstart=time.min,sessionend=time(23,59,59,999990))))
    rows=[]
    for value in dates:
        stamp=datetime.fromisoformat(value)
        if timer.check(api.date2num(stamp)):rows.append(timer.lastwhen.isoformat())
    return rows


@pytest.mark.parametrize('options',[
 {},{'offset':timedelta(minutes=15)}, {'repeat':timedelta(minutes=5)},
 {'weekdays':[1,3,5]}, {'weekdays':[1,3,5],'weekcarry':True},
 {'monthdays':[1,15]}, {'monthdays':[1,15],'monthcarry':False},
 {'allow':lambda d:d.day%2==0},
])
def test_timer_schedule_matches_fixed_bt_on_sparse_calendar_and_repeats(options):
    dates=['2026-01-01T08:59','2026-01-01T09:00','2026-01-01T09:06','2026-01-01T10:10','2026-01-02T09:20','2026-01-05T09:20','2026-01-16T09:20','2026-02-03T09:20']
    assert schedule(tl,dates,**options)==schedule(bt,dates,**options)


@pytest.mark.parametrize('zone',[-5,9,0])
@pytest.mark.parametrize('repeat',[timedelta(),timedelta(minutes=5)])
def test_existing_timezone_timer_and_repeat_keep_fixed_bt_scheduled_when(zone,repeat):
    from datetime import tzinfo
    class FixedZone(tzinfo):
        def utcoffset(self,dt):return timedelta(hours=zone)
        def dst(self,dt):return timedelta()
        def tzname(self,dt):return f'UTC{zone:+}'
        def localize(self,value):return value.replace(tzinfo=self)
    dates=['2026-01-01T00:00','2026-01-01T00:06','2026-01-01T09:00','2026-01-01T09:06','2026-01-01T14:00','2026-01-01T14:06','2026-01-02T14:20']
    assert schedule(tl,dates,tzdata=FixedZone(),repeat=repeat)==schedule(bt,dates,tzdata=FixedZone(),repeat=repeat)
