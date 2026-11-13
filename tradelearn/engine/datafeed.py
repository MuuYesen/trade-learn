from __future__ import annotations

import pandas as pd

from tradelearn.backtest.feed import RuntimeDataFeed
from tradelearn.backtest._bt_datetime import BTDateLine
from tradelearn.engine.base import LineRoot, LineSeries


class _DateLine(BTDateLine, LineSeries):
    """BT view on unchanged Unix storage."""


class DataFeed(RuntimeDataFeed, LineRoot):
    """Backtrader-style DataFeed with LineSeries magic."""
    _reset_line_aliases = True
    _bt_line_views = True
    lines = ("close", "low", "high", "open", "volume", "openinterest", "datetime")

    @property
    def _indicator_line(self):
        return self.close

    def __getitem__(self, ago):
        return self.close[ago]

    def __len__(self):
        visible=vars(self).get('_replay_visible')
        return visible._cursor+1 if visible is not None else super().__len__()

    def _advance(self, cursor):
        super()._advance(cursor)
        visible=vars(self).get('_replay_visible')
        if visible is None:return
        if cursor<0:
            visible._cursor=-1
            return
        logical=int(self._replay_lengths[cursor])-1
        visible._cursor=logical
        if logical<0:
            visible._empty_current={name:values[cursor] for name,values in self._replay_values.items()}
            visible._empty_epoch_ns=int(self._replay_label_ns[cursor])
            return
        visible._empty_current=None;visible._empty_epoch_ns=None
        for name,array in self._replay_lines.items():
            array[logical]=self._replay_values[name][cursor]
        labels=vars(self).get('_replay_label_ns')
        stamp=(int(labels[cursor]) if labels is not None else int(self._clock_timestamps[cursor]) if self._clock_unit=='ns'
               else int(self._datetime[cursor])*1_000_000_000)
        self._replay_epoch_ns[logical]=stamp

    def __init__(self, data: pd.DataFrame, name: str | None = None, **kwargs) -> None:
        preserve_order=kwargs.pop('_preserve_row_order',False)
        RuntimeDataFeed.__init__(self, data, name=name,assume_normalized=preserve_order)
        if preserve_order and not self._frame.index.is_monotonic_increasing:
            import numpy as np
            from tradelearn.backtest.clock import native_timestamps
            self._delivery_unit='ns'
            self._delivery_datetime=np.maximum.accumulate(native_timestamps(self,'ns'))
        self.datetime = _DateLine(self._datetime, is_datetime=True,
                                  buffer=self._bar_buffer, buffer_name="datetime")
        self.datetime._cursor_source = self
        if self._clock_unit == 'ns':
            self.datetime._epoch_ns_values = self._clock_timestamps
        self._lines_list[0] = self.datetime
        LineRoot._base_init(self, **kwargs)
        from ._timezone import resolve_timezone
        self._tz=resolve_timezone(kwargs.get('tz',None))
        self.datetime._tz=self._tz
        from ._calendar import resolve_calendar
        self._calendar=resolve_calendar(kwargs.get('calendar',None))
        self.lines.datetime = self.datetime
        self.lines.open = self.open
        self.lines.high = self.high
        self.lines.low = self.low
        self.lines.close = self.close
        self.lines.volume = self.volume

        self.datetime = self.lines.datetime
        self.open = self.lines.open
        self.high = self.lines.high
        self.low = self.lines.low
        self.close = self.lines.close
        self.volume = self.lines.volume

    def date2num(self,value):
        from tradelearn.backtest.clock import date2num
        from ._timezone import localize
        return date2num(localize(value,self._tz))

    def num2date(self,value=None,tz=None,naive=True):
        from tradelearn.backtest.clock import num2date
        return num2date(self.datetime[0] if value is None else value,tz=tz or self._tz,naive=naive)

    def _getnexteos(self):
        from datetime import datetime,timedelta,time
        from tradelearn.backtest.clock import date2num,num2date
        source=vars(self).get('_eos_source')
        if source is not None:return source._getnexteos()
        if not len(self):return datetime.min,0.
        return self._session_end_at(self.datetime[0])

    def _session_end_at(self,value):
        from datetime import datetime,timedelta,time
        from tradelearn.backtest.clock import date2num,num2date
        now=num2date(value)
        if self._calendar is not None:
            from ._calendar import CalendarTimezone
            zone=CalendarTimezone(self._tz) if self._tz is not None else None
            _,end=self._calendar.schedule(now,zone)
            return end,date2num(end)
        end=datetime.combine(now.date(),getattr(self.p,'sessionend',None) or time(23,59,59,999990))
        end=num2date(self.date2num(end))
        if end<now:end+=timedelta(days=1)
        return end,date2num(end)
