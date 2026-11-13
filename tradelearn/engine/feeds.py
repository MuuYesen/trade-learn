"""Pandas feeds using the frozen Backtrader column parameter convention."""
from __future__ import annotations
import datetime as dt
import pandas as pd
from .datafeed import DataFeed
from .base import TimeFrame
from tradelearn.backtest.lines import LineSeries, line_names


class PandasData(DataFeed):
    params = (("dataname", None), ("name", ""), ("nocase", True),
              ("datetime", None), ("open", -1), ("high", -1), ("low", -1),
              ("close", -1), ("volume", -1), ("openinterest", -1),
              ("fromdate", None), ("todate", None), ("timeframe", TimeFrame.Days),
              ("compression", 1), ("tz", None), ("tzinput", None),
              ("sessionstart", None), ("sessionend", None), ("calendar", None),
              ("qcheck", 0.0), ("filters", []))
    lines = ("close", "low", "high", "open", "volume", "openinterest", "datetime")

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("PandasData accepts declared keyword parameters only")
        params = self.p.asdict()
        for key in ("qcheck", "filters"):
            if params[key]:
                raise NotImplementedError(f"PandasData {key} is not implemented")
        from ._calendar import resolve_calendar
        params['calendar']=resolve_calendar(params['calendar'])
        from ._timezone import resolve_timezone,localize
        output_zone=resolve_timezone(params['tz'])
        input_zone=resolve_timezone(params['tzinput'])
        for field,default in [('sessionstart',dt.time.min),('sessionend',dt.time(23,59,59,999990))]:
            value=params[field]
            if isinstance(value,dt.datetime):value=value.time()
            if value is None:value=default
            if not isinstance(value,dt.time):raise TypeError(f'{field} must be time or datetime')
            params[field]=value
        frame = params["dataname"]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("PandasData dataname must be a DataFrame")
        frame = frame.copy()
        columns = list(frame.columns)

        def column(selector, name):
            if selector is None:
                return None
            if isinstance(selector, int):
                if selector >= 0:
                    return columns[selector]
                matches = [c for c in columns if (str(c).lower() if params["nocase"] else c) == name]
                return matches[0] if matches else None
            matches=[c for c in columns if (str(c).lower()==str(selector).lower() if params["nocase"] else c==selector)]
            if not matches:
                raise ValueError(f"missing PandasData column {selector!r}")
            return matches[0]

        date_column = column(params["datetime"], "datetime")
        index = pd.DatetimeIndex(frame[date_column] if date_column is not None else frame.index)
        if input_zone is not None:
            values=[]
            for value in index:
                naive=value.tz_convert('UTC').tz_localize(None) if value.tzinfo is not None else value
                interpreted=localize(naive.to_pydatetime(warn=False),input_zone)
                offset=interpreted.utcoffset()
                values.append(value.value-int(offset.total_seconds()*1_000_000_000))
            index=pd.DatetimeIndex(pd.to_datetime(values,utc=True),name=index.name)
        mapped = pd.DataFrame(index=index)
        for name in (name for name in line_names(type(self)) if name != "datetime"):
            source = column(params.get(name,-1), name)
            mapped[name] = frame[source].to_numpy() if source is not None else float("nan")
        for bound, lower in ((params["fromdate"], True), (params["todate"], False)):
            if bound is not None:
                if isinstance(bound,dt.date) and not isinstance(bound,dt.datetime):
                    bound=dt.datetime.combine(bound,params['sessionstart' if lower else 'sessionend'])
                stamp=pd.Timestamp(bound)
                if output_zone is not None:
                    stamp=pd.Timestamp(localize(stamp.to_pydatetime(warn=False),output_zone)).tz_convert('UTC')
                # BT converts both input dates and ordinary naive bounds to UTC
                # instants. Only adapt comparison representation; source times
                # and the business/execution clock are not rewritten.
                if mapped.index.tz is not None:
                    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp
                elif stamp.tzinfo is not None:
                    stamp = stamp.tz_convert('UTC').tz_localize(None)
                mapped = mapped.loc[mapped.index >= stamp if lower else mapped.index <= stamp]
        super().__init__(mapped, name=params["name"], _preserve_row_order=True, **{k:v for k,v in params.items() if k != "name"})
        self._timeframe, self._compression = self.p.timeframe, self.p.compression
        for name in (n for n in line_names(type(self)) if n not in {"datetime","open","high","low","close","volume"}):
            line=LineSeries(self.get_array(name))
            line._cursor_source=self
            setattr(self,name,line)
            setattr(self.lines,name,line)
            self._lines_list.append(line)



__all__ = ["PandasData"]
