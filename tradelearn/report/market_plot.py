"""Lightweight Charts market replay HTML export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

LIGHTWEIGHT_CHARTS_CDN = (
    "https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"
)


def market_replay_html(
    *,
    market_data: pd.DataFrame | None,
    fills: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
    title: str = "Tradelearn Market Replay",
    script_url: str = LIGHTWEIGHT_CHARTS_CDN,
) -> str:
    """Return a self-contained market replay HTML document using Lightweight Charts."""
    payload = _payload(market_data=market_data, fills=fills, equity=equity)
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <script src="{escape(script_url)}"></script>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --border: #d8e1e8;
      --text: #1f2d38;
      --muted: #6c7a86;
      --grid: #edf2f6;
      --up: #26a69a;
      --down: #ef5350;
      --blue: #2962ff;
      --amber: #f5b942;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
        Arial, sans-serif;
    }}
    .shell {{ max-width: 1480px; margin: 0 auto; padding: 20px 22px 28px; }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }}
    h1 {{ font-size: 22px; line-height: 1.2; margin: 0; font-weight: 720; letter-spacing: .01em; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(31, 45, 56, 0.06);
      overflow: hidden;
    }}
    .chart-title {{
      height: 34px;
      padding: 8px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      font-weight: 700;
      color: #314451;
      background: linear-gradient(180deg, #ffffff, #f9fbfd);
    }}
    #equity {{ height: 150px; }}
    #price {{ height: 520px; }}
    #volume {{ height: 130px; }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      padding: 9px 14px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfe;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 6px;
      border-radius: 2px;
      vertical-align: middle;
    }}
    .legend .up::before {{ background: var(--up); }}
    .legend .down::before {{ background: var(--down); }}
    .legend .equity::before {{ background: var(--blue); }}
    .legend .trade::before {{ background: var(--amber); }}
    .empty {{ padding: 40px; color: var(--muted); text-align: center; }}
  </style>
</head>
<body>
  <main class="shell">
    <div class="header">
      <h1>{escape(title)}</h1>
      <div class="meta">Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    </div>
    <section class="chart-card">
      <div class="chart-title">Equity</div>
      <div id="equity"></div>
      <div class="chart-title">OHLC / Trades</div>
      <div id="price"></div>
      <div class="chart-title">Volume</div>
      <div id="volume"></div>
      <div class="legend">
        <span class="equity">Strategy equity</span>
        <span class="up">Up candle / buy</span>
        <span class="down">Down candle / sell</span>
        <span class="trade">Trade marker</span>
      </div>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const LC = window.LightweightCharts;
    const commonLayout = {{
      layout: {{ background: {{ color: '#ffffff' }}, textColor: '#314451', fontSize: 12 }},
      grid: {{ vertLines: {{ color: '#edf2f6' }}, horzLines: {{ color: '#edf2f6' }} }},
      rightPriceScale: {{ borderColor: '#d8e1e8' }},
      timeScale: {{ borderColor: '#d8e1e8', timeVisible: true, secondsVisible: false }},
      crosshair: {{ mode: LC.CrosshairMode.Normal }},
    }};

    function makeChart(id, height) {{
      const el = document.getElementById(id);
      const chart = LC.createChart(el, {{ ...commonLayout, height }});
      new ResizeObserver(entries => {{
        for (const entry of entries) {{
          chart.applyOptions({{ width: Math.floor(entry.contentRect.width) }});
        }}
      }}).observe(el);
      return chart;
    }}

    if (!payload.candles.length) {{
      document.getElementById('price').innerHTML = '<div class="empty">No market data</div>';
    }} else {{
      const equityChart = makeChart('equity', 150);
      const priceChart = makeChart('price', 520);
      const volumeChart = makeChart('volume', 130);

      if (payload.equity.length) {{
        const equitySeries = equityChart.addSeries(
          LC.LineSeries,
          {{ color: '#2962ff', lineWidth: 2 }}
        );
        equitySeries.setData(payload.equity);
      }}

      const candles = priceChart.addSeries(LC.CandlestickSeries, {{
        upColor: '#26a69a', downColor: '#ef5350', borderUpColor: '#16877d',
        borderDownColor: '#d33f3c', wickUpColor: '#78909c', wickDownColor: '#78909c'
      }});
      candles.setData(payload.candles);
      if (payload.markers.length) candles.setMarkers(payload.markers);

      if (payload.tradeLines.length) {{
        for (const line of payload.tradeLines) {{
          const series = priceChart.addSeries(LC.LineSeries, {{
            color: line.color, lineWidth: 2, lineStyle: LC.LineStyle.Dotted,
            priceLineVisible: false, lastValueVisible: false,
          }});
          series.setData(line.points);
        }}
      }}

      if (payload.volume.length) {{
        const volume = volumeChart.addSeries(LC.HistogramSeries, {{
          color: '#90a4ae', priceFormat: {{ type: 'volume' }}, priceLineVisible: false,
        }});
        volume.setData(payload.volume);
      }}

      const charts = [equityChart, priceChart, volumeChart];
      charts.forEach(chart => chart.timeScale().fitContent());
      const syncRange = source => {{
        const range = source.timeScale().getVisibleLogicalRange();
        if (!range) return;
        charts.forEach(chart => {{
          if (chart !== source) chart.timeScale().setVisibleLogicalRange(range);
        }});
      }};
      charts.forEach(chart => {{
        chart.timeScale().subscribeVisibleLogicalRangeChange(() => syncRange(chart));
      }});
    }}
  </script>
</body>
</html>
"""


def write_market_replay_html(
    path: str | Path,
    *,
    market_data: pd.DataFrame | None,
    fills: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
    title: str = "Tradelearn Market Replay",
) -> Path:
    """Write a Lightweight Charts market replay HTML document."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        market_replay_html(market_data=market_data, fills=fills, equity=equity, title=title),
        encoding="utf-8",
    )
    return output


def _payload(
    *,
    market_data: pd.DataFrame | None,
    fills: pd.DataFrame | None,
    equity: pd.Series | None,
) -> dict[str, Any]:
    market = _market_frame(market_data)
    fill_frame = _fills_frame(fills)
    fill_frame = _attach_fill_time(fill_frame, market)
    return {
        "candles": _candles(market),
        "volume": _volume(market),
        "equity": _equity(equity),
        "markers": _markers(fill_frame),
        "tradeLines": _trade_lines(fill_frame),
    }


def _market_frame(market_data: pd.DataFrame | None) -> pd.DataFrame:
    if market_data is None or market_data.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(market_data).copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "close" not in frame:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "time"})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    for column in ["open", "high", "low"]:
        if column not in frame:
            frame[column] = frame["close"]
    if "volume" not in frame:
        frame["volume"] = 0.0
    return frame.dropna(subset=["time", "open", "high", "low", "close"])


def _fills_frame(fills: pd.DataFrame | None) -> pd.DataFrame:
    if fills is None or fills.empty or not {"datetime", "price", "side"}.issubset(fills.columns):
        return pd.DataFrame()
    frame = pd.DataFrame(fills).copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    return frame.dropna(subset=["datetime", "price"])


def _attach_fill_time(fills: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if fills.empty or market.empty:
        return fills
    projected = pd.merge_asof(
        fills.sort_values("datetime"),
        market[["time"]].sort_values("time"),
        left_on="datetime",
        right_on="time",
        direction="nearest",
    )
    return projected.dropna(subset=["time"])


def _candles(market: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "time": _time(row.time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in market.itertuples(index=False)
    ]


def _volume(market: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "time": _time(row.time),
            "value": float(row.volume),
            "color": "rgba(38, 166, 154, 0.45)"
            if float(row.close) >= float(row.open)
            else "rgba(239, 83, 80, 0.45)",
        }
        for row in market.itertuples(index=False)
    ]


def _equity(equity: pd.Series | None) -> list[dict[str, Any]]:
    if equity is None or equity.empty:
        return []
    series = pd.Series(equity).dropna()
    if series.empty:
        return []
    base = float(series.iloc[0])
    if not base:
        return []
    return [
        {"time": _time(index), "value": float(value) / base}
        for index, value in series.items()
        if pd.notna(value)
    ]


def _markers(fills: pd.DataFrame) -> list[dict[str, Any]]:
    if fills.empty:
        return []
    markers = []
    for fill in fills.itertuples(index=False):
        side = str(fill.side).lower()
        is_buy = side == "buy"
        markers.append(
            {
                "time": _time(fill.time),
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#26a69a" if is_buy else "#ef5350",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": f"{'Buy' if is_buy else 'Sell'} {float(fill.price):.4g}",
            }
        )
    return markers


def _trade_lines(fills: pd.DataFrame) -> list[dict[str, Any]]:
    if fills.empty:
        return []
    active: dict[str, Any] = {}
    lines = []
    for fill in fills.sort_values("time").itertuples(index=False):
        side = str(fill.side).lower()
        signed = float(getattr(fill, "size", 0.0) or 0.0)
        if signed == 0:
            signed = 1.0 if side == "buy" else -1.0
        direction = 1 if signed > 0 else -1
        data_name = str(getattr(fill, "data", "") or "__default__")
        current = active.get(data_name)
        if current is None:
            active[data_name] = fill
            continue
        current_side = str(current.side).lower()
        current_direction = 1 if current_side == "buy" else -1
        if current_direction == direction:
            active[data_name] = fill
            continue
        entry_price = float(current.price)
        exit_price = float(fill.price)
        pnl = (exit_price / entry_price - 1.0) * current_direction if entry_price else 0.0
        lines.append(
            {
                "color": "rgba(38, 166, 154, 0.8)" if pnl >= 0 else "rgba(239, 83, 80, 0.8)",
                "points": [
                    {"time": _time(current.time), "value": entry_price},
                    {"time": _time(fill.time), "value": exit_price},
                ],
            }
        )
        active.pop(data_name, None)
    return lines


def _time(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat(timespec="seconds")
