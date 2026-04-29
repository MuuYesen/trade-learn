"""Lightweight Charts market replay HTML export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

LIGHTWEIGHT_CHARTS_CDN = (
    "https://unpkg.com/lightweight-charts@5/dist/lightweight-charts.standalone.production.js"
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
      --bg: #f3f6f9;
      --panel: #ffffff;
      --border: #d6e0e8;
      --text: #1f2d38;
      --muted: #697985;
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
    .shell {{ max-width: 1480px; margin: 0 auto; padding: 18px 22px 28px; }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }}
    h1 {{ font-size: 22px; line-height: 1.2; margin: 0; font-weight: 740; }}
    .meta {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
    .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(31, 45, 56, 0.06);
      overflow: hidden;
    }}
    .chart-head {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 1px;
      background: var(--border);
      border-bottom: 1px solid var(--border);
    }}
    .metric {{ background: #fbfcfe; padding: 10px 14px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .metric-value {{ color: var(--text); font-size: 18px; font-weight: 740; }}
    #market-chart {{ height: 860px; }}
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
    .legend .pl::before {{ background: var(--amber); }}
    .attribution {{ padding: 0 14px 12px; color: var(--muted); font-size: 11px; }}
    .attribution a {{ color: #3d6f8f; text-decoration: none; }}
    .empty {{ padding: 56px; color: var(--muted); text-align: center; }}
    @media (max-width: 900px) {{
      .chart-head {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      #market-chart {{ height: 760px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <div class="header">
      <h1>{escape(title)}</h1>
      <div class="meta">Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    </div>
    <section class="chart-card">
      <div class="chart-head">
        <div class="metric"><div class="metric-label">Final Equity</div><div class="metric-value" id="m-equity">-</div></div>
        <div class="metric"><div class="metric-label">Return</div><div class="metric-value" id="m-return">-</div></div>
        <div class="metric"><div class="metric-label">Trades</div><div class="metric-value" id="m-trades">-</div></div>
        <div class="metric"><div class="metric-label">Bars</div><div class="metric-value" id="m-bars">-</div></div>
      </div>
      <div id="market-chart"></div>
      <div class="legend">
        <span class="equity">Equity</span>
        <span class="pl">Profit / Loss</span>
        <span class="up">Up candle / buy</span>
        <span class="down">Down candle / sell</span>
      </div>
      <div class="attribution">
        Charts powered by <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView Lightweight Charts</a>.
      </div>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const LC = window.LightweightCharts;
    const container = document.getElementById('market-chart');

    function fmtPct(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return `${{(value * 100).toFixed(2)}}%`;
    }}
    function fmtMoney(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
    }}
    document.getElementById('m-bars').textContent = payload.stats.bars;
    document.getElementById('m-trades').textContent = payload.stats.trades;
    document.getElementById('m-equity').textContent = fmtMoney(payload.stats.finalEquity);
    document.getElementById('m-return').textContent = fmtPct(payload.stats.returnPct);

    if (!payload.candles.length) {{
      container.innerHTML = '<div class="empty">No market data</div>';
    }} else {{
      const chart = LC.createChart(container, {{
        autoSize: true,
        height: 860,
        attributionLogo: false,
        layout: {{
          background: {{ color: '#ffffff' }},
          textColor: '#314451',
          fontSize: 12,
          attributionLogo: false,
          panes: {{
            separatorColor: '#d6e0e8',
            separatorHoverColor: '#9fb3c1',
            enableResize: true,
          }},
        }},
        grid: {{
          vertLines: {{ color: '#edf2f6' }},
          horzLines: {{ color: '#edf2f6' }},
        }},
        rightPriceScale: {{ borderColor: '#d6e0e8' }},
        timeScale: {{ borderColor: '#d6e0e8', timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: LC.CrosshairMode.Normal }},
      }});

      const equitySeries = chart.addSeries(
        LC.LineSeries,
        {{ color: '#2962ff', lineWidth: 2, priceLineVisible: false }},
        0
      );
      if (payload.equity.length) equitySeries.setData(payload.equity);

      const plSeries = chart.addSeries(
        LC.HistogramSeries,
        {{
          priceFormat: {{ type: 'percent' }},
          priceLineVisible: false,
          base: 0,
        }},
        1
      );
      if (payload.pnl.length) plSeries.setData(payload.pnl);

      const candles = chart.addSeries(LC.CandlestickSeries, {{
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderUpColor: '#16877d',
        borderDownColor: '#d33f3c',
        wickUpColor: '#78909c',
        wickDownColor: '#78909c',
      }}, 2);
      candles.setData(payload.candles);
      if (payload.markers.length) LC.createSeriesMarkers(candles, payload.markers);

      if (payload.tradeLines.length) {{
        for (const line of payload.tradeLines) {{
          const series = chart.addSeries(LC.LineSeries, {{
            color: line.color,
            lineWidth: 2,
            lineStyle: LC.LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
          }}, 2);
          series.setData(line.points);
        }}
      }}

      const volume = chart.addSeries(LC.HistogramSeries, {{
        priceFormat: {{ type: 'volume' }},
        priceLineVisible: false,
      }}, 3);
      if (payload.volume.length) volume.setData(payload.volume);

      chart.timeScale().fitContent();
      const panes = chart.panes ? chart.panes() : [];
      if (panes.length >= 4) {{
        panes[0].setHeight(150);
        panes[1].setHeight(110);
        panes[2].setHeight(500);
        panes[3].setHeight(120);
      }}
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
    trade_lines = _trade_lines(fill_frame)
    return {
        "candles": _candles(market),
        "volume": _volume(market),
        "equity": _equity(equity),
        "markers": _markers(fill_frame),
        "tradeLines": trade_lines,
        "pnl": _profit_loss(trade_lines),
        "stats": _stats(market, equity, trade_lines),
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
                "returnPct": pnl,
                "points": [
                    {"time": _time(current.time), "value": entry_price},
                    {"time": _time(fill.time), "value": exit_price},
                ],
            }
        )
        active.pop(data_name, None)
    return lines


def _profit_loss(trade_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "time": line["points"][-1]["time"],
            "value": float(line.get("returnPct", 0.0)),
            "color": "rgba(38, 166, 154, 0.75)"
            if float(line.get("returnPct", 0.0)) >= 0
            else "rgba(239, 83, 80, 0.75)",
        }
        for line in trade_lines
        if line.get("points")
    ]


def _stats(
    market: pd.DataFrame,
    equity: pd.Series | None,
    trade_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    final_equity: float | None = None
    return_pct: float | None = None
    if equity is not None and not equity.empty:
        series = pd.Series(equity).dropna()
        if not series.empty:
            first = float(series.iloc[0])
            final_equity = float(series.iloc[-1])
            return_pct = final_equity / first - 1.0 if first else None
    return {
        "bars": int(len(market)),
        "trades": int(len(trade_lines)),
        "finalEquity": final_equity,
        "returnPct": return_pct,
    }


def _time(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat(timespec="seconds")
