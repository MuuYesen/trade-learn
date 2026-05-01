#!/usr/bin/env python3
"""
Estrategia 3: MACD + Confirmación de Volumen

Compra: MACD cruza por encima de la señal Y volumen > promedio 20 velas
Venta: MACD cruza por debajo de la señal
Stop Loss: ATR 14 * 1.5
Take Profit: ATR 14 * 3 (ratio 1:2)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import tradelearn.lite as tl

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data" / "backtesting"


class MACD_Volume_Strategy(tl.Strategy):
    """
    Estrategia MACD + Confirmación de Volumen
    """
    # Parámetros MACD
    macd_fast = 12
    macd_slow = 26
    macd_signal_period = 9
    
    # Parámetros de riesgo
    position_pct = 0.95  # 95% del equity
    atr_period = 14
    sl_multiplier = 1.5
    tp_multiplier = 3.0
    
    # Volumen mínimo: 1.2x el promedio
    volume_threshold = 1.2
    
    def init(self):
        close = self.data.close.df
        volume = self.data.volume.df
        
        # Calcular MACD
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
        
        self.macd_line_ind = self.I(lambda: macd_line)
        self.macd_signal_ind = self.I(lambda: macd_signal_line)
        
        # Promedio de volumen
        self.volume_sma = self.I(lambda: volume.rolling(window=20).mean())
        
        # ATR para stops
        high = self.data.high.df
        low = self.data.low.df
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = self.I(lambda: tr.rolling(window=self.atr_period).mean())
        
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0
        
    def next(self):
        price = self.data.close[0]
        volume = self.data.volume[0]
        atr = self.atr[0]
        macd_line = self.macd_line_ind[0]
        macd_signal = self.macd_signal_ind[0]
        macd_line_prev = self.macd_line_ind[-1] if len(self.macd_line_ind) > 1 else macd_line
        macd_signal_prev = (
            self.macd_signal_ind[-1] if len(self.macd_signal_ind) > 1 else macd_signal
        )
        
        # Si estamos en posición
        if self.position():
            # Stop Loss
            if price <= self.sl_price:
                self.position().close()
                self.entry_price = 0
                return
            
            # Take Profit
            if price >= self.tp_price:
                self.position().close()
                self.entry_price = 0
                return
            
            # Señal de salida: MACD cruce bajista (señal cruza por encima de línea)
            if macd_signal_prev <= macd_line_prev and macd_signal > macd_line:
                self.position().close()
                self.entry_price = 0
                return
        
        # Buscar entrada
        else:
            # Confirmación de volumen
            vol_avg = self.volume_sma[0]
            volume_ok = volume > (vol_avg * self.volume_threshold)
            
            # Señal de compra: MACD cruce alcista (línea cruza por encima de señal)
            if (
                macd_line_prev <= macd_signal_prev
                and macd_line > macd_signal
                and volume_ok
                and not np.isnan(atr)
            ):
                # Calcular stops basados en ATR
                self.sl_price = price - (atr * self.sl_multiplier)
                self.tp_price = price + (atr * self.tp_multiplier)
                
                self.buy(size=self.position_pct)
                self.entry_price = price


def load_data(symbol):
    """Carga datos históricos y escala si es necesario"""
    filepath = DATA_DIR / f'{symbol}_30m.csv'
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if symbol == 'BTCUSDT':
        scale_factor = 1000
        df['Open'] = df['Open'] / scale_factor
        df['High'] = df['High'] / scale_factor
        df['Low'] = df['Low'] / scale_factor
        df['Close'] = df['Close'] / scale_factor
        df['Volume'] = df['Volume'] * scale_factor
    
    return df


def run_backtest(symbol):
    """Ejecuta backtest"""
    print(f"\n{'='*60}")
    print(f"Backtest MACD + Volume - {symbol}")
    print(f"{'='*60}")
    
    data = load_data(symbol)
    initial_cash = 5000 if symbol == 'ETHUSDT' else 500000
    
    bt = tl.Backtest(
        data,
        MACD_Volume_Strategy,
        cash=initial_cash,
        commission=0.0008,
        exclusive_orders=True
    )
    
    stats = bt.run()
    
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    bt.plot(filename=f'{results_dir}/macd_volume_{symbol}.html', open_browser=False)
    
    return {
        'symbol': symbol,
        'strategy': 'MACD_Volume',
        'return_pct': stats['return_pct'],
        'return_ann_pct': stats.get('Return (Ann.) [%]', 0.0),
        'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown_pct': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate_pct': stats['win_rate_pct'],
        'total_trades': stats['total_trades'],
        'profit_factor': stats.get('Profit Factor', 0),
        'avg_trade_pct': stats.get('Avg. Trade [%]', 0.0),
        'equity_final': stats['final_value']
    }


if __name__ == '__main__':
    results_btc = run_backtest('BTCUSDT')
    results_eth = run_backtest('ETHUSDT')
    
    print("\n" + "="*60)
    print("RESUMEN MACD + VOLUMEN")
    print("="*60)
    
    for r in [results_btc, results_eth]:
        print(f"\n{r['symbol']}:")
        print(f"  Retorno Total: {r['return_pct']:.2f}%")
        print(f"  Retorno Anualizado: {r['return_ann_pct']:.2f}%")
        print(f"  Max Drawdown: {r['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {r['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {r['win_rate_pct']:.2f}%")
        print(f"  Total Trades: {r['total_trades']}")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
