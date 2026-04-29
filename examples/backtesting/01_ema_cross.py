#!/usr/bin/env python3
"""
Estrategia 1: EMA Crossover (EMA 9/21) con Trailing Stop

Compra: EMA 9 cruza por encima de EMA 21
Venta: EMA 9 cruza por debajo de EMA 21
Trailing Stop: Ajustado al 3% por debajo del máximo alcanzado en largo
"""

import os
from pathlib import Path

import pandas as pd
from tradelearn.lite import Backtest, Strategy

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data" / "backtesting"


class EMA_Cross_Strategy(Strategy):
    """
    Estrategia de cruce de medias móviles exponenciales
    """
    # Parámetros configurables
    ema_fast = 9
    ema_slow = 21
    trailing_pct = 0.03  # 3% trailing stop
    position_pct = 0.95  # 95% del equity
    
    def init(self):
        close = self.data.close.df
        self.ema9_ind = self.I(lambda: close.ewm(span=self.ema_fast, adjust=False).mean())
        self.ema21_ind = self.I(lambda: close.ewm(span=self.ema_slow, adjust=False).mean())
        self.highest_since_entry = 0
        
    def next(self):
        price = self.data.close[0]
        ema9 = self.ema9_ind[0]
        ema21 = self.ema21_ind[0]
        ema9_prev = self.ema9_ind[-1] if len(self.ema9_ind) > 1 else ema9
        ema21_prev = self.ema21_ind[-1] if len(self.ema21_ind) > 1 else ema21
        
        # Si estamos en posición, manejar trailing stop
        if self.position():
            # Actualizar máximo alcanzado
            if price > self.highest_since_entry:
                self.highest_since_entry = price
            
            # Calcular nivel de trailing stop
            trailing_stop = self.highest_since_entry * (1 - self.trailing_pct)
            
            # Cerrar si el precio cae por debajo del trailing stop
            if price < trailing_stop:
                self.position().close()
                self.highest_since_entry = 0
                return
            
            # Cerrar si hay cruce bajista (ema9 cruza por debajo de ema21)
            if ema9_prev >= ema21_prev and ema9 < ema21:
                self.position().close()
                self.highest_since_entry = 0
                return
        
        # Si no estamos en posición, buscar señal de entrada
        else:
            # Señal de compra: cruce alcista (ema9 cruza por encima de ema21)
            if ema9_prev <= ema21_prev and ema9 > ema21:
                # Calcular tamaño como porcentaje del equity
                self.buy(size=self.position_pct)
                self.highest_since_entry = price


def load_data(symbol):
    """Carga datos históricos y escala si es necesario"""
    filepath = DATA_DIR / f'{symbol}_30m.csv'
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Para BTC, escalamos los precios para poder operar
    if symbol == 'BTCUSDT':
        scale_factor = 1000  # Trabajar en miles de dólares
        df['Open'] = df['Open'] / scale_factor
        df['High'] = df['High'] / scale_factor
        df['Low'] = df['Low'] / scale_factor
        df['Close'] = df['Close'] / scale_factor
        df['Volume'] = df['Volume'] * scale_factor  # Mantener proporción volumen*precio
    
    return df


def run_backtest(symbol):
    """Ejecuta backtest para un símbolo"""
    print(f"\n{'='*60}")
    print(f"Backtest EMA Cross - {symbol}")
    print(f"{'='*60}")
    
    # Cargar datos
    data = load_data(symbol)
    
    # Capital inicial - más alto para BTC debido al escalado
    initial_cash = 5000 if symbol == 'ETHUSDT' else 500000  # $500K para BTC escalado
    
    # Crear y ejecutar backtest
    bt = Backtest(
        data, 
        EMA_Cross_Strategy,
        cash=initial_cash,
        commission=0.0008,  # 0.08% taker fee
        exclusive_orders=True
    )
    
    stats = bt.run()
    
    # Guardar resultados
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar gráfico
    bt.plot(filename=f'{results_dir}/ema_cross_{symbol}.html', open_browser=False)
    
    # Retornar estadísticas
    return {
        'symbol': symbol,
        'strategy': 'EMA_Cross_9_21',
        'return_pct': stats['Return [%]'],
        'return_ann_pct': stats.get('Return (Ann.) [%]', 0.0),
        'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown_pct': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate_pct': stats['Win Rate [%]'],
        'total_trades': stats['# Trades'],
        'profit_factor': stats.get('Profit Factor', 0),
        'avg_trade_pct': stats.get('Avg. Trade [%]', 0.0),
        'equity_final': stats['Equity Final [$]']
    }


if __name__ == '__main__':
    results_btc = run_backtest('BTCUSDT')
    results_eth = run_backtest('ETHUSDT')
    
    print("\n" + "="*60)
    print("RESUMEN EMA CROSS STRATEGY")
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
