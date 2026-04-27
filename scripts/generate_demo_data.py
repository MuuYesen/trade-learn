import pandas as pd
import numpy as np
from pathlib import Path

def generate_demo_data(symbol: str, periods: int = 200):
    index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    close = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, size=periods))
    df = pd.DataFrame({
        "open": close * (1 - 0.001 * np.random.rand(periods)),
        "high": close * (1 + 0.002 * np.random.rand(periods)),
        "low": close * (1 - 0.002 * np.random.rand(periods)),
        "close": close,
        "volume": np.random.randint(1000, 10000, size=periods).astype(float)
    }, index=index)
    
    output_dir = Path("data/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / f"{symbol}.parquet")
    print(f"Generated data/demo/{symbol}.parquet")

if __name__ == "__main__":
    generate_demo_data("AAPL")
    generate_demo_data("GOOG")
    generate_demo_data("TSLA")
