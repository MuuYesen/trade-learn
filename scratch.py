from tradelearn._rust import RustBacktestEngine
import numpy as np

# Create a minimal engine with 2 bars
engine = RustBacktestEngine(
    np.array([1704067200, 1704153600], dtype=np.int64),
    np.array([100.0, 101.0]), np.array([105.0, 106.0]),
    np.array([95.0, 96.0]), np.array([102.0, 103.0]),
    np.array([1000.0, 1100.0]),
    100000.0, 0.0, False, False, False, 0.0, 0.0, False, False, False, 1.0, 1.0
)

# Submit a buy order
engine.submit_order("buy", "market", 10.0)
engine.step_open(1)
engine.step_close(1)

fills = engine.get_fills()
print(f"Fills: {fills}")
