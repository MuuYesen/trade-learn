import importlib
import pickle

import numpy as np
import pandas as pd


def test_training_artifact_and_runtime_prefix_match():
    workflow = importlib.import_module("examples.research._workflow")
    dates = pd.date_range("2026-01-01", periods=50, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "symbol"])
    close = np.array([100 + i + 3 * np.sin(i / 3 + j) for i in range(50) for j in range(2)])
    bars = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0}, index=index
    )
    bundle, result, test_bars = workflow.train(bars, split=dates[30])
    assert test_bars.index.get_level_values("timestamp").max() == dates[-1]
    assert result.artifacts["label"]["purged_rows"] > 0
    assert (result.target.label_end < dates[30]).all()
    restored = pickle.loads(pickle.dumps(bundle))
    timestamp = dates[40]
    prefix = bars.loc[bars.index.get_level_values("timestamp") <= timestamp]
    bounded = prefix.groupby(level="symbol", group_keys=False).tail(bundle["window_size"])
    full = restored["pipeline"].transform(restored["feature_set"].transform(prefix))
    current = restored["pipeline"].transform(restored["feature_set"].transform(bounded))
    pd.testing.assert_series_equal(
        restored["scorer"].predict(full), restored["scorer"].predict(current)
    )
    assert {step.name for step in result.steps} >= {
        "feature_set",
        "pipeline",
        "model",
        "scorer",
        "allocator",
    }


def test_engine_and_lite_execute_fitted_research_without_refit(monkeypatch):
    import tradelearn.engine as bt
    import tradelearn.lite as tl
    from examples.research._workflow import train
    from examples.research.index_enhance_engine_pipeline import EngineResearchIndexEnhance
    from examples.research.index_enhance_lite_pipeline import LiteResearchIndexEnhance

    dates = pd.date_range("2026-01-01", periods=45, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "symbol"])
    close = np.array([100 + i + np.sin(i + j) for i in range(45) for j in range(2)])
    bars = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0}, index=index
    )
    bundle, _, test_bars = train(bars, split=dates[25])
    # Assert the research contract directly: same frozen predictions/allocations,
    # no refit. Engine f64 and legacy Lite six-place execution are separate.
    from examples.research import index_enhance_engine_pipeline as engine_example
    from examples.research import index_enhance_lite_pipeline as lite_example
    decisions = {'engine': [], 'lite': []}
    def capture(name, original):
        def apply(bundle, window):
            result = original(bundle, window)
            if result is not None and not result.empty:
                decisions[name].append(result.copy())
            return result
        return apply
    monkeypatch.setattr(engine_example, 'infer', capture('engine', engine_example.infer))
    monkeypatch.setattr(lite_example, 'infer', capture('lite', lite_example.infer))
    def no_fit(*args, **kwargs):
        raise AssertionError('runtime attempted to refit the frozen model')
    monkeypatch.setattr(bundle['model'], 'fit', no_fit)

    cerebro = bt.Cerebro()
    for symbol in ["A", "B"]:
        cerebro.adddata(
            bt.feeds.PandasData(dataname=test_bars.xs(symbol, level="symbol")), name=symbol
        )
    cerebro.broker.setcash(100_000.0)
    cerebro.addstrategy(EngineResearchIndexEnhance, bundle=bundle)
    [strategy] = cerebro.run()
    lite = tl.Backtest(test_bars, LiteResearchIndexEnhance, cash=100_000.0)
    result = lite.run(bundle=bundle)
    assert decisions['engine'] and len(decisions['engine']) == len(decisions['lite'])
    for engine_weights, lite_weights in zip(decisions['engine'], decisions['lite']):
        pd.testing.assert_series_equal(engine_weights, lite_weights)
    assert result.summary["final_value"] > 0
    assert strategy.stats.summary["final_value"] > 0
