from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "zoo" / "tushare_sw_hs300" / "alpha101_hs300_backtest.py"
STRATEGY = ROOT / "zoo" / "tushare_sw_hs300" / "hs300_alpha101_strategy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("alpha101_hs300_backtest", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_hs300_backtest_uses_repo_relative_paths_and_hs300_names() -> None:
    source = _source()

    assert "D:/Program_Files" not in source
    assert "SCRIPT_DIR" in source
    assert "LOCAL_DATA_PATH = (" in source
    assert "\"data\"" in source
    assert "\"hs300_20150101_20260601\"" in source
    assert "\"000300SH_20150101_20260601.csv\"" in source
    assert "HS300_BENCHMARK_PATH = (" in source
    assert "\"raw\"" in source
    assert "\"index_daily_000300_20150101_20260601.csv\"" in source
    assert "alpha101-hs300-top20" in source
    assert "alpha101_hs300_top20_backtest_report.html" in source
    assert "alpha101_us_tech" not in source


def test_hs300_backtest_trades_next_open_only_on_rebalance_dates() -> None:
    source = _source()
    strategy_source = STRATEGY.read_text(encoding="utf-8")

    assert "trade_on_close=False" in source
    assert "weights.has_current()" in strategy_source


def test_hs300_backtest_separates_factor_count_from_stock_count() -> None:
    source = _source()

    assert "TOP_N_FACTORS = 5" in source
    assert "TOP_N_STOCKS = 20" in source
    assert "MIN_AUX_COVERAGE = 0.90" in source
    assert "ENFORCE_TRADE_CONSTRAINTS = True" in source
    assert "tradestatus" in source
    assert "up_limit" in source
    assert "down_limit" in source
    assert "selected = ranking.head(TOP_N_FACTORS).copy()" in source
    assert "k=top_n_stocks" in source
    assert "TOP_K" not in source


def test_hs300_backtest_sorts_rebalance_dates_explicitly() -> None:
    source = _source()

    assert ".unique().sort_values()[::rebalance_every]" in source


def test_hs300_backtest_reports_actual_loaded_date_range() -> None:
    source = _source()

    assert "actual data range:" in source


def test_load_hs300_benchmark_returns_uses_index_close_returns(tmp_path) -> None:
    module = _load_module()
    csv_path = tmp_path / "index_daily_000300.csv"
    csv_path.write_text(
        "\n".join(
            [
                "index_code,date,index_close,index_pctChg",
                "399300.SZ,2023-01-04,102.0,0.1",
                "399300.SZ,2023-01-02,100.0,0.0",
                "399300.SZ,2023-01-03,101.0,5.0",
                "399300.SZ,2023-01-05,104.0,0.2",
            ]
        ),
        encoding="utf-8",
    )

    returns = module.load_hs300_benchmark_returns(
        csv_path,
        start="2023-01-03",
        end="2023-01-05",
    )

    expected = pd.Series(
        [0.0, 102.0 / 101.0 - 1.0, 104.0 / 102.0 - 1.0],
        index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
        name="HS300",
    )
    expected.index.name = "date"
    pd.testing.assert_series_equal(returns, expected)


def test_hs300_backtest_passes_hs300_benchmark_to_report() -> None:
    source = _source()

    assert "HS300_BENCHMARK_PATH" in source
    assert "benchmark_returns = load_hs300_benchmark_returns(" in source
    assert "benchmark=benchmark_returns" in source


def test_compute_alpha101_factors_uses_matching_cache(tmp_path, monkeypatch) -> None:
    module = _load_module()
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100.0, 200.0],
            "vwap": [1.4, 2.4],
        }
    )
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB"],
            "alpha001_101": [0.1, 0.2],
        }
    )
    cache_path = tmp_path / "factors.parquet"
    expected.to_parquet(cache_path)
    module.write_cache_meta(cache_path, module.factor_cache_key(bars))

    def fail_alpha101(_bars):
        raise AssertionError("alpha101 should not run when cache is valid")

    monkeypatch.setattr(module, "alpha101", fail_alpha101)

    result = module.compute_alpha101_factors(bars, cache_path)

    pd.testing.assert_frame_equal(result, expected)


def test_compute_alpha101_factors_can_force_cache_update(tmp_path, monkeypatch) -> None:
    module = _load_module()
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100.0, 200.0],
            "vwap": [1.4, 2.4],
        }
    )
    stale = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "symbol": ["AAA"],
            "alpha001_101": [99.0],
        }
    )
    fresh = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "symbol": ["AAA"],
            "alpha001_101": [1.0],
        }
    )
    cache_path = tmp_path / "factors.parquet"
    stale.to_parquet(cache_path)
    module.write_cache_meta(cache_path, module.factor_cache_key(bars))

    monkeypatch.setattr(module, "alpha101", lambda _bars: fresh)

    result = module.compute_alpha101_factors(bars, cache_path, update_cache=True)
    from_disk = pd.read_parquet(cache_path)

    pd.testing.assert_frame_equal(result, fresh)
    pd.testing.assert_frame_equal(from_disk, fresh)


def test_preprocessed_factor_table_cache_reads_parquet_with_matching_meta(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_path = tmp_path / "preprocessed_factor_table.parquet"
    cached = pd.DataFrame(
        {
            "alpha001_101": [0.1, 0.2],
            "ind_code": ["A", "B"],
            "cir_a": [10.0, 20.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
            ],
            names=["date", "symbol"],
        ),
    )
    cached.to_parquet(cache_path)
    module.write_cache_meta(
        cache_path,
        module.preprocessed_factor_table_cache_meta(
            factors=pd.DataFrame(),
            bars=pd.DataFrame(),
            factor_columns=("alpha001_101",),
            enable_industry_fill=True,
            enable_neutralize=True,
            min_aux_coverage=module.MIN_AUX_COVERAGE,
        ),
    )

    def fail_preprocess(*_args, **_kwargs):
        raise AssertionError("preprocess should not run when parquet cache is usable")

    monkeypatch.setattr(module, "preprocess_factor_table", fail_preprocess)

    result = module.load_or_build_preprocessed_factor_table(
        factors=pd.DataFrame(),
        bars=pd.DataFrame(),
        factor_columns=("alpha001_101",),
        cache_path=cache_path,
    )

    pd.testing.assert_frame_equal(result, cached)
    assert module.cache_meta_path(cache_path).exists()


def test_preprocessed_factor_table_cache_writes_parquet(tmp_path) -> None:
    module = _load_module()
    cache_path = tmp_path / "preprocessed_factor_table.parquet"
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB"],
            "alpha001_101": [1.0, 2.0],
        }
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100.0, 200.0],
            "vwap": [1.0, 2.0],
        }
    )

    result = module.load_or_build_preprocessed_factor_table(
        factors=factors,
        bars=bars,
        factor_columns=("alpha001_101",),
        cache_path=cache_path,
        enable_industry_fill=False,
        enable_neutralize=False,
    )

    assert cache_path.exists()
    assert module.cache_meta_path(cache_path).exists()
    from_disk = pd.read_parquet(cache_path)
    pd.testing.assert_frame_equal(result, from_disk)


def test_preprocessed_factor_table_cache_rebuilds_when_meta_mismatches(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    cache_path = tmp_path / "preprocessed_factor_table.parquet"
    stale = pd.DataFrame(
        {
            "alpha001_101": [99.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-02"), "AAA")],
            names=["date", "symbol"],
        ),
    )
    stale.to_parquet(cache_path)
    module.write_cache_meta(cache_path, {"version": -1})

    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "symbol": ["AAA"],
            "alpha001_101": [1.0],
        }
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "symbol": ["AAA"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100.0],
            "vwap": [1.0],
        }
    )

    rebuilt = module.load_or_build_preprocessed_factor_table(
        factors=factors,
        bars=bars,
        factor_columns=("alpha001_101",),
        cache_path=cache_path,
        enable_industry_fill=False,
        enable_neutralize=False,
    )

    assert rebuilt.iloc[0]["alpha001_101"] == 1.0
    assert json.loads(module.cache_meta_path(cache_path).read_text(encoding="utf-8"))[
        "enable_neutralize"
    ] is False


def test_factor_selection_cache_reads_ranking_and_selected_parquet(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    ranking_path = tmp_path / "ranking.parquet"
    selected_path = tmp_path / "selected.parquet"
    ranking = pd.DataFrame(
        {
            "factor": ["alpha040", "alpha082"],
            "column": ["alpha040_101", "alpha082_101"],
            "direction": [1, 1],
            "score": [7.0, 6.0],
        }
    )
    selected = ranking.head(1).copy()
    ranking.to_parquet(ranking_path)
    selected.to_parquet(selected_path)
    meta = module.factor_selection_cache_meta(
        factor_table=pd.DataFrame(),
        prices=pd.Series(dtype="float64"),
        factor_columns=("alpha040_101",),
        mode="single_split",
    )
    module.write_cache_meta(ranking_path, meta)
    module.write_cache_meta(selected_path, meta)

    def fail_clean(*_args, **_kwargs):
        raise AssertionError("clean_factor_and_forward_returns should not run")

    monkeypatch.setattr(module, "clean_factor_and_forward_returns", fail_clean)

    result_ranking, result_selected, clean = module.load_or_build_factor_selection(
        factor_table=pd.DataFrame(),
        prices=pd.Series(dtype="float64"),
        factor_columns=("alpha040_101",),
        ranking_cache_path=ranking_path,
        selected_cache_path=selected_path,
    )

    pd.testing.assert_frame_equal(result_ranking, ranking)
    pd.testing.assert_frame_equal(result_selected, selected)
    assert clean is None


def test_factor_selection_cache_writes_ranking_and_selected_parquet(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    ranking_path = tmp_path / "ranking.parquet"
    selected_path = tmp_path / "selected.parquet"
    ranking = pd.DataFrame(
        {
            "factor": ["alpha040"],
            "column": ["alpha040_101"],
            "direction": [1],
            "score": [7.0],
        }
    )

    monkeypatch.setattr(module, "rank_factors_fast", lambda *_args, **_kwargs: ranking)

    result_ranking, result_selected, result_clean = module.load_or_build_factor_selection(
        factor_table=pd.DataFrame(
            {"alpha040_101": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2020-01-02"), "AAA")],
                names=["date", "symbol"],
            ),
        ),
        prices=pd.Series(
            [10.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2020-01-02"), "AAA")],
                names=["date", "symbol"],
            ),
        ),
        factor_columns=("alpha040_101",),
        ranking_cache_path=ranking_path,
        selected_cache_path=selected_path,
    )

    assert ranking_path.exists()
    assert selected_path.exists()
    assert module.cache_meta_path(ranking_path).exists()
    assert module.cache_meta_path(selected_path).exists()
    pd.testing.assert_frame_equal(result_ranking, ranking)
    pd.testing.assert_frame_equal(result_selected, ranking.head(module.TOP_N_FACTORS))
    assert result_clean is None


def test_factor_selection_cache_rebuilds_when_meta_mismatches(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    ranking_path = tmp_path / "ranking.parquet"
    selected_path = tmp_path / "selected.parquet"
    stale = pd.DataFrame(
        {
            "factor": ["alpha001"],
            "column": ["alpha001_101"],
            "direction": [1],
            "score": [1.0],
        }
    )
    stale.to_parquet(ranking_path)
    stale.to_parquet(selected_path)
    module.write_cache_meta(ranking_path, {"version": -1})
    module.write_cache_meta(selected_path, {"version": -1})

    fresh = pd.DataFrame(
        {
            "factor": ["alpha040"],
            "column": ["alpha040_101"],
            "direction": [1],
            "score": [7.0],
        }
    )
    monkeypatch.setattr(module, "rank_factors_fast", lambda *_a, **_k: fresh)

    ranking, selected, _ = module.load_or_build_factor_selection(
        factor_table=pd.DataFrame(
            {"alpha040_101": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2020-01-02"), "AAA")],
                names=["date", "symbol"],
            ),
        ),
        prices=pd.Series(
            [10.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2020-01-02"), "AAA")],
                names=["date", "symbol"],
            ),
        ),
        factor_columns=("alpha040_101",),
        ranking_cache_path=ranking_path,
        selected_cache_path=selected_path,
    )

    pd.testing.assert_frame_equal(ranking, fresh)
    pd.testing.assert_frame_equal(selected, fresh.head(module.TOP_N_FACTORS))


def test_semiannual_periods_use_prior_observable_training_cutoff() -> None:
    module = _load_module()
    trading_dates = pd.to_datetime(
        [
            "2024-12-25",
            "2024-12-26",
            "2024-12-27",
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
            "2025-01-03",
            "2025-06-27",
            "2025-06-30",
            "2025-07-01",
            "2025-07-02",
        ]
    )

    periods = module.semiannual_walk_forward_periods(
        start="2025-01-01",
        end="2025-07-02",
        trading_dates=trading_dates,
        forward_period=2,
    )

    assert periods[0] == {
        "period_start": pd.Timestamp("2025-01-01"),
        "period_end": pd.Timestamp("2025-06-30"),
        "training_start": pd.Timestamp("2020-01-01"),
        "training_end": pd.Timestamp("2024-12-30"),
    }
    assert periods[1] == {
        "period_start": pd.Timestamp("2025-07-01"),
        "period_end": pd.Timestamp("2025-07-02"),
        "training_start": pd.Timestamp("2020-07-01"),
        "training_end": pd.Timestamp("2025-06-27"),
    }


def test_walk_forward_weights_use_period_specific_selected_factors() -> None:
    module = _load_module()
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-07-01", "2025-07-01"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "alpha001_101": [10.0, 1.0, 1.0, 10.0],
            "alpha002_101": [1.0, 10.0, 10.0, 1.0],
        }
    )
    selected = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2025-01-01", "2025-07-01"]),
            "period_end": pd.to_datetime(["2025-06-30", "2025-12-31"]),
            "training_start": pd.to_datetime(["2020-01-01", "2020-07-01"]),
            "training_end": pd.to_datetime(["2024-12-31", "2025-06-24"]),
            "column": ["alpha001_101", "alpha002_101"],
            "direction": [1, 1],
        }
    )

    weights = module.build_walk_forward_target_weights(
        factors,
        selected,
        rebalance_every=1,
        top_n_stocks=1,
    )

    assert weights.loc[(pd.Timestamp("2025-01-02"), "AAA")] == 1.0
    assert weights.loc[(pd.Timestamp("2025-07-01"), "AAA")] == 1.0


def test_factor_score_prefers_stronger_top_long_portfolio() -> None:
    module = _load_module()
    ranking = pd.DataFrame(
        {
            "rank_ic_mean": [0.03, 0.03],
            "rank_ic_ir": [1.0, 1.0],
            "q5_q1_annualized": [0.10, 0.10],
            "monotonicity": [0.8, 0.8],
            "turnover": [0.3, 0.3],
            "portfolio_annualized_return": [-0.20, 0.20],
            "portfolio_sharpe": [-1.0, 1.0],
            "portfolio_max_drawdown": [-0.40, -0.05],
            "portfolio_turnover": [0.5, 0.5],
        },
        index=["weak", "strong"],
    )

    scores = module.factor_score(ranking)

    assert scores.loc["strong"] > scores.loc["weak"]


def test_hs300_backtest_selects_factors_before_backtest_period() -> None:
    source = _source()

    assert "UPDATE_FACTOR_CACHE =" in source
    assert "UPDATE_PREPROCESSED_FACTOR_TABLE_CACHE =" in source
    assert "UPDATE_FACTOR_SELECTION_CACHE =" in source
    assert "update_cache=UPDATE_FACTOR_CACHE" in source
    assert "update_cache=UPDATE_PREPROCESSED_FACTOR_TABLE_CACHE" in source
    assert "update_cache=UPDATE_FACTOR_SELECTION_CACHE" in source
    assert "START = \"2015-01-01\"" in source
    assert "END = \"2026-06-01\"" in source
    assert "FACTOR_SELECTION_END = \"2019-12-31\"" in source
    assert "BACKTEST_START = \"2020-01-01\"" in source
    assert "WALK_FORWARD_TRAINING_YEARS = 5" in source
    assert "training_factor_table = filter_dates(factor_table, end=FACTOR_SELECTION_END)" in source
    assert "training_prices = filter_dates(prices, end=FACTOR_SELECTION_END)" in source
    assert "DATE_SYMBOL_COLUMNS = [\"date\", \"symbol\"]" in source
    assert "tradable_index = bars.set_index(DATE_SYMBOL_COLUMNS).sort_index().index" in source
    assert "factor_table = factor_table.loc[factor_table.index.intersection(tradable_index)]" in source
    assert "factor_table = attach_factor_metadata(factor_table, bars)" in source
    assert "factor_table = filter_tradable_factor_rows(factor_table)" in source
    assert "factor_table = preprocess_factor_table(" in source
    assert "ranking = rank_factors_fast(" in source
    assert "training_prices = filter_dates(" in source
    assert "start=str(training_start.date())" in source
    assert "end=str(training_end.date())" in source
    assert "if clean_for_ranking is None:" in source
    assert "report_representative_factor(clean_for_ranking, selected)" in source
    assert "backtest_factors = filter_dates(factor_table, start=BACKTEST_START)" in source
    assert "backtest_factors = filter_tradable_factor_rows(backtest_factors).reset_index()" in source
    assert "bars.set_index(DATE_SYMBOL_COLUMNS).sort_index()[backtest_columns]" in source


def test_hs300_backtest_documents_next_open_execution() -> None:
    source = _source()

    assert "execution model: signal date T, execute at next trading day's open" in source


def test_hs300_backtest_uses_direct_summary_schema_for_ranking() -> None:
    source = _source()

    assert "def pick_column(" not in source
    assert "def get_summary_row_for_period(" not in source
    assert "def get_ic_dates(" not in source
    assert "def infer_ic_dates_from_factor_clean(" not in source
    assert "def rank_factors_fast(" in source
    assert 'prices.groupby(level="symbol").shift(-FORWARD_PERIOD) / prices - 1.0' in source
    assert 'ranking["rank_ic_mean"]' in source
    assert '"portfolio_annualized_return": portfolio_annualized' in source


def test_preprocess_factor_table_winsorizes_by_date_without_touching_other_dates() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [2.0, 2.0, 2.0, 2.0, 100.0, -100.0, 5.0, 6.0],
            "alpha002_101": [10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 3.0, 3.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
                (pd.Timestamp("2020-01-02"), "CCC"),
                (pd.Timestamp("2020-01-02"), "DDD"),
                (pd.Timestamp("2020-01-03"), "AAA"),
                (pd.Timestamp("2020-01-03"), "BBB"),
                (pd.Timestamp("2020-01-03"), "CCC"),
                (pd.Timestamp("2020-01-03"), "DDD"),
            ],
            names=["date", "symbol"],
        ),
    )

    processed = module.preprocess_factor_table(
        factor_table,
        ("alpha001_101", "alpha002_101"),
        winsorize_scale=1.0,
    )

    first_date = pd.Timestamp("2020-01-02")
    second_date = pd.Timestamp("2020-01-03")
    assert processed.index.equals(factor_table.index)
    assert processed.loc[(first_date, "AAA"), "alpha001_101"] == 2.0
    assert processed.loc[(first_date, "DDD"), "alpha001_101"] == 2.0
    assert processed.loc[(second_date, "AAA"), "alpha001_101"] == 53.0
    assert processed.loc[(second_date, "BBB"), "alpha001_101"] == -42.0
    assert processed.loc[(second_date, "CCC"), "alpha001_101"] == 5.0
    assert processed["alpha002_101"].tolist() == factor_table["alpha002_101"].tolist()


def test_filter_tradable_factor_rows_excludes_frozen_rows() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [1.0, 2.0],
            "tradestatus": [1, 0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
            ],
            names=["date", "symbol"],
        ),
    )

    filtered = module.filter_tradable_factor_rows(factor_table)

    assert filtered.index.get_level_values("symbol").tolist() == ["AAA"]


def test_preprocess_factor_table_fills_missing_factors_by_industry_then_date() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [1.0, 3.0, None, None],
            "ind_code": ["A", "A", "B", "B"],
            "cir_a": [10.0, 11.0, 12.0, 13.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
                (pd.Timestamp("2020-01-02"), "CCC"),
                (pd.Timestamp("2020-01-02"), "DDD"),
            ],
            names=["date", "symbol"],
        ),
    )

    processed = module.preprocess_factor_table(
        factor_table,
        ("alpha001_101",),
        enable_industry_fill=True,
        enable_neutralize=False,
    )

    assert processed["alpha001_101"].tolist() == [1.0, 3.0, 2.0, 2.0]


def test_preprocess_factor_table_requires_cir_a_for_neutralization() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [1.0, 2.0, 3.0],
            "ind_code": ["A", "B", "C"],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
                (pd.Timestamp("2020-01-02"), "CCC"),
            ],
            names=["date", "symbol"],
        ),
    )

    try:
        module.preprocess_factor_table(
            factor_table,
            ("alpha001_101",),
            enable_neutralize=True,
        )
    except ValueError as exc:
        assert "cir_a" in str(exc)
    else:
        raise AssertionError("expected missing cir_a to raise ValueError")


def test_preprocess_factor_table_rejects_low_cir_a_coverage_for_neutralization() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [1.0, 2.0, 3.0],
            "ind_code": ["A", "B", "C"],
            "cir_a": [None, None, None],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
                (pd.Timestamp("2020-01-02"), "CCC"),
            ],
            names=["date", "symbol"],
        ),
    )

    try:
        module.preprocess_factor_table(
            factor_table,
            ("alpha001_101",),
            enable_neutralize=True,
        )
    except ValueError as exc:
        assert "cir_a coverage" in str(exc)
    else:
        raise AssertionError("expected low cir_a coverage to raise ValueError")


def test_preprocess_factor_table_neutralizes_integer_factors_without_dtype_warning() -> None:
    module = _load_module()
    factor_table = pd.DataFrame(
        {
            "alpha001_101": [1, 2, 3, 4, 5, 6],
            "ind_code": ["A", "A", "B", "B", "C", "C"],
            "cir_a": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-02"), "AAA"),
                (pd.Timestamp("2020-01-02"), "BBB"),
                (pd.Timestamp("2020-01-02"), "CCC"),
                (pd.Timestamp("2020-01-02"), "DDD"),
                (pd.Timestamp("2020-01-02"), "EEE"),
                (pd.Timestamp("2020-01-02"), "FFF"),
            ],
            names=["date", "symbol"],
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        processed = module.preprocess_factor_table(
            factor_table,
            ("alpha001_101",),
            enable_neutralize=True,
            min_aux_coverage=1.0,
        )

    assert processed["alpha001_101"].dtype == "float64"
    assert not [
        warning
        for warning in caught
        if "incompatible dtype" in str(warning.message)
    ]


def test_load_local_ohlcv_drops_blank_codes(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "START", "2020-01-01")
    monkeypatch.setattr(module, "END", "2020-01-31")
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "\n".join(
            [
                "date,code,open,high,low,close,volume",
                "2020-01-02,sh.600000,1,2,1,2,100",
                "2020-01-02,,1,2,1,2,100",
                "2020-01-03,   ,1,2,1,2,100",
            ]
        ),
        encoding="utf-8",
    )

    bars = module.load_local_ohlcv(csv_path)

    assert bars["symbol"].tolist() == ["sh.600000"]


def test_load_local_ohlcv_rejects_conflicting_duplicate_rows(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "START", "2020-01-01")
    monkeypatch.setattr(module, "END", "2020-01-31")
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "\n".join(
            [
                "date,code,open,high,low,close,volume",
                "2020-01-02,sh.600000,1,2,1,2,100",
                "2020-01-02,sh.600000,1,2,1,3,100",
            ]
        ),
        encoding="utf-8",
    )

    try:
        module.load_local_ohlcv(csv_path)
    except ValueError as exc:
        assert "conflicting duplicated timestamp/symbol rows" in str(exc)
    else:
        raise AssertionError("expected conflicting duplicate rows to raise ValueError")
