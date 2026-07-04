import ast
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest


PIPELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "zoo"
    / "backtest"
    / "tushare_sw_hs300"
    / "hs300_pipeline_tushare_sw.py"
)

pytestmark = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="ignored HS300 pipeline example is not present",
)


def _load_function(name):
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert funcs, f"{name} is not defined"
    module = ast.Module(body=funcs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"os": os}
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace[name]


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("hs300_pipeline_tushare_sw", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tushare_code_maps_baostock_code_to_tushare_suffix():
    to_tushare_code = _load_function("_to_tushare_code")

    assert to_tushare_code("sh.600000") == "600000.SH"
    assert to_tushare_code("sz.000001") == "000001.SZ"


def test_tushare_code_leaves_existing_tushare_suffix_unchanged():
    to_tushare_code = _load_function("_to_tushare_code")

    assert to_tushare_code("600000.SH") == "600000.SH"
    assert to_tushare_code("000001.SZ") == "000001.SZ"


def test_tushare_token_comes_from_environment(monkeypatch):
    get_tushare_token = _load_function("_get_tushare_token")

    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    assert get_tushare_token() == "test-token"


def test_base_dir_is_script_relative():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "SCRIPT_DIR" in source
    assert 'DATA_DIR = os.path.join(SCRIPT_DIR, "data")' in source
    assert 'BASE_DIR = os.path.join(DATA_DIR, f"hs300_{PERIOD_TAG}")' in source


def test_skip_can_be_overridden_for_specific_step(monkeypatch, tmp_path):
    skip = _load_function("_skip")
    existing = tmp_path / "existing.csv"
    existing.write_text("already built", encoding="utf-8")
    monkeypatch.setenv("HS300_FORCE_STEPS", "STEP 10")

    assert skip(str(existing), "STEP 10") is False


def test_csi_audit_is_disabled_by_default(monkeypatch):
    csi_audit_enabled = _load_function("_csi_audit_enabled")

    monkeypatch.delenv("HS300_ENABLE_CSI_AUDIT", raising=False)

    assert csi_audit_enabled() is False


def test_csi_audit_can_be_enabled_by_environment(monkeypatch):
    csi_audit_enabled = _load_function("_csi_audit_enabled")

    monkeypatch.setenv("HS300_ENABLE_CSI_AUDIT", "1")

    assert csi_audit_enabled() is True


def test_share_capital_fields_are_requested_and_merged():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'fields="trade_date,total_share,float_share,free_share"' in source
    assert "total_share" in source
    assert "free_share" in source
    assert "total_mv_calc" in source
    assert "free_mv_calc" in source


def test_pipeline_uses_tushare_for_core_data_sources():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "pro.trade_cal(" in source
    assert "pro.index_weight(" in source
    assert "pro.stock_basic(" in source
    assert "pro.daily(" in source
    assert "pro.daily_basic(" in source


def test_pipeline_does_not_use_akshare_baostock_or_excel_inputs():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "import akshare" not in source
    assert "import baostock" not in source
    assert "query_stock_basic" not in source
    assert "query_history_k_data_plus" not in source


def test_pipeline_adds_adjustment_and_auxiliary_market_apis():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "pro.adj_factor(" in source
    assert "qfq_close" in source
    assert "pro.suspend_d(" in source
    assert "pro.stk_limit(" in source
    assert "pro.index_daily(" in source


def test_pipeline_keeps_excluded_refactors_out():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert ".meta.json" not in source
    assert "manifest" not in source.lower()
    assert "for trade_date in" not in source


def test_sw_component_fetch_validates_history_fields_and_falls_back_to_all_members():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'required_history_cols = {"con_code", "in_date", "out_date"}' in source
    assert "_has_history_component_fields(res)" in source
    assert 'for is_new in ["Y", "N"]' in source
    assert "pro.index_member_all(l1_code=raw_code, is_new=is_new)" in source
    assert 'res["con_code"] = res["ts_code"]' in source


def test_low_risk_pipeline_optimizations_are_present():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'sort_values("date")' in source
    assert "force_step7b" in source
    assert "min_industry_coverage = 100" in source
    assert "all_data_parts = []" in source
    assert "date_stock_parts = []" in source
    assert "res_parts = []" in source
    assert "rqdatac" not in source


def test_step9_uses_precomputed_lookup_maps_instead_of_large_dataframe_queries():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "sb_by_code =" in source
    assert "skd_by_key =" in source
    assert "skd_by_code =" in source
    assert "namechange_by_code =" in source
    assert "suspend_keys =" in source
    assert "skd_by_key.get((code, date_str))" in source
    assert "skd.query(f\"code == '{code}' and date == '{date_str}'\")" not in source


def test_step9_keeps_non_tradable_valuation_rows_for_missing_bars():
    pipeline = _load_pipeline_module()
    code = "600001.SH"
    history = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "code": [code],
            "open": [9.8],
            "high": [10.5],
            "low": [9.7],
            "close": [10.0],
            "preclose": [9.9],
            "volume": [10000.0],
            "amount": [100000.0],
            "tradestatus": [1],
            "pctChg": [1.0],
        }
    )

    row = pipeline._frozen_valuation_row(code, "2020-01-03", {code: history})

    assert row is not None
    frozen = row.iloc[0]
    assert frozen["date"] == "2020-01-03"
    assert frozen["open"] == 10.0
    assert frozen["high"] == 10.0
    assert frozen["low"] == 10.0
    assert frozen["close"] == 10.0
    assert frozen["volume"] == 0.0
    assert frozen["amount"] == 0.0
    assert frozen["tradestatus"] == 0


def test_step9_missing_bar_is_valued_but_not_tradable():
    pipeline = _load_pipeline_module()
    code = "600001.SH"
    date = pd.Timestamp("2020-01-03")
    history = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "code": [code],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [10000.0],
            "amount": [100000.0],
            "tradestatus": [1],
        }
    )
    args = (
        {code},
        date,
        "2020-01-03",
        {code: {"list_date": "20190101", "delist_date": ""}},
        {},
        {code: history},
        {},
        set(),
    )

    date_str, valid_set, result = pipeline._filter_single_date(args)

    assert date_str == "2020-01-03"
    assert valid_set == set()
    assert len(result) == 1
    assert result.iloc[0]["code"] == code
    assert result.iloc[0]["close"] == 10.0
    assert result.iloc[0]["volume"] == 0.0
    assert result.iloc[0]["tradestatus"] == 0


def test_pipeline_outputs_drop_csv_index_columns():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert (
        'raw_data = raw_data.loc[:, ~raw_data.columns.astype(str).str.startswith("Unnamed:")]'
        in source
    )
    assert "index=False" in source


def test_step10_uses_close_as_vwap_for_zero_volume_valuation_rows():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'calculated_vwap.where(volume > 0, s_data["close"].astype(float))' in source


def test_pipeline_uses_dynamic_end_date_window_and_paths():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'BEGIN_DATE = os.environ.get("HS300_BEGIN_DATE", "2015-01-01")' in source
    assert "END_DATE_STR = os.environ.get(" in source
    assert '"HS300_END_DATE"' in source
    assert 'pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d")' in source
    assert "PERIOD_TAG = (" in source
    assert 'DATA_DIR = os.path.join(SCRIPT_DIR, "data")' in source
    assert 'BASE_DIR = os.path.join(DATA_DIR, f"hs300_{PERIOD_TAG}")' in source
    assert 'RAW_DIR = os.path.join(BASE_DIR, "raw")' in source
    assert 'PATH_FINAL = os.path.join(BASE_DIR, f"000300SH_{PERIOD_TAG}.csv")' in source
    assert 'PATH_RAW_FILTER = os.path.join(BASE_DIR, f"000300SH_raw_{PERIOD_TAG}.csv")' in source
    assert "2012_2024" not in source


def test_pipeline_daily_mode_paths_are_separate_from_full_period_paths():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert 'PIPELINE_MODE = os.environ.get("HS300_PIPELINE_MODE", "daily")' in source
    assert 'DAILY_DIR = os.path.join(DATA_DIR, "hs300_daily")' in source
    assert 'PATH_DAILY_FINAL = os.path.join(DAILY_DIR, f"000300SH_{DAILY_TAG}.csv")' in source
    assert "def main_daily()" in source
    assert "def main_full()" in source


def test_latest_full_data_path_ignores_daily_directory(tmp_path):
    pipeline = _load_pipeline_module()
    old_dir = tmp_path / "hs300_20220101_20240331"
    new_dir = tmp_path / "hs300_20220101_20240430"
    daily_dir = tmp_path / "hs300_daily"
    old_dir.mkdir()
    new_dir.mkdir()
    daily_dir.mkdir()
    old_file = old_dir / "000300SH_20220101_20240331.csv"
    new_file = new_dir / "000300SH_20220101_20240430.csv"
    daily_file = daily_dir / "000300SH_20240501.csv"
    old_file.write_text("date,code\n", encoding="utf-8")
    new_file.write_text("date,code\n", encoding="utf-8")
    daily_file.write_text("date,code\n", encoding="utf-8")

    assert pipeline._latest_full_data_path(str(tmp_path)) == str(new_file)


def test_load_full_plus_daily_dataset_appends_daily_files_and_deduplicates(tmp_path):
    pipeline = _load_pipeline_module()
    full_dir = tmp_path / "hs300_20220101_20240331"
    daily_dir = tmp_path / "hs300_daily"
    full_dir.mkdir()
    daily_dir.mkdir()
    full_file = full_dir / "000300SH_20220101_20240331.csv"
    full_file.write_text("date,code,close\n2024-03-31,AAA,1\n", encoding="utf-8")
    (daily_dir / "000300SH_20240401.csv").write_text(
        "date,code,close\n2024-04-01,AAA,2\n",
        encoding="utf-8",
    )
    (daily_dir / "000300SH_20240402.csv").write_text(
        "date,code,close\n2024-04-01,AAA,20\n2024-04-02,AAA,3\n",
        encoding="utf-8",
    )

    result = pipeline._load_full_plus_daily_dataset(str(tmp_path))

    assert result["date"].tolist() == ["2024-03-31", "2024-04-01", "2024-04-02"]
    assert result["close"].tolist() == [1, 20, 3]


def test_latest_available_trade_date_falls_back_to_previous_open_day_without_daily_data():
    pipeline = _load_pipeline_module()

    class FakePro:
        def trade_cal(self, **_kwargs):
            return pd.DataFrame(
                {
                    "cal_date": ["20240401", "20240402"],
                    "is_open": [1, 1],
                }
            )

        def daily(self, trade_date, **_kwargs):
            if trade_date == "20240402":
                return pd.DataFrame()
            return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date]})

    assert pipeline._latest_available_trade_date(
        FakePro(),
        "2024-04-02",
    ) == pd.Timestamp("2024-04-01")


def test_build_daily_final_dataset_overwrites_traded_rows_and_freezes_missing(monkeypatch):
    pipeline = _load_pipeline_module()
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_args, **_kwargs: None)

    class FakePro:
        def daily(self, **_kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["AAA.SZ"],
                    "trade_date": ["20240401"],
                    "open": [2.0],
                    "high": [2.1],
                    "low": [1.9],
                    "close": [2.0],
                    "pre_close": [1.0],
                    "change": [1.0],
                    "pct_chg": [100.0],
                    "vol": [3.0],
                    "amount": [4.0],
                }
            )

        def daily_basic(self, **_kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["AAA.SZ"],
                    "trade_date": ["20240401"],
                    "total_share": [100.0],
                    "float_share": [80.0],
                    "free_share": [60.0],
                    "turnover_rate": [1.5],
                    "pe_ttm": [10.0],
                    "ps_ttm": [2.0],
                    "pb": [1.0],
                }
            )

        def suspend_d(self, **_kwargs):
            return pd.DataFrame()

        def stk_limit(self, **_kwargs):
            return pd.DataFrame(
                {
                    "trade_date": ["20240401"],
                    "ts_code": ["AAA.SZ"],
                    "up_limit": [2.2],
                    "down_limit": [1.8],
                }
            )

        def index_daily(self, **_kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["000300.SH"],
                    "trade_date": ["20240401"],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.0],
                    "close": [10.5],
                    "pre_close": [10.0],
                    "change": [0.5],
                    "pct_chg": [5.0],
                    "vol": [1000.0],
                    "amount": [2000.0],
                }
            )

    base_data = pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-03-31"],
            "code": ["AAA.SZ", "BBB.SZ"],
            "open": [1.0, 5.0],
            "high": [1.0, 5.0],
            "low": [1.0, 5.0],
            "close": [1.0, 5.0],
            "preclose": [1.0, 5.0],
            "qfq_open": [1.0, 5.0],
            "qfq_high": [1.0, 5.0],
            "qfq_low": [1.0, 5.0],
            "qfq_close": [1.0, 5.0],
            "qfq_preclose": [1.0, 5.0],
            "volume": [100.0, 500.0],
            "amount": [1000.0, 5000.0],
            "tradestatus": [1, 1],
            "cir_a_num": [10.0, 50.0],
            "cir_a": [10.0, 250.0],
            "total_share": [10.0, 50.0],
            "total_mv_calc": [10.0, 250.0],
            "free_share": [10.0, 50.0],
            "free_mv_calc": [10.0, 250.0],
            "turn": [1.0, 1.0],
            "peTTM": [1.0, 1.0],
            "psTTM": [1.0, 1.0],
            "pbMRQ": [1.0, 1.0],
            "up_limit": [1.1, 5.5],
            "down_limit": [0.9, 4.5],
            "index_code": ["000300.SH", "000300.SH"],
            "index_open": [8.0, 8.0],
            "index_high": [8.0, 8.0],
            "index_low": [8.0, 8.0],
            "index_close": [8.0, 8.0],
            "index_preclose": [8.0, 8.0],
            "index_change": [0.0, 0.0],
            "index_pctChg": [0.0, 0.0],
            "index_volume": [1.0, 1.0],
            "index_amount": [1.0, 1.0],
        }
    )

    result = pipeline._build_daily_final_dataset(
        base_data,
        FakePro(),
        pd.Timestamp("2024-04-01"),
    ).set_index("code")

    assert result.loc["AAA.SZ", "close"] == 2.0
    assert result.loc["AAA.SZ", "volume"] == 300.0
    assert result.loc["AAA.SZ", "cir_a"] == 160.0
    assert result.loc["AAA.SZ", "tradestatus"] == 1
    assert result.loc["BBB.SZ", "close"] == 5.0
    assert result.loc["BBB.SZ", "volume"] == 0.0
    assert result.loc["BBB.SZ", "tradestatus"] == 0
    assert result.loc["AAA.SZ", "index_close"] == 10.5
    assert result.loc["BBB.SZ", "index_close"] == 10.5


def test_step2_csi_audit_is_optional_diagnostic():
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert "HS300_ENABLE_CSI_AUDIT" in source
    assert "if _csi_audit_enabled():" in source
    assert "中证官网成分审计默认关闭" in source


def test_csi_changes_are_converted_to_component_intervals():
    pipeline = _load_pipeline_module()
    current = pd.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "start_date": [pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-01")],
            "end_date": [pd.Timestamp("2099-12-31"), pd.Timestamp("2099-12-31")],
        }
    )
    changes = pd.DataFrame(
        [
            {"symbol": "600002.SH", "date": pd.Timestamp("2020-01-02"), "type": "add"},
            {"symbol": "600003.SH", "date": pd.Timestamp("2020-01-01"), "type": "remove"},
        ]
    )

    intervals = pipeline._build_csi_component_intervals(
        current,
        changes,
        bench_start_date=pd.Timestamp("2005-01-01"),
    )

    row_600002 = intervals.loc[intervals["symbol"] == "600002.SH"].iloc[0]
    row_600003 = intervals.loc[intervals["symbol"] == "600003.SH"].iloc[0]
    assert row_600002["start_date"] == pd.Timestamp("2020-01-02")
    assert row_600002["end_date"] == pd.Timestamp("2099-12-31")
    assert row_600003["start_date"] == pd.Timestamp("2005-01-01")
    assert row_600003["end_date"] == pd.Timestamp("2020-01-01")


def test_csi_interval_builder_ignores_orphan_add_events():
    pipeline = _load_pipeline_module()
    current = pd.DataFrame(
        {
            "symbol": ["600001.SH"],
            "start_date": [pd.Timestamp("2005-01-01")],
            "end_date": [pd.Timestamp("2099-12-31")],
        }
    )
    changes = pd.DataFrame(
        [{"symbol": "600002.SH", "date": pd.Timestamp("2020-01-02"), "type": "add"}]
    )

    intervals = pipeline._build_csi_component_intervals(
        current,
        changes,
        bench_start_date=pd.Timestamp("2005-01-01"),
    )

    assert intervals["symbol"].tolist() == ["600001.SH"]


def test_component_audit_flags_snapshot_differences():
    pipeline = _load_pipeline_module()
    tushare_snapshots = pd.DataFrame(
        {
            "2020-01-02": ["600001.SH", "600002.SH"],
            "2020-01-03": ["600001.SH", "600003.SH"],
        }
    )
    csi_intervals = pd.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "start_date": [pd.Timestamp("2005-01-01"), pd.Timestamp("2020-01-02")],
            "end_date": [pd.Timestamp("2099-12-31"), pd.Timestamp("2099-12-31")],
        }
    )

    audit = pipeline._audit_component_snapshots(tushare_snapshots, csi_intervals)

    first = audit.loc[audit["trade_date"] == "2020-01-02"].iloc[0]
    second = audit.loc[audit["trade_date"] == "2020-01-03"].iloc[0]
    assert first["missing_in_tushare_count"] == 0
    assert first["extra_in_tushare_count"] == 0
    assert second["missing_in_tushare_count"] == 1
    assert second["extra_in_tushare_count"] == 1
    assert second["missing_in_tushare"] == "600002.SH"
    assert second["extra_in_tushare"] == "600003.SH"


def test_current_components_can_fall_back_to_latest_tushare_snapshot():
    pipeline = _load_pipeline_module()
    tushare_snapshots = pd.DataFrame(
        {
            "2020-01-02": ["600001.SH", "600002.SH", None],
            "2020-01-03": ["600001.SH", "600003.SH", "600003.SH"],
        }
    )

    current = pipeline._current_components_from_tushare_snapshots(tushare_snapshots)

    assert current["symbol"].tolist() == ["600001.SH", "600003.SH"]
    assert set(current.columns) == {"symbol", "start_date", "end_date"}
    assert current["end_date"].eq(pd.Timestamp("2099-12-31")).all()


def test_csi_current_component_fetch_rejects_future_snapshot(monkeypatch):
    pipeline = _load_pipeline_module()
    monkeypatch.setattr(pipeline, "END_DATE", pd.Timestamp("2026-06-01"))

    class Response:
        content = b"excel-bytes"

    monkeypatch.setattr(pipeline, "_retry_get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        pipeline.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame(
            {"日期Date": [20260626], "成份券代码Constituent Code": [1]}
        ),
    )

    try:
        pipeline._fetch_csi_current_components()
    except RuntimeError as exc:
        assert "晚于数据截止日" in str(exc)
    else:
        raise AssertionError("future CSI snapshot should be rejected")


def test_csi_attachment_download_uses_range_prewarm_and_cache(monkeypatch, tmp_path):
    pipeline = _load_pipeline_module()
    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

    def fake_retry_get(url, **kwargs):
        calls.append((url, kwargs))
        headers = kwargs.get("headers") or {}
        if headers.get("Range") == "bytes=0-65535":
            return Response(b"partial")
        return Response(b"full-excel-content")

    monkeypatch.setattr(pipeline, "CSI_ATTACHMENTS_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "_retry_get", fake_retry_get)

    first = pipeline._download_csi_attachment("https://example.com/file.xlsx")
    second = pipeline._download_csi_attachment("https://example.com/file.xlsx")

    assert first == b"full-excel-content"
    assert second == b"full-excel-content"
    assert len(calls) == 2
    assert calls[0][1]["headers"]["Range"] == "bytes=0-65535"
    assert calls[0][1]["timeout"] == (5, 10)
    assert calls[1][1]["timeout"] == (5, 30)
    assert len(list(tmp_path.iterdir())) == 1


def test_csi_attachment_errors_are_written_to_csv(monkeypatch, tmp_path):
    pipeline = _load_pipeline_module()
    errors = [{"notice_url": "n", "excel_url": "e", "error": "timeout"}]

    monkeypatch.setattr(pipeline, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "PATH_COMPONENT_AUDIT_ERRORS", str(tmp_path / "errors.csv"))

    pipeline._write_component_audit_errors(errors)

    written = pd.read_csv(tmp_path / "errors.csv")
    assert written.to_dict("records") == errors


def test_daily_increment_fetches_market_data_by_trade_date_once() -> None:
    pipeline = _load_pipeline_module()

    class FakeTushare:
        def __init__(self):
            self.calls = []

        def daily(self, **kwargs):
            self.calls.append(("daily", kwargs))
            return pd.DataFrame(
                {
                    "ts_code": ["600001.SH", "000001.SZ", "999999.SH"],
                    "trade_date": ["20260630", "20260630", "20260630"],
                    "open": [1.0, 2.0, 9.0],
                    "high": [1.1, 2.1, 9.1],
                    "low": [0.9, 1.9, 8.9],
                    "close": [1.0, 2.0, 9.0],
                    "pre_close": [0.9, 1.9, 8.9],
                    "change": [0.1, 0.1, 0.1],
                    "pct_chg": [1.0, 1.0, 1.0],
                    "vol": [10.0, 20.0, 90.0],
                    "amount": [100.0, 200.0, 900.0],
                }
            )

        def daily_basic(self, **kwargs):
            self.calls.append(("daily_basic", kwargs))
            return pd.DataFrame(
                {
                    "ts_code": ["600001.SH", "000001.SZ", "999999.SH"],
                    "trade_date": ["20260630", "20260630", "20260630"],
                    "total_share": [1000.0, 2000.0, 9000.0],
                    "float_share": [800.0, 1600.0, 8000.0],
                    "free_share": [700.0, 1500.0, 7000.0],
                    "turnover_rate": [1.0, 2.0, 9.0],
                    "pe_ttm": [10.0, 20.0, 90.0],
                    "ps_ttm": [1.0, 2.0, 9.0],
                    "pb": [1.0, 2.0, 9.0],
                }
            )

        def suspend_d(self, **kwargs):
            self.calls.append(("suspend_d", kwargs))
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20260630"],
                    "suspend_timing": ["全天"],
                    "suspend_type": ["S"],
                }
            )

        def stk_limit(self, **kwargs):
            self.calls.append(("stk_limit", kwargs))
            return pd.DataFrame(
                {
                    "trade_date": ["20260630", "20260630", "20260630"],
                    "ts_code": ["600001.SH", "000001.SZ", "999999.SH"],
                    "pre_close": [0.9, 1.9, 8.9],
                    "up_limit": [1.1, 2.1, 9.1],
                    "down_limit": [0.8, 1.8, 8.8],
                }
            )

    pro = FakeTushare()
    codes = ["600001.SH", "000001.SZ"]

    daily, basic, suspend, limit_data = pipeline._fetch_one_day_market_data(
        pro, codes, pd.Timestamp("2026-06-30")
    )

    assert [name for name, _ in pro.calls] == ["daily", "daily_basic", "suspend_d", "stk_limit"]
    for _, kwargs in pro.calls:
        assert kwargs["trade_date"] == "20260630"
        assert "ts_code" not in kwargs
        assert "start_date" not in kwargs
        assert "end_date" not in kwargs
    assert daily["ts_code"].tolist() == codes
    assert basic["ts_code"].tolist() == codes
    assert suspend["ts_code"].tolist() == ["000001.SZ"]
    assert limit_data["ts_code"].tolist() == codes
