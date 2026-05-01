"""Tests for Tongdaxin-style indicator wrappers."""

import numpy as np
import pandas as pd

from tradelearn import ta
from tradelearn.indicators.tdx.mytt_adapter import MyTT

RTOL = 1e-10


def test_tdx_ma_and_ema_match_mytt_and_preserve_index() -> None:
    """ta.tdx moving averages match the vendored MyTT formulas."""
    close = _close()

    ma = ta.tdx.ma(close, n=5)
    ema = ta.tdx.EMA(close, N=6)

    np.testing.assert_allclose(
        ma.to_numpy(),
        MyTT.MA(close.to_numpy(), 5),
        rtol=RTOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        ema.to_numpy(),
        MyTT.EMA(close.to_numpy(), 6),
        rtol=RTOL,
        equal_nan=True,
    )
    pd.testing.assert_index_equal(ma.index, close.index)
    pd.testing.assert_index_equal(ema.index, close.index)
    assert ma.name == "MA_5"
    assert ema.name == "EMA_6"


def test_tdx_macd_matches_mytt_with_stable_columns() -> None:
    """ta.tdx.macd matches MyTT MACD output and returns named columns."""
    close = _close()

    result = ta.tdx.macd(close, short=4, long=9, m=3)
    expected = MyTT.MACD(close.to_numpy(), SHORT=4, LONG=9, M=3)

    assert list(result.columns) == ["DIF", "DEA", "MACD"]
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, rtol=RTOL, equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)


def test_tdx_kdj_matches_mytt_with_stable_columns() -> None:
    """ta.tdx.kdj matches MyTT KDJ output and returns named columns."""
    close = _close()
    high = _high()
    low = _low()

    result = ta.tdx.kdj(close, high, low, n=5, m1=3, m2=3)
    expected = MyTT.KDJ(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5, M1=3, M2=3)

    assert list(result.columns) == ["K", "D", "J"]
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, rtol=RTOL, equal_nan=True)
    pd.testing.assert_index_equal(result.index, close.index)


def test_tdx_rsi_matches_mytt() -> None:
    """ta.tdx.rsi matches MyTT RSI."""
    close = _close()

    result = ta.tdx.rsi(close, n=6)

    np.testing.assert_allclose(
        result.to_numpy(),
        MyTT.RSI(close.to_numpy(), N=6),
        rtol=RTOL,
        equal_nan=True,
    )
    pd.testing.assert_index_equal(result.index, close.index)
    assert result.name == "RSI_6"


def test_tdx30_alias_is_removed() -> None:
    assert not hasattr(ta, "tdx30")
    assert not hasattr(ta.tdx, "tdx30")


def test_tdx_wr_bias_boll_psy_match_mytt_with_stable_columns() -> None:
    """Multi-column tdx oscillators match MyTT formulas."""
    close = _close()
    high = _high()
    low = _low()

    wr = ta.tdx.wr(close, high, low, n=5, n1=3)
    bias = ta.tdx.bias(close, l1=3, l2=5, l3=7)
    boll = ta.tdx.boll(close, n=5, p=2)
    psy = ta.tdx.psy(close, n=5, m=3)

    _assert_frame_matches(
        wr,
        ["WR", "WR1"],
        MyTT.WR(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5, N1=3),
    )
    _assert_frame_matches(
        bias,
        ["BIAS1", "BIAS2", "BIAS3"],
        MyTT.BIAS(close.to_numpy(), L1=3, L2=5, L3=7),
    )
    _assert_frame_matches(boll, ["UPPER", "MID", "LOWER"], MyTT.BOLL(close.to_numpy(), N=5, P=2))
    _assert_frame_matches(psy, ["PSY", "PSYMA"], MyTT.PSY(close.to_numpy(), N=5, M=3))


def test_tdx_cci_atr_expma_obv_match_mytt() -> None:
    """Single and two-column tdx indicators match MyTT formulas."""
    close = _close()
    high = _high()
    low = _low()
    volume = _volume()

    cci = ta.tdx.cci(close, high, low, n=5)
    atr = ta.tdx.atr(close, high, low, n=5)
    expma = ta.tdx.expma(close, n1=4, n2=8)
    obv = ta.tdx.obv(close, volume)

    np.testing.assert_allclose(
        cci.to_numpy(),
        MyTT.CCI(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5),
        rtol=RTOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        atr.to_numpy(),
        MyTT.ATR(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5),
        rtol=RTOL,
        equal_nan=True,
    )
    _assert_frame_matches(expma, ["EXPMA1", "EXPMA2"], MyTT.EXPMA(close.to_numpy(), N1=4, N2=8))
    np.testing.assert_allclose(
        obv.to_numpy(),
        MyTT.OBV(close.to_numpy(), volume.to_numpy()),
        rtol=RTOL,
        equal_nan=True,
    )
    pd.testing.assert_index_equal(cci.index, close.index)
    pd.testing.assert_index_equal(atr.index, close.index)
    pd.testing.assert_index_equal(obv.index, close.index)
    assert cci.name == "CCI_5"
    assert atr.name == "ATR_5"
    assert obv.name == "OBV"


def test_tdx_wma_bbi_vr_mfi_match_mytt() -> None:
    """Additional classic tdx indicators match MyTT formulas."""
    close = _close()
    high = _high()
    low = _low()
    volume = _volume()

    wma = ta.tdx.wma(close, n=4)
    bbi = ta.tdx.bbi(close, m1=3, m2=4, m3=5, m4=6)
    vr = ta.tdx.vr(close, volume, m1=5)
    mfi = ta.tdx.mfi(close, high, low, volume, n=5)

    np.testing.assert_allclose(
        wma.to_numpy(),
        MyTT.WMA(close.to_numpy(), 4),
        rtol=RTOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bbi.to_numpy(),
        MyTT.BBI(close.to_numpy(), M1=3, M2=4, M3=5, M4=6),
        rtol=RTOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        vr.to_numpy(),
        MyTT.VR(close.to_numpy(), volume.to_numpy(), M1=5),
        rtol=RTOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        mfi.to_numpy(),
        MyTT.MFI(close.to_numpy(), high.to_numpy(), low.to_numpy(), volume.to_numpy(), N=5),
        rtol=RTOL,
        equal_nan=True,
    )
    assert wma.name == "WMA_4"
    assert bbi.name == "BBI"
    assert vr.name == "VR_5"
    assert mfi.name == "MFI_5"


def test_tdx_dmi_trix_mtm_roc_match_mytt_with_stable_columns() -> None:
    """Trend and momentum tdx indicators match MyTT tuple outputs."""
    close = _close()
    high = _high()
    low = _low()

    dmi = ta.tdx.dmi(close, high, low, m1=5, m2=3)
    trix = ta.tdx.trix(close, m1=4, m2=3)
    mtm = ta.tdx.mtm(close, n=4, m=3)
    roc = ta.tdx.roc(close, n=4, m=3)

    _assert_frame_matches(
        dmi,
        ["PDI", "MDI", "ADX", "ADXR"],
        MyTT.DMI(close.to_numpy(), high.to_numpy(), low.to_numpy(), M1=5, M2=3),
    )
    _assert_frame_matches(trix, ["TRIX", "TRMA"], MyTT.TRIX(close.to_numpy(), M1=4, M2=3))
    _assert_frame_matches(mtm, ["MTM", "MTMMA"], MyTT.MTM(close.to_numpy(), N=4, M=3))
    _assert_frame_matches(roc, ["ROC", "MAROC"], MyTT.ROC(close.to_numpy(), N=4, M=3))


def test_tdx_channel_and_momentum_extensions_match_mytt() -> None:
    """Channel and momentum extension indicators match MyTT tuple outputs."""
    close = _close()
    high = _high()
    low = _low()
    volume = _volume()

    taq = ta.tdx.taq(high, low, n=5)
    ktn = ta.tdx.ktn(close, high, low, n=5, m=4)
    cr = ta.tdx.cr(close, high, low, n=5)
    emv = ta.tdx.emv(high, low, volume, n=5, m=3)
    dpo = ta.tdx.dpo(close, m1=5, m2=3, m3=2)
    dfma = ta.tdx.dfma(close, n1=3, n2=6, m=3)
    mass = ta.tdx.mass(high, low, n1=3, n2=5, m=3)

    _assert_frame_matches(
        taq,
        ["UP", "MID", "DOWN"],
        MyTT.TAQ(high.to_numpy(), low.to_numpy(), N=5),
    )
    _assert_frame_matches(
        ktn,
        ["UPPER", "MID", "LOWER"],
        MyTT.KTN(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5, M=4),
    )
    np.testing.assert_allclose(
        cr.to_numpy(),
        MyTT.CR(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=5),
        rtol=RTOL,
        equal_nan=True,
    )
    _assert_frame_matches(
        emv,
        ["EMV", "MAEMV"],
        MyTT.EMV(high.to_numpy(), low.to_numpy(), volume.to_numpy(), N=5, M=3),
    )
    _assert_frame_matches(dpo, ["DPO", "MADPO"], MyTT.DPO(close.to_numpy(), M1=5, M2=3, M3=2))
    _assert_frame_matches(dfma, ["DIF", "DIFMA"], MyTT.DFMA(close.to_numpy(), N1=3, N2=6, M=3))
    _assert_frame_matches(
        mass,
        ["MASS", "MA_MASS"],
        MyTT.MASS(high.to_numpy(), low.to_numpy(), N1=3, N2=5, M=3),
    )
    assert cr.name == "CR_5"


def test_tdx_asi_brar_xsii_match_mytt() -> None:
    """Open-price tdx indicators match MyTT formulas."""
    open_ = _open()
    close = _close()
    high = _high()
    low = _low()

    brar = ta.tdx.brar(open_, close, high, low, m1=5)
    asi = ta.tdx.asi(open_, close, high, low, m1=5, m2=3)
    xsii = ta.tdx.xsii(close, high, low, n=90, m=5)

    _assert_frame_matches(
        brar,
        ["AR", "BR"],
        MyTT.BRAR(open_.to_numpy(), close.to_numpy(), high.to_numpy(), low.to_numpy(), M1=5),
    )
    _assert_frame_matches(
        asi,
        ["ASI", "ASIT"],
        MyTT.ASI(open_.to_numpy(), close.to_numpy(), high.to_numpy(), low.to_numpy(), M1=5, M2=3),
    )
    _assert_frame_matches(
        xsii,
        ["TD1", "TD2", "TD3", "TD4"],
        MyTT.XSII(close.to_numpy(), high.to_numpy(), low.to_numpy(), N=90, M=5),
    )


def _assert_frame_matches(
    result: pd.DataFrame,
    columns: list[str],
    expected: tuple[np.ndarray, ...],
) -> None:
    """Assert a result frame matches MyTT tuple output."""
    assert list(result.columns) == columns
    for column, values in zip(result.columns, expected, strict=True):
        np.testing.assert_allclose(result[column].to_numpy(), values, rtol=RTOL, equal_nan=True)
    pd.testing.assert_index_equal(result.index, _close().index)


def _close() -> pd.Series:
    """Return deterministic close prices with a date index."""
    return pd.Series(
        [
            10.0,
            10.5,
            10.2,
            11.0,
            11.5,
            11.2,
            12.0,
            12.4,
            12.1,
            12.8,
            13.2,
            13.0,
        ],
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
        name="close",
    )


def _open() -> pd.Series:
    """Return deterministic open prices."""
    close = _close()
    return pd.Series(close.to_numpy() - 0.1, index=close.index, name="open")


def _high() -> pd.Series:
    """Return deterministic high prices."""
    close = _close()
    return pd.Series(close.to_numpy() + 0.4, index=close.index, name="high")


def _low() -> pd.Series:
    """Return deterministic low prices."""
    close = _close()
    return pd.Series(close.to_numpy() - 0.3, index=close.index, name="low")


def _volume() -> pd.Series:
    """Return deterministic volume values."""
    close = _close()
    return pd.Series(
        [1000, 1200, 900, 1500, 1800, 1100, 1700, 1600, 1300, 1900, 2100, 1400],
        index=close.index,
        name="volume",
    )
