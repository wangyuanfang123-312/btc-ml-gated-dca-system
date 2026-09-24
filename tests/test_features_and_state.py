import numpy as np
import pandas as pd

from btc_dca.features import aggregate_4h, build_features_and_label
from btc_dca.walkforward import hysteresis


def test_aggregate_uses_only_complete_48_bar_utc_windows():
    idx = pd.date_range("2024-01-01", periods=49, freq="5min", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": np.arange(100.0, 149.0), "volume": 1.0}, index=idx)
    result = aggregate_4h(bars)
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-01-01 04:00", tz="UTC")
    assert result.iloc[0]["close"] == 147.0


def test_unsafe_label_uses_only_future_lows():
    idx = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    close = np.full(8, 100.0)
    low = np.array([99, 98, 90, 99, 99, 99, 99, 99], dtype=float)
    bars = pd.DataFrame({"open": close, "high": close, "low": low,
                         "close": close, "volume": np.arange(1, 9)}, index=idx)
    data = build_features_and_label(bars, horizon_bars=2, unsafe_drawdown=-0.08)
    assert data.loc[idx[0], "unsafe"] == 1
    assert data.loc[idx[2], "unsafe"] == 0
    assert np.isnan(data.loc[idx[-1], "unsafe"])


def test_hysteresis_requires_two_low_bars_and_one_high_bar_turns_off():
    idx = pd.date_range("2024-01-01", periods=7, freq="4h", tz="UTC")
    probs = pd.Series([0.10, 0.20, 0.24, 0.30, 0.40, 0.15, 0.15], index=idx)
    state = hysteresis(probs)
    assert state.tolist() == [False, True, True, True, False, False, True]

