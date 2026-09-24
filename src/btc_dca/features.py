from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_18", "ret_42",
    "vol_6", "vol_18", "vol_42", "atr14_pct",
    "ema20_gap", "ema50_gap", "ema200_gap", "ema_slope_6",
    "adx14", "rsi14", "macd_hist_pct",
    "volume_z42", "volume_change_6", "obv_slope_18",
]


def aggregate_4h(ohlcv_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate UTC 5m bars into complete 4h bars labelled at close time."""
    grouped = ohlcv_5m.resample("4h", closed="left", label="right", origin="start_day")
    result = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), bars=("close", "count"),
    )
    return result.loc[result["bars"] == 48, ["open", "high", "low", "close", "volume"]].copy()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smooth = lambda series: series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    atr = smooth(true_range)
    plus_di = 100 * smooth(plus_dm) / atr.replace(0, np.nan)
    minus_di = 100 * smooth(minus_dm) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return smooth(dx)


def build_features_and_label(
    bars_4h: pd.DataFrame,
    horizon_bars: int = 42,
    unsafe_drawdown: float = -0.12,
) -> pd.DataFrame:
    """Create OHLCV-only features and a strictly forward-looking unsafe label."""
    data = bars_4h.copy()
    close = data["close"].astype(float)
    ret = close.pct_change(fill_method=None)
    for n, name in [(1, "ret_1"), (3, "ret_3"), (6, "ret_6"), (18, "ret_18"), (42, "ret_42")]:
        data[name] = close.pct_change(n, fill_method=None)
    for n, name in [(6, "vol_6"), (18, "vol_18"), (42, "vol_42")]:
        data[name] = ret.rolling(n, min_periods=n).std(ddof=1)

    previous_close = close.shift(1)
    tr = pd.concat(
        [data["high"] - data["low"], (data["high"] - previous_close).abs(),
         (data["low"] - previous_close).abs()], axis=1,
    ).max(axis=1)
    data["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close

    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    data["ema20_gap"] = ema20 / ema50 - 1
    data["ema50_gap"] = ema50 / ema200 - 1
    data["ema200_gap"] = ema200 / close - 1
    data["ema_slope_6"] = ema20.pct_change(6, fill_method=None)
    data["adx14"] = _adx(data["high"], data["low"], close)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi14"] = 100 - 100 / (1 + rs)
    ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    data["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / close

    volume = data["volume"].astype(float)
    volume_mean = volume.rolling(42, min_periods=42).mean()
    volume_std = volume.rolling(42, min_periods=42).std(ddof=1)
    data["volume_z42"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    data["volume_change_6"] = volume.pct_change(6, fill_method=None)
    signed_volume = np.sign(close.diff()).fillna(0) * volume
    obv = signed_volume.cumsum()
    data["obv_slope_18"] = obv.diff(18) / volume.rolling(18, min_periods=18).sum().replace(0, np.nan)

    future_low = data["low"].shift(-1)[::-1].rolling(horizon_bars, min_periods=horizon_bars).min()[::-1]
    data["future_min_drawdown"] = future_low / close - 1
    data["unsafe"] = (data["future_min_drawdown"] <= unsafe_drawdown).astype("float64")
    data.loc[data["future_min_drawdown"].isna(), "unsafe"] = np.nan
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return data
