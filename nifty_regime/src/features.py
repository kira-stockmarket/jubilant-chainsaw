import numpy as np
import pandas as pd


def _atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _efficiency_ratio(close, n=20):
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return change / volatility.replace(0, np.nan)


def _choppiness(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    rng = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(atr_sum / rng.replace(0, np.nan)) / np.log10(n)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    f["ret_1"] = c.pct_change(1)
    f["ret_3"] = c.pct_change(3)
    f["ret_5"] = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)

    f["vol_5"] = f["ret_1"].rolling(5).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"].replace(0, np.nan)

    f["atr_14"] = _atr(df, 14)
    f["atr_pct"] = f["atr_14"] / c
    f["hl_range"] = (h - l) / c

    f["er_10"] = _efficiency_ratio(c, 10)
    f["er_20"] = _efficiency_ratio(c, 20)
    f["chop_14"] = _choppiness(df, 14)

    f["slope_5"] = c.rolling(5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean(), raw=True)
    f["slope_10"] = c.rolling(10).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean(), raw=True)

    roll_hi = h.rolling(20).max()
    roll_lo = l.rolling(20).min()
    f["range_pos_20"] = (c - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)

    f["vol_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    f["gap"] = (df["open"] - c.shift(1)) / c.shift(1)

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = f["atr_14"].replace(0, np.nan)
    f["di_plus"] = pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr14
    f["di_minus"] = pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr14
    f["di_diff"] = f["di_plus"] - f["di_minus"]

    sign = np.sign(c.diff()).fillna(0)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    f["streak"] = np.where(sign >= 0, streak, -streak)

    return f.replace([np.inf, -np.inf], np.nan)
