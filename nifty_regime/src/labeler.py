import numpy as np
import pandas as pd


def label_regimes(df: pd.DataFrame, horizon: int = 5,
                  trend_th: float = 0.025, range_th: float = 0.02,
                  vol_q: float = 0.75, quiet_q: float = 0.25) -> pd.Series:
    c = df["close"]
    fwd_ret = c.shift(-horizon) / c - 1
    fwd_vol = fwd_ret.rolling(horizon).std()

    vol_hi = fwd_vol.expanding().quantile(vol_q)
    vol_lo = fwd_vol.expanding().quantile(quiet_q)

    labels = pd.Series(index=df.index, data=2, dtype="Int64")
    labels[fwd_ret > trend_th] = 0
    labels[fwd_ret < -trend_th] = 1
    labels[(fwd_vol > vol_hi) & (fwd_ret.abs() > range_th)] = 3
    labels[(fwd_vol < vol_lo) & (fwd_ret.abs() < range_th)] = 4
    return labels
