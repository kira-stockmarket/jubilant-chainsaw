import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_nifty(start="2007-01-01", end=None, interval="1d", force=False):
    """
    Load Nifty 50 index data. Uses ^NSEI ticker.
    Caches locally so re-runs are fast.
    """
    cache = DATA_DIR / f"nifty_{interval}.parquet"

    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        print(f"[loader] Loaded from cache: {len(df)} rows")
        return df

    print("[loader] Downloading Nifty 50 data from Yahoo Finance...")
    df = yf.download(
        "^NSEI",
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower).dropna()
    df.index.name = "date"
    df.to_parquet(cache)
    print(f"[loader] Saved: {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")
    return df

if __name__ == "__main__":
    df = load_nifty(force=True)
    print(df.tail())
    