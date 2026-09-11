import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_nifty(start="2007-01-01", end=None, interval="1d", force=False):
    cache = DATA_DIR / f"nifty_{interval}.parquet"

    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        print(f"[loader] cache hit: {len(df)} rows")
        return df

    print("[loader] downloading ^NSEI ...")
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
    print(f"[loader] saved {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")
    return df


if __name__ == "__main__":
    print(load_nifty(force=True).tail())
