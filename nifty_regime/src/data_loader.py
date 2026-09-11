import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _ensure_parquet_engine():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for parquet caching. "
            "Add `pyarrow` to requirements.txt. Original error: " + str(e)
        )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, lowercase, ensure datetime index, sort."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)

    # Ensure datetime index — critical fix for the .date() crash
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df[df.index.notna()]
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    # Keep only OHLCV
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[loader] missing columns after download: {missing}. "
                         f"Got: {list(df.columns)}")

    df = df[REQUIRED_COLS].dropna()
    df.index.name = "date"
    return df


def _write_cache(df: pd.DataFrame, interval: str):
    parquet_path = DATA_DIR / f"nifty_{interval}.parquet"
    df.to_parquet(parquet_path)
    print(f"[loader] cached parquet -> {parquet_path.name}")


def _read_cache(interval: str):
    parquet_path = DATA_DIR / f"nifty_{interval}.parquet"
    csv_path = DATA_DIR / f"nifty_{interval}.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[loader] parquet read failed ({e}); trying CSV")

    if csv_path.exists():
        return pd.read_csv(csv_path, index_col="date", parse_dates=True)

    return None


def load_nifty(start="2007-01-01", end=None, interval="1d", force=False):
    _ensure_parquet_engine()

    if not force:
        cached = _read_cache(interval)
        if cached is not None and len(cached) > 0:
            print(f"[loader] cache hit: {len(cached)} rows")
            return cached

    print(f"[loader] downloading ^NSEI (interval={interval}) ...")
    df = yf.download(
        "^NSEI",
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,          # avoids occasional thread race on runners
    )

    if df is None or len(df) == 0:
        raise RuntimeError(
            "[loader] yfinance returned empty data. Possible causes: "
            "network, rate limit, or ^NSEI temporarily unavailable. "
            "Retry in a few minutes or run the debug workflow."
        )

    df = _normalize(df)

    if len(df) == 0:
        raise RuntimeError("[loader] data empty after normalization.")

    _write_cache(df, interval)

    first = df.index.min()
    last = df.index.max()
    print(f"[loader] saved {len(df)} rows "
          f"({first.date()} -> {last.date()})")
    return df


if __name__ == "__main__":
    print(load_nifty(force=True).tail())
