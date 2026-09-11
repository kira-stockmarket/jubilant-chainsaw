import time
import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
def _ensure_parquet_engine():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for parquet caching. "
            "Add `pyarrow` to requirements.txt. Original error: " + str(e)
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, lowercase, ensure tz-naive DatetimeIndex, sort."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df[df.index.notna()]
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[loader] missing columns after download: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[REQUIRED_COLS].dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Download with retry
# ---------------------------------------------------------------------------
def _download_yahoo_with_retry(start, end, interval, max_attempts=3):
    """
    Retry yfinance download with exponential backoff.
    Handles:
      - empty DataFrames from silent failures
      - transient network errors
      - Yahoo rate-limits on shared CI runner IPs
    Returns a DataFrame or None.
    """
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[loader] yahoo attempt {attempt}/{max_attempts} ...")
            df = yf.download(
                "^NSEI",
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,     # avoids a known race condition on CI
            )

            if df is not None and len(df) > 0:
                print(f"[loader] yahoo OK: {len(df)} raw rows")
                return df

            print(f"[loader] attempt {attempt}: empty result")

        except Exception as e:
            last_err = e
            print(f"[loader] attempt {attempt} raised: {type(e).__name__}: {e}")

        # Backoff: 3s, 6s, 9s
        if attempt < max_attempts:
            wait = 3 * attempt
            print(f"[loader] sleeping {wait}s before retry ...")
            time.sleep(wait)

    print(f"[loader] yahoo exhausted retries. last_err={last_err}")
    return None


# ---------------------------------------------------------------------------
# Stooq fallback
# ---------------------------------------------------------------------------
def _download_stooq_fallback(start, end, interval="d"):
    """
    Stooq mirrors daily Nifty data. Used only if Yahoo gives nothing.
    Note: Stooq only serves daily; interval is ignored.
    """
    url = "https://stooq.com/q/d/l/?s=^nsei&i=d"
    try:
        print("[loader] stooq fallback: downloading ...")
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            print(f"[loader] stooq unexpected columns: {list(df.columns)}")
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].set_index("date").sort_index()

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        if len(df) == 0:
            return None

        print(f"[loader] stooq OK: {len(df)} rows")
        return df[REQUIRED_COLS]
    except Exception as e:
        print(f"[loader] stooq fallback failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_nifty(start="2007-01-01", end=None, interval="1d", force=False):
    _ensure_parquet_engine()

    # 1. Cache
    if not force:
        cached = _read_cache(interval)
        if cached is not None and len(cached) > 0:
            print(f"[loader] cache hit: {len(cached)} rows")
            return cached

    # 2. Yahoo (retry x3)
    raw = _download_yahoo_with_retry(start, end, interval, max_attempts=3)

    # 3. Stooq fallback
    if raw is None or len(raw) == 0:
        print("[loader] yahoo failed -> switching to stooq fallback")
        raw = _download_stooq_fallback(start, end, interval)

    # 4. Hard fail
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "[loader] all data sources failed. "
            "Check the debug workflow or retry in a few minutes."
        )

    # 5. Normalize + sanity check
    df = _normalize(raw)
    if len(df) == 0:
        raise RuntimeError("[loader] data empty after normalization.")

    # 6. Persist
    _write_cache(df, interval)

    first = df.index.min()
    last = df.index.max()
    print(f"[loader] saved {len(df)} rows ({first.date()} -> {last.date()})")
    return df


if __name__ == "__main__":
    print(load_nifty(force=True).tail())
