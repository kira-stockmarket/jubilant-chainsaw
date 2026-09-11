import time
import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Dependency guards
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
            f"[loader] missing columns: {missing}. Got: {list(df.columns)}"
        )

    df = df[REQUIRED_COLS].dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _write_cache(df: pd.DataFrame, interval: str):
    path = DATA_DIR / f"nifty_{interval}.parquet"
    df.to_parquet(path)
    print(f"[loader] cached -> {path.name}")


def _read_cache(interval: str):
    p = DATA_DIR / f"nifty_{interval}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"[loader] parquet read failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Source 1: nselib  (primary — works from CI)
# ---------------------------------------------------------------------------
def _download_nselib(start, end):
    """
    nselib returns NSE index historical data.
    Symbol for Nifty 50 index is 'NIFTY 50'.
    """
    try:
        from nselib import capital_market
        print("[loader] nselib: requesting NIFTY 50 ...")

        # nselib expects from_date / to_date as 'DD-MM-YYYY'
        from_dt = pd.Timestamp(start).strftime("%d-%m-%Y")
        to_dt = pd.Timestamp(end).strftime("%d-%m-%Y") if end else pd.Timestamp.today().strftime("%d-%m-%Y")

        df = capital_market.index_data(
            index="NIFTY 50",
            from_date=from_dt,
            to_date=to_dt,
        )

        if df is None or len(df) == 0:
            print("[loader] nselib returned empty")
            return None

        df.columns = [c.strip().lower() for c in df.columns]
        # nselib columns typically: 'date', 'open', 'high', 'low', 'close', 'volume' (volume may be absent for index)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].notna()].set_index("date").sort_index()

        if "volume" not in df.columns:
            df["volume"] = 0

        print(f"[loader] nselib OK: {len(df)} rows")
        return df[REQUIRED_COLS]

    except Exception as e:
        print(f"[loader] nselib failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 2: NSE India direct CSV (backup)
# ---------------------------------------------------------------------------
def _download_nse_direct(start, end):
    """
    Direct NSE India historical index CSV endpoint.
    Uses the official NSE API used by their website — stable.
    """
    try:
        print("[loader] NSE direct CSV: requesting ...")
        url = (
            "https://www.nseindia.com/api/historical/indicesHistory"
            f"?indexType=NIFTY%2050&from={pd.Timestamp(start).strftime('%d-%m-%Y')}"
            f"&to={(pd.Timestamp(end) if end else pd.Timestamp.today()).strftime('%d-%m-%Y')}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=15)  # cookie warm-up
        r = s.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()

        rows = js.get("data", {}).get("indexCloseOnlineRecords") or \
               js.get("data", {}).get("indexPortfolioData") or []
        if not rows:
            print(f"[loader] NSE direct: no rows in response. keys={list(js.keys())}")
            return None

        df = pd.DataFrame(rows)
        df.columns = [c.strip().lower() for c in df.columns]

        # Typical columns: EOD_TIMESTAMP, EOD_OPEN_INDEX_VAL, EOD_HIGH_INDEX_VAL, EOD_LOW_INDEX_VAL, EOD_CLOSE_INDEX_VAL
        rename_map = {
            "eod_timestamp": "date",
            "eod_open_index_val": "open",
            "eod_high_index_val": "high",
            "eod_low_index_val": "low",
            "eod_close_index_val": "close",
        }
        df = df.rename(columns=rename_map)

        if "date" not in df.columns:
            print(f"[loader] NSE direct: unexpected columns {list(df.columns)}")
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].set_index("date").sort_index()
        df["volume"] = 0

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        print(f"[loader] NSE direct OK: {len(df)} rows")
        return df[REQUIRED_COLS]

    except Exception as e:
        print(f"[loader] NSE direct failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 3: yfinance (last resort)
# ---------------------------------------------------------------------------
def _download_yahoo_with_retry(start, end, interval, max_attempts=2):
    import yfinance as yf
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[loader] yahoo attempt {attempt}/{max_attempts} ...")
            df = yf.download(
                "^NSEI", start=start, end=end, interval=interval,
                auto_adjust=False, progress=False, threads=False,
            )
            if df is not None and len(df) > 0:
                print(f"[loader] yahoo OK: {len(df)} rows")
                return df
            print(f"[loader] yahoo attempt {attempt}: empty")
        except Exception as e:
            last_err = e
            print(f"[loader] yahoo attempt {attempt} err: {e}")
        if attempt < max_attempts:
            time.sleep(3 * attempt)
    print(f"[loader] yahoo exhausted. last_err={last_err}")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_nifty(start="2007-01-01", end=None, interval="1d", force=False):
    _ensure_parquet_engine()

    if not force:
        cached = _read_cache(interval)
        if cached is not None and len(cached) > 0:
            print(f"[loader] cache hit: {len(cached)} rows")
            return cached

    candidates = [
        ("nselib",      lambda: _download_nselib(start, end)),
        ("nse_direct",  lambda: _download_nse_direct(start, end)),
        ("yahoo",       lambda: _download_yahoo_with_retry(start, end, interval)),
    ]

    raw = None
    for name, fn in candidates:
        print(f"[loader] trying source: {name}")
        try:
            raw = fn()
        except Exception as e:
            print(f"[loader] {name} raised: {type(e).__name__}: {e}")
            raw = None
        if raw is not None and len(raw) > 0:
            print(f"[loader] source OK: {name}")
            break
        print(f"[loader] {name} yielded nothing, trying next ...")

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "[loader] all data sources failed (nselib, nse_direct, yahoo). "
            "This is usually temporary — retry in a few minutes."
        )

    df = _normalize(raw)
    _write_cache(df, interval)
    print(f"[loader] saved {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")
    return df


if __name__ == "__main__":
    print(load_nifty(force=True).tail())
