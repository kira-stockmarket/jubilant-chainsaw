import time
import os
from pathlib import Path

import pandas as pd
import requests

# Fresh CA bundle for GH runners
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
def _ensure_parquet_engine():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyarrow is required. Add pyarrow to requirements.txt. " + str(e)
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
        raise ValueError(f"[loader] missing columns: {missing}. Got: {list(df.columns)}")

    df = df[REQUIRED_COLS].dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Cache
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
# Source 1: NSE India direct API (primary)
# ---------------------------------------------------------------------------
def _nse_session():
    """Open a requests session with NSE cookies warm."""
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)
    except Exception as e:
        print(f"[loader] NSE warm-up warning: {e}")
    return s


def _download_nse_direct(start, end):
    try:
        print("[loader] NSE direct: requesting ...")
        from_dt = pd.Timestamp(start).strftime("%d-%m-%Y")
        to_dt = (pd.Timestamp(end) if end else pd.Timestamp.today()).strftime("%d-%m-%Y")

        url = (
            "https://www.nseindia.com/api/historical/indicesHistory"
            f"?indexType=NIFTY%2050&from={from_dt}&to={to_dt}"
        )

        s = _nse_session()
        r = s.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()

        data = js.get("data", {})
        rows = (
            data.get("indexCloseOnlineRecords")
            or data.get("indexPortfolioData")
            or []
        )
        if not rows:
            print(f"[loader] NSE direct: empty. top-level keys={list(js.keys())}, "
                  f"data keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}")
            return None

        df = pd.DataFrame(rows)
        df.columns = [c.strip().lower() for c in df.columns]

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

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = 0

        print(f"[loader] NSE direct OK: {len(df)} rows")
        return df[REQUIRED_COLS]

    except Exception as e:
        print(f"[loader] NSE direct failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 2: yfinance (fallback)
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
        ("nse_direct", lambda: _download_nse_direct(start, end)),
        ("yahoo",      lambda: _download_yahoo_with_retry(start, end, interval)),
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
            "[loader] all data sources failed (nse_direct, yahoo). "
            "Retry in a few minutes."
        )

    df = _normalize(raw)
    _write_cache(df, interval)
    print(f"[loader] saved {len(df)} rows "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


if __name__ == "__main__":
    print(load_nifty(force=True).tail())
